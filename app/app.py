"""广播剧混音审听台 —— Flask 后端。

声音导演交片前逐段审听：
- 上传同一集的两条等长、同采样率的 16 位单声道 PCM WAV（轨 A / 轨 B）
- 按采样编号为每轨设置/清除静音区间（半开区间 [start, end)，单位：采样）
- 两轨相加并限幅到 16 位得到混音；混音按 5 秒一段（末段可不足 5 秒）
- 逐段试听并确认通过；确认时校验"刚试听的 PCM"与该段当前内容一致（哈希）
- 保存静音改动后，仅混音样本值实际变化的段需要重听；保存失败则原混音与通过状态不变
- 全部段落通过后才可下载整集 WAV；下载按段流式生成，内存中最多暂存一段混音 PCM
- 下载开始后即快照当时通过的混音，浏览器停读/断连不影响他人保存改动
"""

import hashlib
import os
import struct
import threading
import wave

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

SEGMENT_SECONDS = 5
INT16_MIN, INT16_MAX = -32768, 32767
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", os.path.normpath(os.path.join(BASE_DIR, "..", "samples")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ---------------------------------------------------------------------------
# WAV 读取与混音（纯函数，便于下载快照在无锁状态下使用）
# ---------------------------------------------------------------------------

def read_wav_16mono(fileobj):
    """读取并校验 16 位单声道 PCM WAV，返回 (int16 numpy 数组, 采样率)。"""
    try:
        with wave.open(fileobj, "rb") as w:
            if w.getcomptype() != "NONE":
                raise ValueError("只支持未压缩的 PCM WAV")
            if w.getnchannels() != 1:
                raise ValueError(f"只支持单声道（该文件为 {w.getnchannels()} 声道）")
            if w.getsampwidth() != 2:
                raise ValueError(f"只支持 16 位采样（该文件为 {w.getsampwidth() * 8} 位）")
            rate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
    except wave.Error as exc:
        raise ValueError(f"无法解析 WAV 文件：{exc}") from exc
    data = np.frombuffer(raw, dtype="<i2").copy()
    if data.size != nframes:
        raise ValueError("WAV 数据长度与文件头不符")
    return data, rate


def apply_mutes(buf, mutes, seg_start):
    """把绝对采样坐标系下的静音区间应用到段缓冲（int32）上。"""
    seg_end = seg_start + buf.shape[0]
    for s, e in mutes:
        lo = max(s, seg_start) - seg_start
        hi = min(e, seg_end) - seg_start
        if hi > lo:
            buf[lo:hi] = 0


def mix_segment_pcm(track_a, track_b, mutes_a, mutes_b, start, end):
    """生成 [start, end) 区间的混音 PCM 字节：静音置零 -> 相加 -> 限幅到 16 位。"""
    sa = track_a[start:end].astype(np.int32)
    sb = track_b[start:end].astype(np.int32)
    apply_mutes(sa, mutes_a, start)
    apply_mutes(sb, mutes_b, start)
    mixed = sa + sb
    np.clip(mixed, INT16_MIN, INT16_MAX, out=mixed)
    return mixed.astype("<i2").tobytes()


def pcm_hash(pcm):
    return hashlib.sha256(pcm).hexdigest()


def wav_header(num_samples, sample_rate):
    """44 字节规范 PCM WAV 头（单声道 16 位）。"""
    data_size = num_samples * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )


def compute_seg_hashes(track_a, track_b, mutes, sample_rate, num_samples):
    """计算全部段落的当前混音哈希。"""
    seg_len = sample_rate * SEGMENT_SECONDS
    hashes = []
    for start in range(0, num_samples, seg_len):
        end = min(start + seg_len, num_samples)
        pcm = mix_segment_pcm(track_a, track_b, mutes["A"], mutes["B"], start, end)
        hashes.append(pcm_hash(pcm))
    return hashes


# ---------------------------------------------------------------------------
# 剧集状态（单剧集、跨窗口共享；所有可变字段仅在锁内读写）
# ---------------------------------------------------------------------------

class Episode:
    def __init__(self):
        self.lock = threading.RLock()
        self.loaded = False
        self.sample_rate = 0
        self.num_samples = 0
        self.tracks = {"A": None, "B": None}   # int16 numpy 数组，载入后不再原地修改
        self.mutes = {"A": [], "B": []}        # [(start, end), ...]，保存时整体替换
        self.seg_hashes = []                   # 每段当前混音哈希
        self.confirmed_hashes = []             # 每段"曾被确认通过"的哈希集合（持久保留）
        self.version = 0                       # 上传/静音保存时递增，供前端发现外部修改

    def seg_len(self):
        return self.sample_rate * SEGMENT_SECONDS

    def is_passed(self, i):
        # 当前混音内容曾被试听确认过即视为通过：静音改动后内容未变的段自动保持
        # 通过；内容改回某个已确认版本时也自动恢复通过。
        return self.seg_hashes[i] in self.confirmed_hashes[i]

    def all_passed(self):
        return bool(self.seg_hashes) and all(self.is_passed(i) for i in range(len(self.seg_hashes)))

    def to_dict(self):
        """调用方须持有锁。"""
        seg_len = self.seg_len()
        segments = []
        for i, h in enumerate(self.seg_hashes):
            start = i * seg_len
            end = min(start + seg_len, self.num_samples)
            segments.append({
                "index": i,
                "start_sample": start,
                "end_sample": end,
                "t0": start / self.sample_rate,
                "t1": end / self.sample_rate,
                "hash": h,
                "passed": self.is_passed(i),
            })
        return {
            "loaded": True,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "duration": self.num_samples / self.sample_rate,
            "segment_seconds": SEGMENT_SECONDS,
            "segments": segments,
            "all_passed": self.all_passed(),
            "mutes": {"A": [list(r) for r in self.mutes["A"]],
                      "B": [list(r) for r in self.mutes["B"]]},
            "version": self.version,
        }


EP = Episode()


def install_episode(track_a, track_b, rate):
    """校验通过后整体替换剧集；返回 (状态 dict, None) 或 (None, (响应, 状态码))。"""
    if track_a.shape[0] == 0:
        return None, (jsonify({"error": "音频为空"}), 400)
    hashes = compute_seg_hashes(track_a, track_b, {"A": [], "B": []}, rate, track_a.shape[0])
    with EP.lock:
        EP.tracks = {"A": track_a, "B": track_b}
        EP.sample_rate = rate
        EP.num_samples = int(track_a.shape[0])
        EP.mutes = {"A": [], "B": []}
        EP.seg_hashes = hashes
        EP.confirmed_hashes = [set() for _ in hashes]
        EP.loaded = True
        EP.version += 1
        return EP.to_dict(), None


# ---------------------------------------------------------------------------
# 页面与 API
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with EP.lock:
        if not EP.loaded:
            return jsonify({"loaded": False})
        return jsonify(EP.to_dict())


@app.post("/api/upload")
def api_upload():
    if "track_a" not in request.files or "track_b" not in request.files:
        return jsonify({"error": "需要同时上传 track_a 与 track_b 两个 WAV 文件"}), 400
    try:
        track_a, rate_a = read_wav_16mono(request.files["track_a"])
    except ValueError as exc:
        return jsonify({"error": f"轨 A：{exc}"}), 400
    try:
        track_b, rate_b = read_wav_16mono(request.files["track_b"])
    except ValueError as exc:
        return jsonify({"error": f"轨 B：{exc}"}), 400
    if rate_a != rate_b:
        return jsonify({"error": f"两轨采样率不一致（{rate_a} Hz vs {rate_b} Hz）"}), 400
    if track_a.shape[0] != track_b.shape[0]:
        return jsonify({"error": f"两轨长度不一致（{track_a.shape[0]} vs {track_b.shape[0]} 采样）"}), 400
    state, err = install_episode(track_a, track_b, rate_a)
    if err:
        return err
    return jsonify(state)


@app.post("/api/load_sample")
def api_load_sample():
    try:
        with open(os.path.join(SAMPLES_DIR, "sample_a.wav"), "rb") as f:
            track_a, rate_a = read_wav_16mono(f)
        with open(os.path.join(SAMPLES_DIR, "sample_b.wav"), "rb") as f:
            track_b, rate_b = read_wav_16mono(f)
    except FileNotFoundError:
        return jsonify({"error": "样例文件缺失（samples/sample_a.wav, sample_b.wav）"}), 500
    except ValueError as exc:
        return jsonify({"error": f"样例文件损坏：{exc}"}), 500
    state, err = install_episode(track_a, track_b, rate_a)
    if err:
        return err
    return jsonify(state)


def _as_int(value):
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError


@app.post("/api/mutes")
def api_save_mutes():
    """保存静音改动。任一区间非法则整体失败，原混音与通过状态保持不变。

    请求体：{"A": [[start, end], ...], "B": [...]}；缺省的轨保持原样。
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "请求体须为 JSON 对象"}), 400
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        tracks = EP.tracks
        rate = EP.sample_rate
        n = EP.num_samples
        cur_mutes = {"A": list(EP.mutes["A"]), "B": list(EP.mutes["B"])}

    new_mutes = {}
    for t in ("A", "B"):
        if t not in body:
            new_mutes[t] = cur_mutes[t]
            continue
        raw_list = body[t]
        if not isinstance(raw_list, list):
            return jsonify({"error": f"轨 {t} 的静音列表格式错误"}), 400
        ranges = []
        for item in raw_list:
            try:
                s, e = _as_int(item[0]), _as_int(item[1])
            except (ValueError, TypeError, IndexError):
                return jsonify({"error": f"轨 {t} 存在非法的静音区间：{item!r}"}), 400
            if not (0 <= s < e <= n):
                return jsonify({"error": f"轨 {t} 静音区间 [{s}, {e}) 须满足 0 ≤ 起 < 止 ≤ {n}"}), 400
            ranges.append((s, e))
        ranges.sort()
        new_mutes[t] = ranges

    # 先在锁外算好全部新哈希；任何异常都发生在状态被修改之前
    new_hashes = compute_seg_hashes(tracks["A"], tracks["B"], new_mutes, rate, n)

    with EP.lock:
        if not EP.loaded or EP.tracks is not tracks:
            return jsonify({"error": "剧集已被其他窗口更换，请刷新后重试"}), 409
        old_hashes = EP.seg_hashes
        invalidated = [i for i in range(len(new_hashes))
                       if old_hashes[i] in EP.confirmed_hashes[i]
                       and new_hashes[i] not in EP.confirmed_hashes[i]]
        EP.mutes = new_mutes
        EP.seg_hashes = new_hashes
        EP.version += 1
        state = EP.to_dict()
    return jsonify({"ok": True, "invalidated": invalidated, "state": state})


@app.get("/api/segment/<int:idx>.wav")
def api_segment(idx):
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        nseg = len(EP.seg_hashes)
        if not (0 <= idx < nseg):
            return jsonify({"error": "段落不存在"}), 404
        tracks = EP.tracks
        mutes = {"A": list(EP.mutes["A"]), "B": list(EP.mutes["B"])}
        rate = EP.sample_rate
        n = EP.num_samples
    seg_len = rate * SEGMENT_SECONDS
    start = idx * seg_len
    end = min(start + seg_len, n)
    pcm = mix_segment_pcm(tracks["A"], tracks["B"], mutes["A"], mutes["B"], start, end)
    return Response(
        wav_header(end - start, rate) + pcm,
        mimetype="audio/wav",
        headers={
            "X-Mix-Hash": pcm_hash(pcm),
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="segment_{idx + 1}.wav"',
        },
    )


@app.post("/api/confirm/<int:idx>")
def api_confirm(idx):
    """确认通过：客户端回传刚试听到的混音哈希，须与该段当前内容一致。"""
    body = request.get_json(silent=True) or {}
    client_hash = str(body.get("hash", ""))
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        if not (0 <= idx < len(EP.seg_hashes)):
            return jsonify({"error": "段落不存在"}), 404
        current = EP.seg_hashes[idx]
        if client_hash != current:
            return jsonify({"error": "该段内容已变化，请重新试听后再确认",
                            "current_hash": current}), 409
        EP.confirmed_hashes[idx].add(client_hash)
        return jsonify(EP.to_dict())


@app.get("/api/download")
def api_download():
    """全部段落通过后才可下载。开始下载即快照当时通过的混音来源（轨道引用 +
    静音表），之后任何静音改动/重新上传都不影响本次下载；逐段生成，内存中
    最多暂存一段待发送的混音 PCM。"""
    with EP.lock:
        if not EP.loaded:
            return jsonify({"error": "尚未上传音频"}), 409
        if not EP.all_passed():
            return jsonify({"error": "仍有段落未通过审听，不能下载"}), 403
        tracks = EP.tracks            # 数组载入后不原地修改，引用即快照
        mutes_a = list(EP.mutes["A"])  # 静音表保存时整体替换，拷贝列表即快照
        mutes_b = list(EP.mutes["B"])
        rate = EP.sample_rate
        n = EP.num_samples

    seg_len = rate * SEGMENT_SECONDS
    track_a, track_b = tracks["A"], tracks["B"]

    def generate():
        yield wav_header(n, rate)
        for start in range(0, n, seg_len):
            end = min(start + seg_len, n)
            yield mix_segment_pcm(track_a, track_b, mutes_a, mutes_b, start, end)

    return Response(
        generate(),
        mimetype="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="episode_mix.wav"',
            "Content-Length": str(44 + n * 2),
            "Cache-Control": "no-store",
        },
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "文件过大（上限 512 MB）"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
