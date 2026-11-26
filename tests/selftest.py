"""进程内自测：Flask test_client 全流程验证。

运行：python3 tests/selftest.py
覆盖：上传校验 / 分段与末段不足 5 秒 / 试听哈希确认 / 静音保存的精确失效 /
保存失败保留原状态 / 确认竞态 / 下载门控 / 下载中改音仍交付旧混音 /
跨窗口共享状态 / 限幅正确性（与纯 Python 参考实现比对）。
"""

import io
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import app as app_module  # noqa: E402

EP = app_module.EP
client = app_module.app.test_client()

RATE = 8000
DUR = 12.0
N = int(RATE * DUR)          # 96000 -> 3 段：40000 / 40000 / 16000（末段 2 秒）
SEG = RATE * 5

FAILED = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, ("| " + str(extra) if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


def make_wav_bytes(arr, rate=RATE, sampwidth=2, nchannels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        if sampwidth == 2:
            w.writeframes(np.asarray(arr, dtype="<i2").tobytes())
        else:
            w.writeframes(bytes(arr))
    return buf.getvalue()


def upload(a_bytes, b_bytes):
    return client.post("/api/upload", data={
        "track_a": (io.BytesIO(a_bytes), "a.wav"),
        "track_b": (io.BytesIO(b_bytes), "b.wav"),
    }, content_type="multipart/form-data")


def ref_mix(a, b, mutes_a=(), mutes_b=()):
    """纯 Python 参考混音：静音置零 -> 相加 -> 限幅。"""
    out = np.empty(len(a), dtype=np.int64)
    for i in range(len(a)):
        va = 0 if any(s <= i < e for s, e in mutes_a) else int(a[i])
        vb = 0 if any(s <= i < e for s, e in mutes_b) else int(b[i])
        out[i] = max(-32768, min(32767, va + vb))
    return out.astype("<i2")


def get_state():
    r = client.get("/api/state")
    assert r.status_code == 200
    return r.get_json()


def save_mutes(payload):
    return client.post("/api/mutes", json=payload)


def listen(i):
    r = client.get(f"/api/segment/{i}.wav")
    assert r.status_code == 200, r.status_code
    return r.headers["X-Mix-Hash"], r.data


def confirm(i, h):
    return client.post(f"/api/confirm/{i}", json={"hash": h})


def pass_all(nseg):
    for i in range(nseg):
        h, _ = listen(i)
        r = confirm(i, h)
        assert r.status_code == 200, (i, r.status_code, r.get_json())


def parse_wav(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return params, pcm


# ---------------------------------------------------------------- 测试数据
A = np.full(N, 1000, dtype=np.int16)
A[40000:41000] = 0            # 段 1 内的静音区（用于"静音了无声音样本 -> 无变化"）
A[5000:5100] = 30000          # 段 0 内的限幅区
B = np.full(N, 500, dtype=np.int16)
B[5000:5100] = 20000          # 30000+20000=50000 -> 限幅 32767
B[60000:60500] = 3000         # 段 1 内可辨识的内容
WAV_A = make_wav_bytes(A)
WAV_B = make_wav_bytes(B)

print("== 0. 未上传时的行为 ==")
check("state 未载入", get_state() == {"loaded": False})
check("未上传时下载 -> 409", client.get("/api/download").status_code == 409)
check("未上传时保存静音 -> 409", save_mutes({"A": []}).status_code == 409)
check("未上传时试听 -> 409", client.get("/api/segment/0.wav").status_code == 409)

print("== 1. 上传校验 ==")
stereo = make_wav_bytes(np.zeros(1000, dtype=np.int16), nchannels=2)
r = upload(stereo, WAV_B)
check("立体声 -> 400", r.status_code == 400, r.get_json())
eight_bit = make_wav_bytes([128] * N, sampwidth=1)
r = upload(eight_bit, WAV_B)
check("8 位 -> 400", r.status_code == 400, r.get_json())
r = upload(make_wav_bytes(A, rate=16000), WAV_B)
check("采样率不一致 -> 400", r.status_code == 400, r.get_json())
r = upload(make_wav_bytes(A[:50000]), WAV_B)
check("长度不一致 -> 400", r.status_code == 400, r.get_json())
r = upload(WAV_A, WAV_B)
check("合法上传 -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("3 段", len(st["segments"]) == 3, st["segments"])
check("末段不足 5 秒", st["segments"][2]["end_sample"] - st["segments"][2]["start_sample"] == 16000)
check("初始均未通过", not any(s["passed"] for s in st["segments"]) and not st["all_passed"])

print("== 2. 下载门控 ==")
check("未全部通过时下载 -> 403", client.get("/api/download").status_code == 403)

print("== 3. 试听与确认 ==")
check("段越界试听 -> 404", client.get("/api/segment/99.wav").status_code == 404)
check("段越界确认 -> 404", confirm(99, "x").status_code == 404)
h0, wav0 = listen(0)
params, pcm0 = parse_wav(wav0)
check("段 0 WAV 参数", params == (1, 2, RATE, SEG), params)
exp = ref_mix(A, B)
check("段 0 PCM 与参考混音一致", np.array_equal(pcm0, exp[:SEG]))
check("限幅正确", pcm0[5000:5100].max() == 32767 and (exp[5000:5100] == 32767).all())
h2, wav2 = listen(2)
params2, pcm2 = parse_wav(wav2)
check("末段 16000 帧", params2[3] == 16000, params2)
check("错误哈希确认 -> 409", confirm(0, "deadbeef").status_code == 409)
check("正确哈希确认 -> 200", confirm(0, h0).status_code == 200)
check("段 0 已通过", get_state()["segments"][0]["passed"])

print("== 4. 静音保存：无实际样本变化则不失效 ==")
r = save_mutes({"A": [[40000, 41000]]})   # 该区域本来就是 0
check("保存成功", r.status_code == 200, r.get_json())
d = r.get_json()
check("无段失效", d["invalidated"] == [], d["invalidated"])
check("段 0 仍通过", get_state()["segments"][0]["passed"])

print("== 5. 静音保存：仅实际变化的段失效 ==")
pass_all(3)
check("全部通过", get_state()["all_passed"])
r = save_mutes({"A": [[45000, 46000]]})   # 段 1 内、有声音的区域
d = r.get_json()
check("仅段 1 失效", d["invalidated"] == [1], d["invalidated"])
st = get_state()
check("段 0/2 保持通过", st["segments"][0]["passed"] and st["segments"][2]["passed"])
check("段 1 需重听", not st["segments"][1]["passed"])

print("== 6. 保存失败保留原混音与通过状态 ==")
before = get_state()
for bad in ({"A": [[-1, 50]]}, {"A": [[100, 50]]}, {"A": [[0, N + 1]]},
            {"A": [["x", 50]]}, {"A": "notalist"}, {"A": [[1.5, 50]]}):
    r = save_mutes(bad)
    check(f"非法静音 {bad} -> 400", r.status_code == 400, r.get_json())
after = get_state()
check("失败后状态不变",
      before["mutes"] == after["mutes"]
      and [s["passed"] for s in before["segments"]] == [s["passed"] for s in after["segments"]]
      and [s["hash"] for s in before["segments"]] == [s["hash"] for s in after["segments"]])

print("== 7. 确认竞态：试听后内容被改 -> 确认被拒 ==")
h1, _ = listen(1)
r = save_mutes({"A": [[45000, 47000]]})   # 改变段 1
check("改音保存成功", r.status_code == 200)
r = confirm(1, h1)
check("旧哈希确认 -> 409", r.status_code == 409, r.get_json())
h1_new, _ = listen(1)
check("重新试听后可确认", confirm(1, h1_new).status_code == 200)

print("== 8. 清除静音后内容回到已确认版本 -> 自动恢复通过 ==")
r = save_mutes({"A": []})
check("清除静音成功", r.status_code == 200)
st = get_state()
check("段 1 内容复原且自动恢复通过", st["segments"][1]["passed"], st["segments"][1])
check("全部通过", st["all_passed"])

print("== 9. 下载内容与限幅 ==")
r = client.get("/api/download")
check("下载 -> 200", r.status_code == 200)
check("Content-Length 正确", r.headers.get("Content-Length") == str(44 + N * 2),
      r.headers.get("Content-Length"))
params, pcm = parse_wav(r.data)
check("整集 WAV 参数", params == (1, 2, RATE, N), params)
check("整集 PCM 与参考混音一致", np.array_equal(pcm, ref_mix(A, B)))

print("== 10. 下载中改音：已开始的下载继续交付当时通过的混音 ==")
r = upload(WAV_A, WAV_B)
assert r.status_code == 200
pass_all(3)
rv = client.get("/api/download", buffered=False)
it = iter(rv.response)
header = next(it)
seg0 = next(it)
check("流式首块为 44 字节头", len(header) == 44, len(header))
check("第二块为段 0", seg0 == ref_mix(A, B)[:SEG].tobytes())
r = save_mutes({"B": [[60000, 60500]]})   # 改变段 1 的混音
check("下载进行中保存静音 -> 200", r.status_code == 200, r.get_json())
seg1 = next(it)
seg2 = next(it)
try:
    next(it)
    tail = "还有多余数据"
except StopIteration:
    tail = None
check("下载恰好 4 块", tail is None)
check("段 1 仍交付改音前的混音", seg1 == ref_mix(A, B)[SEG:2 * SEG].tobytes())
check("段 2 正确", seg2 == ref_mix(A, B)[2 * SEG:].tobytes())
_, wav1_now = listen(1)
_, pcm1_now = parse_wav(wav1_now)
check("当前段 1 已是新混音（与下载内容不同）",
      not np.array_equal(pcm1_now, ref_mix(A, B)[SEG:2 * SEG]))
rv.close()

print("== 11. 跨窗口共享状态 ==")
c1 = app_module.app.test_client()
c2 = app_module.app.test_client()
r = c1.post("/api/upload", data={
    "track_a": (io.BytesIO(WAV_A), "a.wav"),
    "track_b": (io.BytesIO(WAV_B), "b.wav"),
}, content_type="multipart/form-data")
check("窗口1 上传 -> 200", r.status_code == 200)
st2 = c2.get("/api/state").get_json()
check("窗口2 立即可见", st2["loaded"] and len(st2["segments"]) == 3)
h, _ = (lambda resp: (resp.headers["X-Mix-Hash"], resp.data))(c2.get("/api/segment/0.wav"))
check("窗口2 确认段 0", c2.post("/api/confirm/0", json={"hash": h}).status_code == 200)
check("窗口1 看到段 0 通过", c1.get("/api/state").get_json()["segments"][0]["passed"])
r = c1.post("/api/mutes", json={"B": [[60000, 60500]]})
check("窗口1 改静音 -> 200", r.status_code == 200)
st2 = c2.get("/api/state").get_json()
check("窗口2 看到新静音与失效",
      st2["mutes"]["B"] == [[60000, 60500]] and st2["segments"][1]["passed"] is False)

print("== 12. 重新上传重置状态 ==")
r = upload(WAV_A, WAV_B)
st = get_state()
check("重传后静音清空、全部未通过",
      st["mutes"] == {"A": [], "B": []} and not any(s["passed"] for s in st["segments"]))

print()
if FAILED:
    print(f"共 {len(FAILED)} 项失败：")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print(f"全部自测通过。")
