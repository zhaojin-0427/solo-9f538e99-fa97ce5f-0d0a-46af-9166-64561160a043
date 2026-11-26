"""进程内自测：Flask test_client 全流程验证。

运行：python3 tests/selftest.py
覆盖：上传校验 / 分段与末段不足 5 秒 / 试听哈希确认 / 静音保存的精确失效 /
保存失败保留原状态 / 确认竞态 / 下载门控 / 下载中改音仍交付旧混音 /
跨窗口共享状态 / 限幅正确性（与纯 Python 参考实现比对）。
"""

import io
import os
import sys
import time
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

import threading  # noqa: E402


print("== 13. 按轨版本号：只提交一轨不影响另一轨版本 ==")
st = get_state()
check("初始每轨版本为 0", st["track_versions"] == {"A": 0, "B": 0})
r = save_mutes({"A": [[10, 20]], "base_versions": {"A": 0}})
check("带基线保存 A -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("仅 A 版本递增", st["track_versions"] == {"A": 1, "B": 0}, st["track_versions"])
check("A 静音已保存、B 仍为空", st["mutes"] == {"A": [[10, 20]], "B": []})
r = save_mutes({"B": [[30, 40]], "base_versions": {"B": 0}})
check("带基线保存 B -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("两轨版本各自递增", st["track_versions"] == {"A": 1, "B": 1}, st["track_versions"])
check("两轨静音并存（后存不抹先存）",
      st["mutes"] == {"A": [[10, 20]], "B": [[30, 40]]}, st["mutes"])

print("== 14. 同轨冲突：基线过期整次拒绝，标出冲突轨并保留输入 ==")
# 窗口 1 与窗口 2 都基于版本 1 编辑 A 轨；窗口 1 先存成功，窗口 2 必须被拒
r = save_mutes({"A": [[11, 22]], "base_versions": {"A": 1}})
check("窗口1 先存 A -> 200", r.status_code == 200, r.get_json())
v_a_win1 = r.get_json()["state"]["track_versions"]["A"]   # 现在 A 的最新版本
before = get_state()
r = save_mutes({"A": [[33, 44]], "base_versions": {"A": 1}})
check("窗口2 过期基线 -> 409", r.status_code == 409, r.get_json())
d = r.get_json()
check("冲突轨为 A", d.get("conflicts") == ["A"], d)
check("响应回带服务端当前静音", d.get("server_mutes", {}).get("A") == [[11, 22]], d)
check("响应回带服务端版本", d.get("server_track_versions", {}).get("A") == v_a_win1, d)
check("响应保留客户端提交输入", d.get("submitted_mutes", {}).get("A") == [[33, 44]], d)
after = get_state()
check("拒绝后服务端状态不变",
      before["mutes"] == after["mutes"] and before["track_versions"] == after["track_versions"])
# 连带提交另一轨时同样整次拒绝，另一轨也不得落库
r = save_mutes({"A": [[50, 60]], "B": [[70, 80]],
                "base_versions": {"A": 1, "B": 1}})
check("同轨冲突时连带的 B 提交也被整次拒绝 -> 409", r.status_code == 409, r.get_json())
after = get_state()
check("B 未落库", after["mutes"]["B"] == [[30, 40]], after["mutes"])
check("A 仍为窗口1 的内容", after["mutes"]["A"] == [[11, 22]], after["mutes"])
check("刷新基线后可正常保存",
      save_mutes({"A": [[33, 44]], "base_versions": {"A": v_a_win1}}).status_code == 200)

print("== 15. 双窗口交错保存：锁外重算期间另一轨先提交，合并两边改动 ==")
r = upload(WAV_A, WAV_B)
assert r.status_code == 200
real_compute = app_module.compute_seg_hashes
entered = threading.Event()
release = threading.Event()


def blocking_compute(track_a, track_b, mutes, rate, n):
    entered.set()
    release.wait(5)
    return real_compute(track_a, track_b, mutes, rate, n)


app_module.compute_seg_hashes = blocking_compute
try:
    t1 = {}

    def win1():
        # 窗口1 改 A 轨（无冲突基线），其哈希重算在锁外被阻塞
        t1["resp"] = save_mutes({"A": [[1000, 2000]], "base_versions": {"A": 0}})

    th = threading.Thread(target=win1)
    th.start()
    assert entered.wait(5), "窗口1 的重算未在预期时间内开始"
    # 此时窗口1 已释放锁：窗口2 改 B 轨并先提交成功
    r2 = save_mutes({"B": [[3000, 4000]], "base_versions": {"B": 0}})
    check("重算期间窗口2 保存 B -> 200", r2.status_code == 200, r2.get_json())
    st_mid = get_state()
    check("窗口2 提交后 B 已生效", st_mid["mutes"]["B"] == [[3000, 4000]])
    check("窗口1 尚未提交（重算阻塞中）", st_mid["mutes"]["A"] == [])
    release.set()
    th.join(5)
    check("窗口1 线程结束", not th.is_alive())
    check("窗口1 重试合并后 -> 200", t1["resp"].status_code == 200, t1["resp"].get_json())
finally:
    app_module.compute_seg_hashes = real_compute
st = get_state()
check("两轨改动都生效（A∪B 合并）",
      st["mutes"] == {"A": [[1000, 2000]], "B": [[3000, 4000]]}, st["mutes"])
check("两轨版本均为 1", st["track_versions"] == {"A": 1, "B": 1}, st["track_versions"])
# 合并后的实际混音与参考实现一致
_, wav_seg0 = listen(0)
_, pcm_seg0 = parse_wav(wav_seg0)
check("合并后段0混音与参考实现一致",
      np.array_equal(pcm_seg0, ref_mix(A, B, [(1000, 2000)], [(3000, 4000)])[:SEG]))

print("== 16. 重算期间可试听；按当前 PCM 的确认保留，过期确认被拒 ==")
r = upload(WAV_A, WAV_B)
assert r.status_code == 200
h_before, pcm_before = listen(1)
pcm_before = parse_wav(pcm_before)[1]
entered = threading.Event()
release = threading.Event()
app_module.compute_seg_hashes = lambda *a, **k: (entered.set(), release.wait(5),
                                                 real_compute(*a, **k))[2]
try:
    t1 = {}

    def win1():
        # 段1（40000..80000）内的 A 轨静音
        t1["resp"] = save_mutes({"A": [[45000, 46000]], "base_versions": {"A": 0}})

    th = threading.Thread(target=win1)
    th.start()
    assert entered.wait(5)
    # 重算未提交：试听拿到的仍是改前 PCM，且保存不被试听阻塞
    h_during, wav_during = listen(1)
    check("重算期间可试听且为改前内容",
          h_during == h_before and np.array_equal(parse_wav(wav_during)[1], pcm_before))
    r = confirm(1, h_during)
    check("重算期间确认当前 PCM -> 200", r.status_code == 200, r.get_json())
    r = confirm(1, "deadbeef")
    check("重算期间错误哈希确认 -> 409", r.status_code == 409)
    release.set()
    th.join(5)
    check("保存线程正常结束", t1["resp"].status_code == 200, t1["resp"].get_json())
finally:
    app_module.compute_seg_hashes = real_compute
st = get_state()
h_after = st["segments"][1]["hash"]
check("提交后段1哈希已变化", h_after != h_before)
check("段1因实际 PCM 变化而失效", not st["segments"][1]["passed"])
# 重算期间作出的确认仍记录在案：内容改回旧 PCM 时自动恢复通过
r = save_mutes({"A": []}, )  # 清除 A 静音（旧调用形式，不带基线）
check("无基线旧调用清除静音 -> 200", r.status_code == 200, r.get_json())
st = get_state()
check("内容复原后重算期间的确认仍有效、自动恢复通过", st["segments"][1]["passed"])
r = confirm(1, h_before)
check("过期哈希在内容已变场景仍被拒（当前内容复原后哈希相等则通过）",
      r.status_code == 200)

print("== 17. 两轨静音与审听状态合并后的下载门控 ==")
r = upload(WAV_A, WAV_B)
assert r.status_code == 200
# 两窗口分别改两轨并交错保存
save_mutes({"A": [[45000, 46000]], "base_versions": {"A": 0}})
save_mutes({"B": [[60000, 60500]], "base_versions": {"B": 0}})
st = get_state()
check("两轨静音合并生效",
      st["mutes"] == {"A": [[45000, 46000]], "B": [[60000, 60500]]}, st["mutes"])
check("合并后未全通过 -> 下载 403", client.get("/api/download").status_code == 403)
# 对合并后的实际 PCM 逐段试听确认
nseg = len(st["segments"])
for i in range(nseg):
    h, _ = listen(i)
    rr = confirm(i, h)
    assert rr.status_code == 200, (i, rr.get_json())
st = get_state()
check("合并混音全部确认后 all_passed", st["all_passed"])
r = client.get("/api/download")
check("新发起的下载按更新后通过状态放行 -> 200", r.status_code == 200)
params, pcm = parse_wav(r.data)
check("下载内容为两轨合并静音后的限幅混音",
      np.array_equal(pcm, ref_mix(A, B, [(45000, 46000)], [(60000, 60500)])))
check("下载长度正确", len(r.data) == 44 + N * 2)

print("== 18. 并发压测：多线程交错保存两轨，最终状态一致且无丢失 ==")
r = upload(WAV_A, WAV_B)
assert r.status_code == 200
ROUNDS = 25
errors = []


def hammer(track):
    # 模拟真实窗口：每次保存前从服务端读取该轨最新版本作为基线；
    # 遇同轨冲突 409 则人工解决（刷新基线后重提同一输入），遇合并退避 409 则直接重试。
    base = get_state()["track_versions"][track]
    for k in range(ROUNDS):
        payload = {track: [[100 + k, 200 + k]], "base_versions": {track: base}}
        for _attempt in range(20):
            rr = save_mutes(payload)
            if rr.status_code == 200:
                base = rr.get_json()["state"]["track_versions"][track]
                break
            j = rr.get_json()
            if rr.status_code == 409 and j.get("conflicts"):
                base = j["server_track_versions"][track]   # 人工解决：采纳新基线后重提
                payload["base_versions"][track] = base
            elif rr.status_code == 409:
                time.sleep(0.05)                            # 合并退避，直接重试
            else:
                errors.append((track, k, rr.status_code, j))
                return
        else:
            errors.append((track, k, "重试耗尽"))
            return


threads = [threading.Thread(target=hammer, args=("A",)),
           threading.Thread(target=hammer, args=("B",))]
for th in threads:
    th.start()
for th in threads:
    th.join(15)
check("压测无失败请求", not errors, errors[:3])
st = get_state()
check("压测后两轨静音都保留",
      st["mutes"] == {"A": [[100 + ROUNDS - 1, 200 + ROUNDS - 1]],
                      "B": [[100 + ROUNDS - 1, 200 + ROUNDS - 1]]}, st["mutes"])
check("每轨版本递增次数等于保存次数",
      st["track_versions"] == {"A": ROUNDS, "B": ROUNDS}, st["track_versions"])

print("== 19. 乐观并发参数校验 ==")
for bad in (
    {"base_versions": {"A": 0}},                                  # 未提交轨带基线
    {"A": [], "base_versions": {"C": 0}},                         # 未知轨
    {"A": [], "base_versions": {"A": -1}},                        # 负版本
    {"A": [], "base_versions": {"A": "x"}},                       # 非整数版本
    {"A": [], "base_versions": []},                               # 基线非对象
    {},                                                           # 无任何轨
):
    rr = save_mutes(bad)
    check(f"非法请求 {bad} -> 400", rr.status_code == 400, rr.get_json())
check("非法请求后状态不变", get_state()["track_versions"] == {"A": ROUNDS, "B": ROUNDS})

print("== 20. 重传后版本号重置 ==")
r = upload(WAV_A, WAV_B)
check("重传后每轨版本归零", get_state()["track_versions"] == {"A": 0, "B": 0})

print()
if FAILED:
    print(f"共 {len(FAILED)} 项失败：")
    for name in FAILED:
        print("  -", name)
    sys.exit(1)
print(f"全部自测通过。")
