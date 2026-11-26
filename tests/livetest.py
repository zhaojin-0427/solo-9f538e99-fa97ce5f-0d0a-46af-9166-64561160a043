"""真实 HTTP 环境自测：多线程服务器 + 原始 http.client。

验证：
1. 浏览器停读（下载挂着不读）时仍可保存静音改动；
2. 下载进行中改音，已开始的下载继续交付当时通过的混音；
3. 下载中途断连后服务器正常服务；
4. 跨"窗口"（独立连接）状态一致。

运行：python3 tests/livetest.py
"""

import http.client
import io
import json
import os
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import app as app_module  # noqa: E402

HOST, PORT = "127.0.0.1", 8971
RATE, DUR = 8000, 12.0
N = int(RATE * DUR)
SEG = RATE * 5

FAILED = []


def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, ("| " + str(extra) if extra and not cond else ""))
    if not cond:
        FAILED.append(name)


def wav_bytes(arr, rate=RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(arr, dtype="<i2").tobytes())
    return buf.getvalue()


def multipart(fields):
    boundary = "----livetestboundary"
    body = b""
    for name, fname, data in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{fname}\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def conn():
    return http.client.HTTPConnection(HOST, PORT, timeout=10)


def request(method, path, body=None, headers=None):
    c = conn()
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def read_exact(resp, n):
    buf = b""
    while len(buf) < n:
        chunk = resp.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def main():
    from werkzeug.serving import make_server
    srv = make_server(HOST, PORT, app_module.app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)

    a = np.full(N, 1000, dtype=np.int16)
    b = np.full(N, 500, dtype=np.int16)
    b[60000:60500] = 3000
    wa, wb = wav_bytes(a), wav_bytes(b)

    body, ctype = multipart([("track_a", "a.wav", wa), ("track_b", "b.wav", wb)])
    status, _, data = request("POST", "/api/upload", body, {"Content-Type": ctype})
    check("上传 -> 200", status == 200, data[:200])

    # 全部段试听 + 确认
    status, _, data = request("GET", "/api/state")
    st = json.loads(data)
    for seg in st["segments"]:
        status, headers, _ = request("GET", f"/api/segment/{seg['index']}.wav")
        h = headers.get("X-Mix-Hash")
        status, _, data = request("POST", f"/api/confirm/{seg['index']}",
                                  json.dumps({"hash": h}), {"Content-Type": "application/json"})
        assert status == 200, (seg["index"], status, data)
    status, _, data = request("GET", "/api/state")
    check("全部通过", json.loads(data)["all_passed"])

    # 开始下载，只读头 + 第 1 段，然后停读（连接保持打开）
    dl = conn()
    dl.request("GET", "/api/download")
    resp = dl.getresponse()
    check("下载 -> 200", resp.status == 200)
    head_and_seg0 = read_exact(resp, 44 + SEG * 2)
    check("已读头+段0", len(head_and_seg0) == 44 + SEG * 2, len(head_and_seg0))

    # 下载挂着不读，另一连接保存静音（改第 2 段内容）——必须能立即成功
    t0 = time.time()
    status, _, data = request("POST", "/api/mutes",
                              json.dumps({"B": [[60000, 60500]]}),
                              {"Content-Type": "application/json"})
    dt = time.time() - t0
    check("停读期间保存静音 -> 200", status == 200, data[:200])
    check("保存响应迅速(<5s)", dt < 5, f"{dt:.2f}s")

    # 继续读完下载：第 2 段应仍是改音前的混音（快照）
    rest = read_exact(resp, 1 << 30)
    dl.close()
    seg1 = rest[:SEG * 2]
    expected = (a[SEG:2 * SEG].astype(np.int32) + b[SEG:2 * SEG].astype(np.int32))
    np.clip(expected, -32768, 32767, out=expected)
    check("下载中段 1 仍为旧混音", seg1 == expected.astype("<i2").tobytes())
    check("下载总长正确", len(head_and_seg0) + len(rest) == 44 + N * 2,
          len(head_and_seg0) + len(rest))

    # 当前段 1 已是新混音
    status, _, cur = request("GET", "/api/segment/1.wav")
    check("当前段 1 已是新混音", cur[44:] != seg1)

    # 重新试听确认段 1，恢复全部通过
    status, headers, _ = request("GET", "/api/segment/1.wav")
    status, _, data = request("POST", "/api/confirm/1",
                              json.dumps({"hash": headers.get("X-Mix-Hash")}),
                              {"Content-Type": "application/json"})
    status, _, data = request("GET", "/api/state")
    check("重新确认后全部通过", json.loads(data)["all_passed"])

    # 断连：开始新下载（这次是真 200），读一部分后直接关闭，服务器应继续正常服务
    dl2 = conn()
    dl2.request("GET", "/api/download")
    resp2 = dl2.getresponse()
    check("再次下载 -> 200", resp2.status == 200)
    read_exact(resp2, 100)
    dl2.close()
    time.sleep(0.3)
    status, _, data = request("GET", "/api/state")
    check("断连后服务器正常", status == 200 and json.loads(data)["loaded"])

    # 跨窗口：新连接看到的静音与另一连接保存的一致
    check("跨窗口静音一致", json.loads(data)["mutes"]["B"] == [[60000, 60500]])

    # ------------------------------------------------------------------
    # 双窗口并发：分别改 A / B 轨，交错提交后两轨改动都生效（互不覆盖）
    # 先重传重置，保证两轨版本都从 0、静音表都为空开始
    # ------------------------------------------------------------------
    body, ctype = multipart([("track_a", "a.wav", wa), ("track_b", "b.wav", wb)])
    status, _, data = request("POST", "/api/upload", body, {"Content-Type": ctype})
    assert status == 200, data[:200]
    st = json.loads(request("GET", "/api/state")[2])
    check("重传后每轨版本号为 0", st["track_versions"] == {"A": 0, "B": 0})

    results = {}

    def save_async(name, payload, base):
        body = dict(payload)
        body["base_versions"] = base
        status, _, resp = request("POST", "/api/mutes", json.dumps(body),
                                  {"Content-Type": "application/json"})
        results[name] = (status, json.loads(resp))

    t_a = threading.Thread(target=save_async,
                           args=("A", {"A": [[10000, 11000]]}, {"A": 0}))
    t_b = threading.Thread(target=save_async,
                           args=("B", {"B": [[70000, 71000]]}, {"B": 0}))
    t_a.start(); t_b.start(); t_a.join(); t_b.join()
    check("两窗口并发保存均成功",
          all(v[0] == 200 for v in results.values()), results)
    st = json.loads(request("GET", "/api/state")[2])
    check("两轨改动合并生效",
          st["mutes"] == {"A": [[10000, 11000]], "B": [[70000, 71000]]}, st["mutes"])
    check("两轨版本各自只递增一次",
          st["track_versions"] == {"A": 1, "B": 1}, st["track_versions"])

    # 同轨冲突：两个窗口都基于 A=1 编辑，后者必须被整次拒绝且 B 不受影响
    status1, _, d1 = request("POST", "/api/mutes",
                             json.dumps({"A": [[12000, 13000]],
                                         "base_versions": {"A": 1}}),
                             {"Content-Type": "application/json"})
    check("窗口1 同轨先存 -> 200", status1 == 200, d1)
    status2, _, d2raw = request("POST", "/api/mutes",
                                json.dumps({"A": [[14000, 15000]],
                                            "base_versions": {"A": 1}}),
                                {"Content-Type": "application/json"})
    d2 = json.loads(d2raw)
    check("窗口2 过期基线 -> 409", status2 == 409, d2)
    check("标出冲突轨 A", d2.get("conflicts") == ["A"], d2)
    check("回带服务端静音与版本",
          d2.get("server_mutes", {}).get("A") == [[12000, 13000]]
          and d2.get("server_track_versions", {}).get("A") == 2, d2)
    check("保留客户端提交输入", d2.get("submitted_mutes", {}).get("A") == [[14000, 15000]])
    st = json.loads(request("GET", "/api/state")[2])
    check("拒绝后 A 为窗口1 内容、B 改动仍在",
          st["mutes"] == {"A": [[12000, 13000]], "B": [[70000, 71000]]}, st["mutes"])

    # 旧客户端：只提交一轨且不带 base_versions，另一轨改动必须保留
    status, _, d = request("POST", "/api/mutes",
                           json.dumps({"A": [[20000, 21000]]}),
                           {"Content-Type": "application/json"})
    check("无基线单轨旧调用 -> 200", status == 200, d)
    st = json.loads(request("GET", "/api/state")[2])
    check("旧调用不抹掉 B 轨改动",
          st["mutes"] == {"A": [[20000, 21000]], "B": [[70000, 71000]]}, st["mutes"])

    # 冲突刷新基线后重提成功，随后对合并后的实际 PCM 重新逐段试听确认
    cur_a = st["track_versions"]["A"]
    status, _, d = request("POST", "/api/mutes",
                           json.dumps({"A": [[14000, 15000]],
                                       "base_versions": {"A": cur_a}}),
                           {"Content-Type": "application/json"})
    check("刷新基线后重提 -> 200", status == 200, d)
    st = json.loads(request("GET", "/api/state")[2])
    for seg in st["segments"]:
        i = seg["index"]
        _, hdr, _ = request("GET", f"/api/segment/{i}.wav")
        sc, _, dd = request("POST", f"/api/confirm/{i}",
                            json.dumps({"hash": hdr.get("X-Mix-Hash")}),
                            {"Content-Type": "application/json"})
        assert sc == 200, (i, sc, dd)
    st = json.loads(request("GET", "/api/state")[2])
    check("合并混音逐段确认后全部通过", st["all_passed"])
    status, hdr, full = request("GET", "/api/download")
    check("新下载按更新后通过状态放行 -> 200", status == 200)
    check("下载长度与合并混音一致",
          hdr.get("Content-Length") == str(44 + N * 2) and len(full) == 44 + N * 2)

    srv.shutdown()
    print()
    if FAILED:
        print(f"共 {len(FAILED)} 项失败")
        sys.exit(1)
    print("真实 HTTP 环境自测全部通过。")


if __name__ == "__main__":
    main()
