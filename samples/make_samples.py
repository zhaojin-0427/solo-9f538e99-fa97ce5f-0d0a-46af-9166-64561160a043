"""生成可直接试听的样例：samples/sample_a.wav 与 samples/sample_b.wav。

- 27 秒 @ 44.1 kHz，16 位单声道 PCM（6 个审听段：5×5 秒 + 末段 2 秒）
- 轨 A：A 小调五声音阶旋律（正弦 + 泛音 + 指数衰减包络）
- 轨 B：低音根音进行（Am-F-C-G）+ 极轻的"机房底噪"
- 10.0–11.0 秒两轨同相同幅叠加，混音必然触发 16 位限幅（第 3 段）
"""

import os
import wave

import numpy as np

RATE = 44100
DUR = 27.0
N = int(RATE * DUR)

rng = np.random.default_rng(20261126)


def add_note(track, freq, start, dur, amp, decay=3.0):
    i0 = int(start * RATE)
    i1 = min(N, int((start + dur) * RATE))
    if i1 <= i0:
        return
    t = np.arange(i1 - i0) / RATE
    env = np.exp(-decay * t / dur)
    tone = (np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(4 * np.pi * freq * t)
            + 0.1 * np.sin(6 * np.pi * freq * t)) / 1.4
    track[i0:i1] += amp * env * tone


def to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def main():
    track_a = np.zeros(N)
    track_b = np.zeros(N)

    # 轨 A：五声音阶随机漫步旋律，每 0.45 秒一个音
    scale = [440.0, 523.25, 587.33, 659.25, 783.99, 880.0]
    pos = 2
    t = 0.0
    while t < DUR - 0.5:
        add_note(track_a, scale[pos], t, 0.5, 0.45)
        pos = int(np.clip(pos + rng.integers(-2, 3), 0, len(scale) - 1))
        t += 0.45

    # 轨 B：Am - F - C - G 低音进行，每 1.8 秒一个根音，外加轻底噪
    roots = [110.0, 87.31, 130.81, 98.0]
    t = 0.0
    k = 0
    while t < DUR:
        add_note(track_b, roots[k % 4], t, 1.7, 0.5, decay=1.5)
        add_note(track_b, roots[k % 4] * 1.5, t, 1.7, 0.18, decay=1.5)  # 五度
        t += 1.8
        k += 1
    track_b += 0.015 * rng.standard_normal(N)  # 机房底噪

    # 10.0–11.0 秒：两轨叠加同相 220 Hz 大幅正弦，混音必然限幅
    i0, i1 = int(10.0 * RATE), int(11.0 * RATE)
    tt = np.arange(i1 - i0) / RATE
    loud = 0.85 * np.sin(2 * np.pi * 220.0 * tt)
    track_a[i0:i1] += loud
    track_b[i0:i1] += loud

    out_dir = os.path.dirname(os.path.abspath(__file__))
    write_wav(os.path.join(out_dir, "sample_a.wav"), to_int16(track_a))
    write_wav(os.path.join(out_dir, "sample_b.wav"), to_int16(track_b))
    print("written:", os.path.join(out_dir, "sample_a.wav"))
    print("written:", os.path.join(out_dir, "sample_b.wav"))


if __name__ == "__main__":
    main()
