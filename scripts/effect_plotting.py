import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from deepafx_st.processors.dsp.peq import biqaud


def plot_peq_response(
    p_peq_denorm,
    sr,
    ax=None,
    label=None,
    color=None,
    points=False,
    center_line=False,
):

    ls_gain = p_peq_denorm[0]
    ls_freq = p_peq_denorm[1]
    ls_q = p_peq_denorm[2]
    b0, a0 = biqaud(ls_gain, ls_freq, ls_q, sr, filter_type="low_shelf")
    sos0 = np.concatenate((b0, a0))

    f1_gain = p_peq_denorm[3]
    f1_freq = p_peq_denorm[4]
    f1_q = p_peq_denorm[5]
    b1, a1 = biqaud(f1_gain, f1_freq, f1_q, sr, filter_type="peaking")
    sos1 = np.concatenate((b1, a1))

    f2_gain = p_peq_denorm[6]
    f2_freq = p_peq_denorm[7]
    f2_q = p_peq_denorm[8]
    b2, a2 = biqaud(f2_gain, f2_freq, f2_q, sr, filter_type="peaking")
    sos2 = np.concatenate((b2, a2))

    f3_gain = p_peq_denorm[9]
    f3_freq = p_peq_denorm[10]
    f3_q = p_peq_denorm[11]
    b3, a3 = biqaud(f3_gain, f3_freq, f3_q, sr, filter_type="peaking")
    sos3 = np.concatenate((b3, a3))

    f4_gain = p_peq_denorm[12]
    f4_freq = p_peq_denorm[13]
    f4_q = p_peq_denorm[14]
    b4, a4 = biqaud(f4_gain, f4_freq, f4_q, sr, filter_type="peaking")
    sos4 = np.concatenate((b4, a4))

    hs_gain = p_peq_denorm[15]
    hs_freq = p_peq_denorm[16]
    hs_q = p_peq_denorm[17]
    b5, a5 = biqaud(hs_gain, hs_freq, hs_q, sr, filter_type="high_shelf")
    sos5 = np.concatenate((b5, a5))

    sos = [sos0, sos1, sos2, sos3, sos4, sos5]
    sos = np.array(sos)
    # print(sos.shape)
    # print(sos)

    # measure freq response
    w, h = scipy.signal.sosfreqz(sos, fs=22050, worN=2048)

    if ax is None:
        fig, axs = plt.subplots()

    if center_line:
        ax.plot(w, np.zeros(w.shape), color="lightgray")

    ax.plot(w, 20 * np.log10(np.abs(h)), label=label, color=color)
    if points:
        ax.scatter(ls_freq, ls_gain, color=color)
        ax.scatter(f1_freq, f1_gain, color=color)
        ax.scatter(f2_freq, f2_gain, color=color)
        ax.scatter(f3_freq, f3_gain, color=color)
        ax.scatter(f4_freq, f4_gain, color=color)
        ax.scatter(hs_freq, hs_gain, color=color)


def plot_comp_response(
    p_comp_denorm,
    sr,
    ax=None,
    label=None,
    color=None,
    center_line=False,
):

    # get parameters
    threshold = p_comp_denorm[0]
    ratio = p_comp_denorm[1]
    attack_ms = p_comp_denorm[2] * 1000
    release_ms = p_comp_denorm[3] * 1000
    knee_db = p_comp_denorm[4]
    makeup_db = p_comp_denorm[5]

    # print(knee_db)

    x = np.linspace(-80, 0)  # input level
    y = np.zeros(x.shape)  # output level

    idx = np.where((2 * (x - threshold)) < -knee_db)
    y[idx] = x[idx]

    idx = np.where((2 * np.abs(x - threshold)) <= knee_db)
    y[idx] = x[idx] + (
        (1 / ratio - 1) * (((x[idx] - threshold + (knee_db / 2))) ** 2)
    ) / (2 * knee_db)

    idx = np.where((2 * (x - threshold)) > knee_db)
    y[idx] = threshold + ((x[idx] - threshold) / (ratio))

    text_height = threshold + ((0 - threshold) / (ratio))

    # plot the first part of the line
    ax.plot(x, y, label=label, color=color)
    if center_line:
        ax.plot(x, x, color="lightgray", linestyle="--")
    ax.text(0, text_height, f"{threshold:0.1f} dB  {ratio:0.1f}:1")
