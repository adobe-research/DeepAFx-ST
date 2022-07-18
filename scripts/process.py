import os
import glob
import torch
import resampy
import argparse
import torchaudio
import numpy as np

from deepafx_st.utils import DSPMode
from deepafx_st.utils import count_parameters
from deepafx_st.system import System

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to audio file to process.",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reference",
        help="Path to reference audio file.",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        help="Path to pre-trained checkpoint.",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        help="Run inference on GPU. (Otherwise CPU).",
        action="store_true",
    )
    parser.add_argument(
        "--time",
        help="Execute inference 100x in a loop and time the model.",
        action="store_true",
    )
    parser.add_argument(
        "--no_dsp",
        help="Only use neural networks for proxy.",
        action="store_true",
    )

    args = parser.parse_args()

    # load the model
    if "proxy" in args.ckpt:
        logdir = os.path.dirname(os.path.dirname(args.ckpt))
        # Assumes speech proxies in specific location
        pckpts = 'checkpoints'
        if 'proxy0m' in logdir or 'proxy2m' in logdir:
            peq_ckpt = os.path.join(pckpts, "proxies/jamendo/peq/lightning_logs/version_0/checkpoints/epoch=326-step=204374-val-jamendo-peq.ckpt" )
            comp_ckpt = os.path.join(pckpts, "proxies/jamendo/comp/lightning_logs/version_0/checkpoints/epoch=274-step=171874-val-jamendo-comp.ckpt" )
        else:
            peq_ckpt = os.path.join(pckpts, "proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt" )
            comp_ckpt = os.path.join(pckpts, "proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt" )

        proxy_ckpts = [peq_ckpt, comp_ckpt]
        print(f"Found {len(proxy_ckpts)}: {proxy_ckpts}")
        dsp_mode = DSPMode.INFER
        if args.no_dsp:
            dsp_mode = DSPMode.NONE

        system = System.load_from_checkpoint(
            args.ckpt, dsp_mode=dsp_mode, proxy_ckpts=proxy_ckpts
        ).eval()
    else:
        use_dsp = False
        system = System.load_from_checkpoint(
            args.ckpt, dsp_mode=DSPMode.NONE, batch_size=1
        ).eval()

    if args.gpu:
        system.to("cuda")

    # load audio data
    x, x_sr = torchaudio.load(args.input)
    r, r_sr = torchaudio.load(args.reference)

    # resample if needed
    if x_sr != 24000:
        print("Resampling to 24000 Hz...")
        x_24000 = torch.tensor(resampy.resample(x.view(-1).numpy(), x_sr, 24000))
        x_24000 = x_24000.view(1, -1)
    else:
        x_24000 = x

    if r_sr != 24000:
        print("Resampling to 24000 Hz...")
        r_24000 = torch.tensor(resampy.resample(r.view(-1).numpy(), r_sr, 24000))
        r_24000 = r_24000.view(1, -1)
    else:
        r_24000 = r

    # peak normalize to -12 dBFS
    x_24000 = x_24000[0:1, : 24000 * 5]
    x_24000 /= x_24000.abs().max()
    x_24000 *= 10 ** (-12 / 20.0)
    x_24000 = x_24000.view(1, 1, -1)

    # peak normalize to -12 dBFS
    r_24000 = r_24000[0:1, : 24000 * 5]
    r_24000 /= r_24000.abs().max()
    r_24000 *= 10 ** (-12 / 20.0)
    r_24000 = r_24000.view(1, 1, -1)

    if args.gpu:
        x_24000 = x_24000.to("cuda")
        r_24000 = r_24000.to("cuda")

    if args.time:
        torch.set_num_threads(1)
        # Warm up
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)

        # pass audio through model
        times = []
        num_times = 13
        warm_up = 3
        with torch.no_grad():
            for i in range(num_times):
                print("iteration", i)
                y_hat, p, e, encoder_time_sec, dsp_time_sec = system(
                    x_24000, dsp_mode=DSPMode.INFER, time_it=args.time
                )
                if i >= warm_up:
                    times.append((encoder_time_sec, dsp_time_sec))

        audio_len_sec = x_24000.shape[-1] / 24000.0
        ave_times = np.mean(np.array(times), axis=0)

        print("**********Config**********")
        print("gpu", args.gpu)
        print("dsp", system.hparams.processor_model)
        print("ave_times", ave_times, "length", audio_len_sec)
        print(
            "rtf (encoder, dsp)",
            ave_times[0] / audio_len_sec,
            ave_times[1] / audio_len_sec,
        )
        print("#parameters", count_parameters(system.processor, trainable_only=False))

    else:
        # pass audio through model
        with torch.no_grad():
            y_hat, p, e = system(x_24000, r_24000)

    y_hat = y_hat.view(1, -1)
    y_hat /= y_hat.abs().max()
    x_24000 /= x_24000.abs().max()

    # save to disk
    dirname = os.path.dirname(args.input)
    filename = os.path.basename(args.input).replace(".wav", "")
    reference = os.path.basename(args.reference).replace(".wav", "")
    out_filepath = os.path.join(dirname, f"{filename}_out_ref={reference}.wav")
    in_filepath = os.path.join(dirname, f"{filename}_in.wav")
    print(f"Saved output to {out_filepath}")
    torchaudio.save(out_filepath, y_hat.cpu().view(1, -1), 24000)
    torchaudio.save(in_filepath, x_24000.cpu().view(1, -1), 24000)

    system.shutdown()

