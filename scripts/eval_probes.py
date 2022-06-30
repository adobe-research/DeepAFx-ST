import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.metrics import classification_report

from deepafx_st.data.style import StyleDataset
from deepafx_st.probes.probe_system import ProbeSystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        help="Path to top-level directory with probe checkpoints.",
        type=str,
    )
    parser.add_argument(
        "--eval_dataset",
        help="Path to directory containing style dataset for evaluation.",
        type=str,
    )
    parser.add_argument(
        "--audio_type",
        help="Evaluate only models trained on this type of audio 'speech' or 'music'.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Save outputs here.",
        type=str,
    )
    parser.add_argument(
        "--subset",
        help="One of either train, val, or test.",
        type=str,
        default="val",
    )
    parser.add_argument("--gpu", help="Use gpu.", action="store_true")
    args = parser.parse_args()

    # ------------------ load models ------------------
    models = {}  # storage for pretrained models
    model_dirs = glob.glob(os.path.join(args.ckpt_dir, "*"))
    model_dirs = [md for md in model_dirs if os.path.isdir(md)]

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        ckpt_paths = glob.glob(
            os.path.join(
                model_dir,
                "lightning_logs",
                "version_0",
                "checkpoints",
                "*.ckpt",
            )
        )

        if len(ckpt_paths) < 1:
            print(f"WARNING: No checkpoint found for {model_name} model.")
            continue

        ckpt_path = ckpt_paths[0]

        if args.audio_type not in ckpt_path:
            print(f"Skipping {ckpt_path}")
            continue

        print(os.path.basename(ckpt_path))
        if "speech" in ckpt_path:
            deepafx_st_autodiff_ckpt = "checkpoints/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt"
            deepafx_st_spsa_ckpt = "checkpoints/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt"
            deepafx_st_proxy0_ckpt = "checkpoints/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt"
        elif "music" in ckpt_path:
            deepafx_st_autodiff_ckpt = "checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt"
            deepafx_st_spsa_ckpt = "checkpoints/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt"
            deepafx_st_proxy0_ckpt = "checkpoints/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt"

        model = ProbeSystem.load_from_checkpoint(
            ckpt_path,
            strict=False,
            deepafx_st_autodiff_ckpt=deepafx_st_autodiff_ckpt,
            deepafx_st_spsa_ckpt=deepafx_st_spsa_ckpt,
            deepafx_st_proxy0_ckpt=deepafx_st_proxy0_ckpt,
        )
        model.eval()
        if args.gpu:
            model.cuda()
        models[model_name] = model

    # create evaluation dataset
    eval_dataset = StyleDataset(
        args.eval_dataset,
        subset=args.subset,
    )

    # iterate over dataset and make predictions with all models
    preds = {"true": []}
    for bidx, batch in enumerate(tqdm(eval_dataset), 0):
        x, y = batch

        if args.gpu:
            x = x.to("cuda")

        preds["true"].append(y)

        for model_name, model in models.items():
            with torch.no_grad():
                y_hat = model(x.view(1, 1, -1))

                if model_name not in preds:
                    preds[model_name] = []

                preds[model_name].append(y_hat.argmax().cpu())

    # create confusion matracies
    print("-------------------------------------------------------")
    print(f"-------------------{args.audio_type}---------------------")
    print("-------------------------------------------------------")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    for model_name, pred in preds.items():
        y_true = np.array(preds["true"]).reshape(-1)
        y_pred = np.array(pred).reshape(-1)
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=eval_dataset.class_labels,
            cmap="Blues",
        )
        acc = accuracy_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred, average="weighted")
        plt.title(f"{model_name} acc: {acc*100:0.2f}%  f1: {f1score:0.2f}")
        print(f"{model_name} acc: {acc*100:0.2f}%  f1: {f1score:0.2f}")
        print(
            classification_report(
                y_true, y_pred, target_names=eval_dataset.class_labels
            )
        )

        filename = f"{model_name}-{args.subset}"
        filepath = os.path.join(args.output_dir, filename)
        plt.savefig(filepath + ".png", dpi=300)

    print("-------------------------------------------------------")
