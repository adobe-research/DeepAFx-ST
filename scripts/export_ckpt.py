import os
import sys
import glob
import torch
import pickle
import pytorch_lightning as pl
import deepafx_st

sys.modules["deepafx_st"] = deepafx_st  # patch for name change

if __name__ == "__main__":

    checkpoint_dir = "checkpoints_fixed"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for experiment in ["probes", "style", "proxies"]:

        for v in [0, 1, 2]:
            ckpt_paths = glob.glob(
                os.path.join(
                    "checkpoints",
                    experiment,
                    "**",
                    "**",
                    "lightning_logs",
                    f"version_{v}",
                    "checkpoints",
                    "*.ckpt",
                )
            )

            for ckpt_path in ckpt_paths:
                print(ckpt_path)

                processor_model_id = ckpt_path.split("/")[-5]
                print(processor_model_id)

                if "m" in processor_model_id:
                    peq_ckpt = "checkpoints/proxies/jamendo/peq/lightning_logs/version_0/checkpoints/epoch=326-step=204374-val-jamendo-peq.ckpt"
                    comp_ckpt = "checkpoints/proxies/jamendo/comp/lightning_logs/version_0/checkpoints/epoch=274-step=171874-val-jamendo-comp.ckpt"
                else:
                    peq_ckpt = "checkpoints/proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt"
                    comp_ckpt = "checkpoints/proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt"

                proxy_ckpts = [peq_ckpt, comp_ckpt]

                if experiment == "style":
                    model = deepafx_st.system.System.load_from_checkpoint(
                        ckpt_path,
                        proxy_ckpts=proxy_ckpts,
                        strict=False,
                    )
                elif experiment == "probes":
                    if "speech" in ckpt_path:
                        deepafx_st_autodiff_ckpt = "checkpoints/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt"
                        deepafx_st_spsa_ckpt = "checkpoints/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt"
                        deepafx_st_proxy0_ckpt = "checkpoints/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt"
                    elif "music" in ckpt_path:
                        deepafx_st_autodiff_ckpt = "checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt"
                        deepafx_st_spsa_ckpt = "checkpoints/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt"
                        deepafx_st_proxy0_ckpt = "checkpoints/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt"

                    model = (
                        deepafx_st.probes.probe_system.ProbeSystem.load_from_checkpoint(
                            ckpt_path,
                            strict=False,
                            deepafx_st_autodiff_ckpt=deepafx_st_autodiff_ckpt,
                            deepafx_st_spsa_ckpt=deepafx_st_spsa_ckpt,
                            deepafx_st_proxy0_ckpt=deepafx_st_proxy0_ckpt,
                        )
                    )
                elif experiment == "proxies":
                    model = deepafx_st.processors.proxy.proxy_system.ProxySystem.load_from_checkpoint(
                        ckpt_path,
                        strict=False,
                    )
                else:
                    raise RuntimeError(f"Invalid experiment: {experiment}")

                ckpt_path_dirname = os.path.dirname(ckpt_path)
                ckpt_path_basename = os.path.basename(ckpt_path)
                ckpt_path_fixed = ckpt_path_dirname.replace(
                    "checkpoints", checkpoint_dir, 1
                )

                if not os.path.isdir(ckpt_path_fixed):
                    os.makedirs(ckpt_path_fixed)

                ckpt_path_fixed = os.path.join(ckpt_path_fixed, ckpt_path_basename)
                print(ckpt_path_fixed)

                trainer = pl.Trainer()
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_fixed)

                del model
                del trainer
