## LibriTTS @ 24kHz (test)

CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs/style/libritts/ \
--root_dir /import/c4dm-datasets/deepafx_st/ \
--gpu \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset test \
--output /import/c4dm-datasets/deepafx_st/
--save \

1000

 Corrupt                PESQ: 3.765  MRSTFT: 1.187  MSD: 2.180  SCE: 687.534  CFE: 6.261  RMS: 6.983  LUFS: 2.426  
 TCN1 (libritts)        PESQ: 4.258  MRSTFT: 0.405  MSD: 0.887  SCE: 128.408  CFE: 2.582  RMS: 2.237  LUFS: 1.066  
 TCN2 (libritts)        PESQ: 4.281  MRSTFT: 0.372  MSD: 0.833  SCE: 117.496  CFE: 2.460  RMS: 1.927  LUFS: 0.925  
 SPSA (libritts)        PESQ: 4.180  MRSTFT: 0.635  MSD: 1.406  SCE: 219.409  CFE: 5.734  RMS: 3.263  LUFS: 1.600  
 proxy0 (libritts)      PESQ: 3.643  MRSTFT: 0.676  MSD: 1.405  SCE: 264.970  CFE: 4.291  RMS: 2.812  LUFS: 1.340  
 Proxy1 (libritts)      PESQ: 3.999  MRSTFT: 1.038  MSD: 2.179  SCE: 440.159  CFE: 5.283  RMS: 5.472  LUFS: 2.679  
 Proxy2 (libritts)      PESQ: 3.945  MRSTFT: 1.058  MSD: 2.088  SCE: 404.867  CFE: 5.328  RMS: 6.820  LUFS: 3.197  
 Autodiff (libritts)    PESQ: 4.310  MRSTFT: 0.388  MSD: 0.882  SCE: 111.549  CFE: 4.079  RMS: 1.828  LUFS: 0.823  
 Baseline (libritts)    PESQ: 3.856  MRSTFT: 0.943  MSD: 1.955  SCE: 410.330  CFE: 4.013  RMS: 4.204  LUFS: 1.674  
--------------------------------

 Corrupt      3.765  & 1.187  & 2.180  & 687.534    & 6.983  & 2.426 
 Baseline     3.856  & 0.943  & 1.955  & 410.330   & 4.204  & 1.674  
 TCN1      4.258  & 0.405  & 0.887  & 128.408    & 2.237  & 1.066  
 TCN2      4.281  & 0.372  & 0.833  & 117.496    & 1.927  & 0.925 
 SPSA     4.180  & 0.635  & 1.406  & 219.409   & 3.263  & 1.600  
 proxy0      3.643  & 0.676  & 1.405  & 264.970    & 2.812  & 1.340  
 Proxy1      3.999  & 1.038  & 2.179  & 440.159   & 5.472  & 2.679  
 Proxy2      3.945  & 1.058  & 2.088  & 404.867   & 6.820  & 3.197  
 Autodiff      4.310  & 0.388  & 0.882  & 111.549    & 1.828  & 0.823 
--------------------------------

1000 NEW MSD Scores

 Corrupt                PESQ: 3.765  MRSTFT: 1.187  MSD: 5.311  SCE: 687.534  CFE: 6.261  RMS: 6.983  LUFS: 2.426  
 TCN1 (libritts)        PESQ: 4.258  MRSTFT: 0.405  MSD: 1.647  SCE: 128.400  CFE: 2.582  RMS: 2.237  LUFS: 1.066  
 TCN2 (libritts)        PESQ: 4.281  MRSTFT: 0.372  MSD: 1.529  SCE: 117.493  CFE: 2.460  RMS: 1.927  LUFS: 0.925  
 SPSA (libritts)        PESQ: 4.180  MRSTFT: 0.635  MSD: 2.894  SCE: 219.409  CFE: 5.734  RMS: 3.263  LUFS: 1.600  
 proxy0 (libritts)      PESQ: 3.643  MRSTFT: 0.676  MSD: 2.483  SCE: 264.947  CFE: 4.291  RMS: 2.811  LUFS: 1.340  
 Proxy1 (libritts)      PESQ: 3.999  MRSTFT: 1.038  MSD: 4.766  SCE: 440.159  CFE: 5.283  RMS: 5.472  LUFS: 2.679  
 Proxy2 (libritts)      PESQ: 3.945  MRSTFT: 1.058  MSD: 4.858  SCE: 404.866  CFE: 5.328  RMS: 6.820  LUFS: 3.197  
 Autodiff (libritts)    PESQ: 4.310  MRSTFT: 0.388  MSD: 1.692  SCE: 111.507  CFE: 4.079  RMS: 1.828  LUFS: 0.823  
 Baseline (libritts)    PESQ: 3.856  MRSTFT: 0.943  MSD: 4.002  SCE: 410.330  CFE: 4.013  RMS: 4.204  LUFS: 1.674  
--------------------------------
Evaluation complete.

## DAPS @ 24kHz (train) (we don't train with train set)

CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs/style/libritts/ \
--root_dir /import/c4dm-datasets/deepafx_st/ \
--gpu \
--dataset daps \
--dataset_dir daps_24000/cleanraw \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset train \
--output /import/c4dm-datasets/deepafx_st/
#--save \

1000

 Corrupt                PESQ: 3.684  MRSTFT: 1.179  MSD: 2.151  SCE: 641.683  CFE: 6.133  RMS: 6.900  LUFS: 2.314
 TCN1 (daps)            PESQ: 4.185  MRSTFT: 0.419  MSD: 0.884  SCE: 124.609  CFE: 2.473  RMS: 2.098  LUFS: 1.006
 TCN2 (daps)            PESQ: 4.224  MRSTFT: 0.391  MSD: 0.841  SCE: 113.863  CFE: 2.352  RMS: 1.886  LUFS: 0.913
 SPSA (daps)            PESQ: 4.099  MRSTFT: 0.645  MSD: 1.379  SCE: 213.596  CFE: 5.166  RMS: 2.989  LUFS: 1.511
 proxy0 (daps)          PESQ: 3.605  MRSTFT: 0.685  MSD: 1.362  SCE: 249.159  CFE: 4.222  RMS: 2.732  LUFS: 1.350
 Proxy1 (daps)          PESQ: 3.903  MRSTFT: 1.022  MSD: 2.113  SCE: 451.879  CFE: 4.927  RMS: 5.104  LUFS: 2.535
 Proxy2 (daps)          PESQ: 3.891  MRSTFT: 1.037  MSD: 2.045  SCE: 395.421  CFE: 5.112  RMS: 6.754  LUFS: 3.117
 Autodiff (daps)        PESQ: 4.222  MRSTFT: 0.416  MSD: 0.895  SCE: 109.004  CFE: 4.290  RMS: 1.758  LUFS: 0.799
 Baseline (daps)        PESQ: 3.787  MRSTFT: 0.917  MSD: 1.882  SCE: 399.714  CFE: 3.742  RMS: 3.705  LUFS: 1.481
--------------------------------
Evaluation complete.


 Corrupt                & 3.684  & 1.179  & 2.151  & 641.683   & 6.900  & 2.314
 TCN1 (daps)            & 4.185  & 0.419  & 0.884  & 124.609   & 2.098  & 1.006
 TCN2 (daps)            & 4.224  & 0.391  & 0.841  & 113.863   & 1.886  & 0.913
 SPSA (daps)            & 4.099  & 0.645  & 1.379  & 213.596   & 2.989  & 1.511
 proxy0 (daps)          & 3.605  & 0.685  & 1.362  & 249.159   & 2.732  & 1.350
 Proxy1 (daps)          & 3.903  & 1.022  & 2.113  & 451.879   & 5.104  & 2.535
 Proxy2 (daps)          & 3.891  & 1.037  & 2.045  & 395.421   & 6.754  & 3.117
 Autodiff (daps)        & 4.222  & 0.416  & 0.895  & 109.004   & 1.758  & 0.799
 Baseline (daps)        & 3.787  & 0.917  & 1.882  & 399.714   & 3.705  & 1.481

1000 New MSD

 Corrupt                PESQ: 3.684  MRSTFT: 1.179  MSD: 5.318  SCE: 641.683  CFE: 6.133  RMS: 6.900  LUFS: 2.314  
 TCN1 (daps)            PESQ: 4.185  MRSTFT: 0.419  MSD: 1.714  SCE: 124.609  CFE: 2.473  RMS: 2.098  LUFS: 1.006  
 TCN2 (daps)            PESQ: 4.224  MRSTFT: 0.391  MSD: 1.621  SCE: 113.863  CFE: 2.352  RMS: 1.886  LUFS: 0.913  
 SPSA (daps)            PESQ: 4.099  MRSTFT: 0.645  MSD: 3.007  SCE: 213.596  CFE: 5.166  RMS: 2.989  LUFS: 1.511  
 proxy0 (daps)          PESQ: 3.605  MRSTFT: 0.685  MSD: 2.520  SCE: 249.159  CFE: 4.222  RMS: 2.732  LUFS: 1.350  
 Proxy1 (daps)          PESQ: 3.903  MRSTFT: 1.022  MSD: 4.749  SCE: 451.879  CFE: 4.927  RMS: 5.104  LUFS: 2.535  
 Proxy2 (daps)          PESQ: 3.891  MRSTFT: 1.037  MSD: 4.999  SCE: 395.421  CFE: 5.112  RMS: 6.754  LUFS: 3.117  
 Autodiff (daps)        PESQ: 4.222  MRSTFT: 0.416  MSD: 1.802  SCE: 109.004  CFE: 4.290  RMS: 1.758  LUFS: 0.799  
 Baseline (daps)        PESQ: 3.787  MRSTFT: 0.917  MSD: 4.065  SCE: 399.714  CFE: 3.742  RMS: 3.705  LUFS: 1.481  
--------------------------------
Evaluation complete.


## VCTK @ 24kHz (train) (we don't train with train set)

CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs/style/libritts/ \
--root_dir /import/c4dm-datasets/deepafx_st/ \
--gpu \
--dataset vctk \
--dataset_dir vctk_24000 \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset train \
--output /import/c4dm-datasets/deepafx_st/

1000

 Corrupt                PESQ: 3.672  MRSTFT: 1.254  MSD: 2.008  SCE: 815.422  CFE: 6.686  RMS: 7.783  LUFS: 2.532  
 TCN1 (vctk)            PESQ: 4.181  MRSTFT: 0.467  MSD: 0.891  SCE: 173.751  CFE: 2.712  RMS: 2.651  LUFS: 1.165  
 TCN2 (vctk)            PESQ: 4.201  MRSTFT: 0.441  MSD: 0.856  SCE: 163.839  CFE: 2.583  RMS: 2.431  LUFS: 1.086  
 SPSA (vctk)            PESQ: 4.023  MRSTFT: 0.730  MSD: 1.359  SCE: 301.608  CFE: 5.477  RMS: 3.535  LUFS: 1.737  
 proxy0 (vctk)          PESQ: 3.651  MRSTFT: 0.737  MSD: 1.300  SCE: 321.701  CFE: 4.591  RMS: 3.166  LUFS: 1.453  
 Proxy1 (vctk)          PESQ: 3.951  MRSTFT: 1.044  MSD: 1.930  SCE: 591.476  CFE: 5.293  RMS: 5.194  LUFS: 2.651  
 Proxy2 (vctk)          PESQ: 3.894  MRSTFT: 1.087  MSD: 1.934  SCE: 514.048  CFE: 5.544  RMS: 7.065  LUFS: 3.363  
 Autodiff (vctk)        PESQ: 4.218  MRSTFT: 0.481  MSD: 0.924  SCE: 152.748  CFE: 5.169  RMS: 2.317  LUFS: 1.006  
 Baseline (vctk)        PESQ: 3.709  MRSTFT: 1.101  MSD: 1.911  SCE: 657.608  CFE: 4.647  RMS: 5.039  LUFS: 2.018  
--------------------------------
Evaluation complete.

 Corrupt                & 3.672  & 1.254  & 2.008  & 815.422  & 7.783  & 2.532  
 TCN1 (vctk)            & 4.181  & 0.467  & 0.891  & 173.751  & 2.651  & 1.165  
 TCN2 (vctk)            & 4.201  & 0.441  & 0.856  & 163.839  & 2.431  & 1.086  
 SPSA (vctk)            & 4.023  & 0.730  & 1.359  & 301.608  & 3.535  & 1.737  
 proxy0 (vctk)          & 3.651  & 0.737  & 1.300  & 321.701  & 3.166  & 1.453  
 Proxy1 (vctk)          & 3.951  & 1.044  & 1.930  & 591.476  & 5.194  & 2.651  
 Proxy2 (vctk)          & 3.894  & 1.087  & 1.934  & 514.048  & 7.065  & 3.363  
 Autodiff (vctk)        & 4.218  & 0.481  & 0.924  & 152.748  & 2.317  & 1.006  
 Baseline (vctk)        & 3.709  & 1.101  & 1.911  & 657.608  & 5.039  & 2.018  

1000

New MSD scores
 Corrupt                PESQ: 3.672  MRSTFT: 1.254  MSD: 4.373  SCE: 815.422  CFE: 6.686  RMS: 7.783  LUFS: 2.532
 TCN1 (vctk)            PESQ: 4.181  MRSTFT: 0.467  MSD: 1.620  SCE: 173.751  CFE: 2.712  RMS: 2.651  LUFS: 1.165
 TCN2 (vctk)            PESQ: 4.201  MRSTFT: 0.441  MSD: 1.569  SCE: 163.839  CFE: 2.583  RMS: 2.431  LUFS: 1.086
 SPSA (vctk)            PESQ: 4.023  MRSTFT: 0.730  MSD: 2.759  SCE: 301.608  CFE: 5.477  RMS: 3.535  LUFS: 1.737
 proxy0 (vctk)          PESQ: 3.651  MRSTFT: 0.737  MSD: 2.254  SCE: 321.701  CFE: 4.591  RMS: 3.166  LUFS: 1.453
 Proxy1 (vctk)          PESQ: 3.951  MRSTFT: 1.044  MSD: 3.948  SCE: 591.476  CFE: 5.293  RMS: 5.194  LUFS: 2.651
 Proxy2 (vctk)          PESQ: 3.894  MRSTFT: 1.087  MSD: 4.248  SCE: 514.048  CFE: 5.544  RMS: 7.065  LUFS: 3.363
 Autodiff (vctk)        PESQ: 4.218  MRSTFT: 0.481  MSD: 1.757  SCE: 152.748  CFE: 5.169  RMS: 2.317  LUFS: 1.006
 Baseline (vctk)        PESQ: 3.709  MRSTFT: 1.101  MSD: 4.005  SCE: 657.608  CFE: 4.647  RMS: 5.039  LUFS: 2.018
--------------------------------
Evaluation complete.


## Style case study (SPSA)
CUDA_VISIBLE_DEVICES=7 python scripts/style_case_study.py \
--ckpt_path "/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
--input_audio "/import/c4dm-datasets/deepafx_st/vctk_24000" \
--style_audio "/import/c4dm-datasets/deepafx_st/daps_24000_styles/val" \
--gpu \

10 inputs x 10 examples per style

telephone 9
System:   & 1.178  MRSTFT 1.814  MSD 2.434  SCE 193.695  CFE 5.946  RMS 15.071  LUFS 6.835  
Baseline: PESQ 1.235  MRSTFT 1.890  MSD 2.532  SCE 212.969  CFE 2.732  RMS 2.732  LUFS 0.989  
Corrupt: PESQ 1.153  MRSTFT 2.757  MSD 3.827  SCE 384.676  CFE 9.193  RMS 9.193  LUFS 3.187  

bright 9
System:   PESQ 1.150  MRSTFT 2.298  MSD 3.011  SCE 524.796  CFE 4.008  RMS 7.016  LUFS 3.618  
Baseline: PESQ 1.132  MRSTFT 2.364  MSD 3.047  SCE 797.717  CFE 4.258  RMS 4.258  LUFS 1.825  
Corrupt: PESQ 1.143  MRSTFT 2.626  MSD 3.755  SCE 2296.843  CFE 9.971  RMS 9.971  LUFS 2.716  

radio 9
System:   PESQ 1.103  MRSTFT 2.064  MSD 2.793  SCE 239.602  CFE 15.518  RMS 2.521  LUFS 1.181  
Baseline: PESQ 1.139  MRSTFT 2.350  MSD 3.292  SCE 772.898  CFE 11.836  RMS 11.836  LUFS 5.307  
Corrupt: PESQ 1.175  MRSTFT 2.391  MSD 3.296  SCE 451.134  CFE 11.745  RMS 11.745  LUFS 5.414  

podcast 9
System:   PESQ 1.145  MRSTFT 2.330  MSD 3.311  SCE 247.877  CFE 4.556  RMS 2.490  LUFS 1.003  
Baseline: PESQ 1.165  MRSTFT 2.417  MSD 3.415  SCE 597.773  CFE 4.114  RMS 4.114  LUFS 1.465  
Corrupt: PESQ 1.149  MRSTFT 2.445  MSD 3.484  SCE 335.067  CFE 5.127  RMS 5.127  LUFS 2.079  

warm 9
System:   PESQ 1.124  MRSTFT 2.348  MSD 3.492  SCE 282.160  CFE 7.555  RMS 5.295  LUFS 3.060  
Baseline: PESQ 1.110  MRSTFT 2.528  MSD 3.804  SCE 790.690  CFE 8.481  RMS 8.481  LUFS 3.364  
Corrupt: PESQ 1.138  MRSTFT 2.530  MSD 3.703  SCE 565.930  CFE 14.402  RMS 14.402  LUFS 5.193  

## Style case study
CUDA_VISIBLE_DEVICES=4 python scripts/style_case_study.py \
--ckpt_paths \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt" \
--style_audio "/import/c4dm-datasets/deepafx_st/daps_24000_styles_1000_diverse/train" \
--output_dir "/import/c4dm-datasets/deepafx_st/style_case_study" \
--gpu \
#--save \
#--plot \

broadcast-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.72s/it]
spsa       MSD: 4.522  SCE: 182.120  RMS: 3.554  LUFS: 1.509  
autodiff   MSD: 4.031  SCE: 352.741  RMS: 2.788  LUFS: 1.148  
proxy0     MSD: 4.615  SCE: 269.970  RMS: 3.619  LUFS: 1.784  
Baseline   MSD: 6.575  SCE: 396.523  RMS: 11.710  LUFS: 5.149  
Corrupt    MSD: 6.950  SCE: 411.573  RMS: 12.435  LUFS: 5.258  

broadcast-->telephone
100%|███████████████████████████████████████████| 10/10 [00:46<00:00,  4.61s/it]
spsa       MSD: 6.581  SCE: 130.320  RMS: 14.865  LUFS: 6.979  
autodiff   MSD: 5.966  SCE: 87.473  RMS: 6.932  LUFS: 3.022  
proxy0     MSD: 8.802  SCE: 223.404  RMS: 11.616  LUFS: 5.182  
Baseline   MSD: 6.782  SCE: 283.199  RMS: 5.826  LUFS: 3.391  
Corrupt    MSD: 11.492  SCE: 461.633  RMS: 5.259  LUFS: 2.276  

broadcast-->neutral
100%|███████████████████████████████████████████| 10/10 [00:41<00:00,  4.12s/it]
spsa       MSD: 8.776  SCE: 284.227  RMS: 3.605  LUFS: 1.535  
autodiff   MSD: 8.765  SCE: 375.015  RMS: 8.036  LUFS: 3.435  
proxy0     MSD: 8.891  SCE: 299.929  RMS: 6.967  LUFS: 2.908  
Baseline   MSD: 8.653  SCE: 294.783  RMS: 8.922  LUFS: 4.117  
Corrupt    MSD: 9.496  SCE: 458.657  RMS: 9.055  LUFS: 4.152  

broadcast-->bright
100%|███████████████████████████████████████████| 10/10 [00:57<00:00,  5.80s/it]
spsa       MSD: 5.041  SCE: 632.066  RMS: 12.098  LUFS: 6.022  
autodiff   MSD: 5.274  SCE: 518.414  RMS: 13.832  LUFS: 6.562  
proxy0     MSD: 6.408  SCE: 585.818  RMS: 7.310  LUFS: 4.727  
Baseline   MSD: 5.414  SCE: 782.185  RMS: 20.304  LUFS: 10.463  
Corrupt    MSD: 6.707  SCE: 2252.014  RMS: 11.961  LUFS: 7.429  

broadcast-->warm
100%|███████████████████████████████████████████| 10/10 [00:56<00:00,  5.68s/it]
spsa       MSD: 11.578  SCE: 167.850  RMS: 5.142  LUFS: 3.489  
autodiff   MSD: 10.112  SCE: 247.329  RMS: 12.229  LUFS: 6.786  
proxy0     MSD: 11.477  SCE: 295.929  RMS: 22.112  LUFS: 11.750  
Baseline   MSD: 10.507  SCE: 408.553  RMS: 26.613  LUFS: 12.465  
Corrupt    MSD: 11.337  SCE: 789.713  RMS: 30.505  LUFS: 12.952  

telephone-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:10<00:00,  1.03s/it]
spsa       MSD: 8.697  SCE: 256.463  RMS: 2.159  LUFS: 1.360  
autodiff   MSD: 7.018  SCE: 268.331  RMS: 1.101  LUFS: 0.977  
proxy0     MSD: 8.816  SCE: 1561.290  RMS: 6.828  LUFS: 1.942  
Baseline   MSD: 9.524  SCE: 582.040  RMS: 8.052  LUFS: 4.040  
Corrupt    MSD: 11.522  SCE: 357.191  RMS: 8.650  LUFS: 3.928  

telephone-->telephone
100%|███████████████████████████████████████████| 10/10 [00:11<00:00,  1.15s/it]
spsa       MSD: 5.716  SCE: 99.469  RMS: 6.316  LUFS: 2.624  
autodiff   MSD: 6.058  SCE: 57.992  RMS: 3.994  LUFS: 1.865  
proxy0     MSD: 6.660  SCE: 69.429  RMS: 3.669  LUFS: 1.641  
Baseline   MSD: 6.246  SCE: 134.453  RMS: 2.541  LUFS: 0.724  
Corrupt    MSD: 6.477  SCE: 145.124  RMS: 3.164  LUFS: 1.102  

telephone-->neutral
100%|███████████████████████████████████████████| 10/10 [00:22<00:00,  2.22s/it]
spsa       MSD: 10.208  SCE: 260.075  RMS: 4.410  LUFS: 1.704  
autodiff   MSD: 10.647  SCE: 267.470  RMS: 2.601  LUFS: 1.424  
proxy0     MSD: 11.776  SCE: 782.558  RMS: 14.902  LUFS: 5.210  
Baseline   MSD: 10.858  SCE: 462.504  RMS: 4.944  LUFS: 1.820  
Corrupt    MSD: 11.803  SCE: 359.048  RMS: 11.537  LUFS: 3.876  

telephone-->bright
100%|███████████████████████████████████████████| 10/10 [00:09<00:00,  1.04it/s]
spsa       MSD: 6.466  SCE: 1290.766  RMS: 2.904  LUFS: 1.350  
autodiff   MSD: 6.892  SCE: 359.603  RMS: 4.223  LUFS: 2.198  
proxy0     MSD: 7.223  SCE: 1230.582  RMS: 4.967  LUFS: 2.182  
Baseline   MSD: 7.135  SCE: 601.520  RMS: 3.659  LUFS: 1.522  
Corrupt    MSD: 8.123  SCE: 2198.995  RMS: 4.774  LUFS: 1.853  

telephone-->warm
100%|███████████████████████████████████████████| 10/10 [00:36<00:00,  3.60s/it]
spsa       MSD: 14.016  SCE: 187.455  RMS: 11.562  LUFS: 4.883  
autodiff   MSD: 11.033  SCE: 368.469  RMS: 8.113  LUFS: 4.508  
proxy0     MSD: 15.291  SCE: 1533.699  RMS: 44.513  LUFS: 18.723  
Baseline   MSD: 15.129  SCE: 286.571  RMS: 5.691  LUFS: 1.939  
Corrupt    MSD: 17.063  SCE: 401.732  RMS: 22.030  LUFS: 7.518  

neutral-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:07<00:00,  1.29it/s]
spsa       MSD: 5.379  SCE: 186.673  RMS: 1.992  LUFS: 0.990  
autodiff   MSD: 5.450  SCE: 184.832  RMS: 0.910  LUFS: 0.353  
proxy0     MSD: 5.696  SCE: 426.500  RMS: 2.029  LUFS: 1.023  
Baseline   MSD: 8.856  SCE: 707.260  RMS: 13.779  LUFS: 6.409  
Corrupt    MSD: 9.007  SCE: 253.347  RMS: 12.566  LUFS: 5.949  

neutral-->telephone
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.11s/it]
spsa       MSD: 6.586  SCE: 137.444  RMS: 11.907  LUFS: 5.679  
autodiff   MSD: 6.775  SCE: 79.489  RMS: 7.413  LUFS: 2.940  
proxy0     MSD: 8.618  SCE: 273.167  RMS: 8.310  LUFS: 3.709  
Baseline   MSD: 7.167  SCE: 163.260  RMS: 3.488  LUFS: 1.508  
Corrupt    MSD: 12.148  SCE: 266.549  RMS: 9.099  LUFS: 2.703  

neutral-->neutral
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.79s/it]
spsa       MSD: 10.705  SCE: 227.437  RMS: 3.666  LUFS: 1.459  
autodiff   MSD: 10.884  SCE: 227.260  RMS: 6.592  LUFS: 2.734  
proxy0     MSD: 10.923  SCE: 233.776  RMS: 5.590  LUFS: 2.262  
Baseline   MSD: 10.688  SCE: 445.606  RMS: 3.532  LUFS: 1.476  
Corrupt    MSD: 11.592  SCE: 264.716  RMS: 5.298  LUFS: 2.222  

neutral-->bright
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.79s/it]
spsa       MSD: 6.496  SCE: 602.073  RMS: 3.655  LUFS: 1.890  
autodiff   MSD: 7.098  SCE: 290.838  RMS: 5.375  LUFS: 2.539  
proxy0     MSD: 8.008  SCE: 830.619  RMS: 10.439  LUFS: 6.364  
Baseline   MSD: 8.187  SCE: 940.809  RMS: 6.405  LUFS: 2.563  
Corrupt    MSD: 9.876  SCE: 2139.984  RMS: 9.523  LUFS: 2.784  

neutral-->warm
100%|███████████████████████████████████████████| 10/10 [00:54<00:00,  5.50s/it]
spsa       MSD: 14.845  SCE: 279.135  RMS: 3.300  LUFS: 2.232  
autodiff   MSD: 14.592  SCE: 423.990  RMS: 3.939  LUFS: 1.885  
proxy0     MSD: 13.307  SCE: 515.236  RMS: 16.671  LUFS: 8.101  
Baseline   MSD: 14.943  SCE: 548.626  RMS: 11.837  LUFS: 4.657  
Corrupt    MSD: 17.023  SCE: 861.202  RMS: 14.336  LUFS: 5.142  

bright-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:06<00:00,  1.62it/s]
spsa       MSD: 6.772  SCE: 532.672  RMS: 4.613  LUFS: 2.300  
autodiff   MSD: 5.774  SCE: 341.599  RMS: 2.149  LUFS: 1.043  
proxy0     MSD: 6.704  SCE: 522.506  RMS: 1.888  LUFS: 1.093  
Baseline   MSD: 8.718  SCE: 983.950  RMS: 10.220  LUFS: 4.666  
Corrupt    MSD: 8.120  SCE: 2187.024  RMS: 6.306  LUFS: 4.037  

bright-->telephone
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.75s/it]
spsa       MSD: 5.855  SCE: 270.561  RMS: 8.932  LUFS: 4.175  
autodiff   MSD: 6.628  SCE: 94.206  RMS: 4.192  LUFS: 1.253  
proxy0     MSD: 9.108  SCE: 71.258  RMS: 8.973  LUFS: 4.219  
Baseline   MSD: 6.980  SCE: 230.584  RMS: 4.646  LUFS: 1.684  
Corrupt    MSD: 8.623  SCE: 2354.555  RMS: 4.436  LUFS: 1.660  

bright-->neutral
100%|███████████████████████████████████████████| 10/10 [00:24<00:00,  2.49s/it]
spsa       MSD: 9.165  SCE: 398.710  RMS: 2.955  LUFS: 1.443  
autodiff   MSD: 9.007  SCE: 289.767  RMS: 4.499  LUFS: 2.343  
proxy0     MSD: 9.422  SCE: 306.264  RMS: 10.855  LUFS: 4.741  
Baseline   MSD: 9.376  SCE: 393.232  RMS: 2.709  LUFS: 1.195  
Corrupt    MSD: 10.215  SCE: 2303.824  RMS: 10.234  LUFS: 2.903  

bright-->bright
100%|███████████████████████████████████████████| 10/10 [00:19<00:00,  1.98s/it]
spsa       MSD: 6.685  SCE: 411.594  RMS: 5.279  LUFS: 1.701  
autodiff   MSD: 6.916  SCE: 375.956  RMS: 6.980  LUFS: 2.641  
proxy0     MSD: 7.404  SCE: 978.521  RMS: 4.706  LUFS: 2.541  
Baseline   MSD: 6.885  SCE: 488.962  RMS: 3.045  LUFS: 1.165  
Corrupt    MSD: 7.663  SCE: 433.188  RMS: 5.412  LUFS: 2.207  

bright-->warm
100%|███████████████████████████████████████████| 10/10 [00:41<00:00,  4.16s/it]
spsa       MSD: 14.430  SCE: 702.406  RMS: 7.840  LUFS: 3.216  
autodiff   MSD: 13.001  SCE: 304.595  RMS: 6.781  LUFS: 3.275  
proxy0     MSD: 14.754  SCE: 1581.939  RMS: 43.075  LUFS: 17.552  
Baseline   MSD: 15.660  SCE: 1029.709  RMS: 8.135  LUFS: 2.869  
Corrupt    MSD: 15.150  SCE: 2758.329  RMS: 24.744  LUFS: 8.197  

warm-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:07<00:00,  1.32it/s]
spsa       MSD: 5.225  SCE: 183.420  RMS: 6.149  LUFS: 2.679  
autodiff   MSD: 4.704  SCE: 245.411  RMS: 1.272  LUFS: 0.566  
proxy0     MSD: 5.890  SCE: 717.873  RMS: 2.599  LUFS: 1.820  
Baseline   MSD: 8.872  SCE: 839.614  RMS: 15.062  LUFS: 7.262  
Corrupt    MSD: 12.639  SCE: 694.704  RMS: 29.492  LUFS: 12.191  

warm-->telephone
100%|███████████████████████████████████████████| 10/10 [00:11<00:00,  1.18s/it]
spsa       MSD: 7.045  SCE: 102.885  RMS: 13.941  LUFS: 6.782  
autodiff   MSD: 7.008  SCE: 95.444  RMS: 4.295  LUFS: 2.152  
proxy0     MSD: 8.251  SCE: 163.098  RMS: 12.524  LUFS: 5.308  
Baseline   MSD: 8.355  SCE: 363.053  RMS: 3.255  LUFS: 1.194  
Corrupt    MSD: 15.149  SCE: 508.455  RMS: 20.462  LUFS: 7.136  

warm-->neutral
100%|███████████████████████████████████████████| 10/10 [00:13<00:00,  1.39s/it]
spsa       MSD: 10.097  SCE: 240.049  RMS: 4.867  LUFS: 1.852  
autodiff   MSD: 9.771  SCE: 365.662  RMS: 6.075  LUFS: 2.589  
proxy0     MSD: 22.146  SCE: 667.698  RMS: 12.068  LUFS: 5.956  
Baseline   MSD: 10.539  SCE: 671.609  RMS: 8.868  LUFS: 3.426  
Corrupt    MSD: 13.590  SCE: 711.717  RMS: 14.475  LUFS: 5.441  

warm-->bright
100%|███████████████████████████████████████████| 10/10 [00:09<00:00,  1.01it/s]
spsa       MSD: 6.968  SCE: 621.412  RMS: 3.168  LUFS: 2.040  
autodiff   MSD: 6.544  SCE: 403.204  RMS: 4.901  LUFS: 2.014  
proxy0     MSD: 12.103  SCE: 746.076  RMS: 16.569  LUFS: 9.295  
Baseline   MSD: 8.659  SCE: 1147.820  RMS: 7.271  LUFS: 1.948  
Corrupt    MSD: 15.391  SCE: 2795.258  RMS: 24.478  LUFS: 7.965  

warm-->warm
100%|███████████████████████████████████████████| 10/10 [00:23<00:00,  2.30s/it]
spsa       MSD: 12.112  SCE: 202.358  RMS: 3.484  LUFS: 1.735  
autodiff   MSD: 11.914  SCE: 270.304  RMS: 3.797  LUFS: 1.755  
proxy0     MSD: 13.057  SCE: 790.326  RMS: 10.564  LUFS: 4.503  
Baseline   MSD: 12.075  SCE: 356.166  RMS: 3.608  LUFS: 1.892  
Corrupt    MSD: 12.278  SCE: 366.978  RMS: 2.604  LUFS: 1.345  



## Jamendo @ 24kHz (test)

CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/ \
--root_dir /import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/ \
--gpu \
--dataset jamendo \
--dataset_dir mtg-jamendo_24000/ \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset test \
--save \
--ext flac \
--output /import/c4dm-datasets/deepafx_st/

1000

 Corrupt                PESQ: 2.849  MRSTFT: 1.175  MSD: 2.269  SCE: 669.298  CFE: 5.435  RMS: 6.667  LUFS: 2.510  
 TCN1 (jamendo)         PESQ: 3.351  MRSTFT: 0.547  MSD: 1.164  SCE: 166.483  CFE: 2.382  RMS: 3.495  LUFS: 1.609  
 TCN2 (jamendo)         PESQ: 3.323  MRSTFT: 0.559  MSD: 1.181  SCE: 164.578  CFE: 2.360  RMS: 3.136  LUFS: 1.501  
 SPSA (jamendo)         PESQ: 3.123  MRSTFT: 0.712  MSD: 1.520  SCE: 220.400  CFE: 4.087  RMS: 3.003  LUFS: 1.396  
 proxy0 (jamendo)       PESQ: 2.941  MRSTFT: 0.773  MSD: 1.611  SCE: 221.463  CFE: 4.070  RMS: 3.118  LUFS: 1.482  
 Proxy1 (jamendo)       PESQ: 2.828  MRSTFT: 1.074  MSD: 2.425  SCE: 423.571  CFE: 5.936  RMS: 6.143  LUFS: 2.961  
 Proxy2 (jamendo)       PESQ: 2.792  MRSTFT: 1.002  MSD: 2.121  SCE: 291.825  CFE: 6.212  RMS: 3.480  LUFS: 1.702  
 Autodiff (jamendo)     PESQ: 3.348  MRSTFT: 0.500  MSD: 1.145  SCE: 154.312  CFE: 4.247  RMS: 2.451  LUFS: 1.098  
 Baseline (jamendo)     PESQ: 2.841  MRSTFT: 0.878  MSD: 1.995  SCE: 254.154  CFE: 3.615  RMS: 3.750  LUFS: 1.531  
--------------------------------
Evaluation complete.

New Mel STFT for MSD
1000

 Corrupt                PESQ: 2.849  MRSTFT: 1.175  MSD: 6.186  SCE: 669.298  CFE: 5.435  RMS: 6.667  LUFS: 2.510
 TCN1 (jamendo)         PESQ: 3.351  MRSTFT: 0.547  MSD: 2.480  SCE: 166.480  CFE: 2.382  RMS: 3.495  LUFS: 1.609
 TCN2 (jamendo)         PESQ: 3.320  MRSTFT: 0.559  MSD: 2.485  SCE: 164.577  CFE: 2.360  RMS: 3.136  LUFS: 1.501
 SPSA (jamendo)         PESQ: 3.123  MRSTFT: 0.712  MSD: 3.203  SCE: 220.400  CFE: 4.087  RMS: 3.003  LUFS: 1.396
 proxy0 (jamendo)       PESQ: 2.941  MRSTFT: 0.773  MSD: 2.965  SCE: 221.462  CFE: 4.070  RMS: 3.118  LUFS: 1.482
 Proxy1 (jamendo)       PESQ: 2.828  MRSTFT: 1.074  MSD: 7.014  SCE: 423.571  CFE: 5.936  RMS: 6.143  LUFS: 2.961
 Proxy2 (jamendo)       PESQ: 2.793  MRSTFT: 1.002  MSD: 5.180  SCE: 291.825  CFE: 6.212  RMS: 3.480  LUFS: 1.702
 Autodiff (jamendo)     PESQ: 3.348  MRSTFT: 0.500  MSD: 2.426  SCE: 154.307  CFE: 4.247  RMS: 2.451  LUFS: 1.098
 Baseline (jamendo)     PESQ: 2.839  MRSTFT: 0.878  MSD: 4.285  SCE: 254.154  CFE: 3.615  RMS: 3.750  LUFS: 1.531
--------------------------------
Evaluation complete.

 Corrupt                & 2.849  & 1.175  & 6.186  & 669.298    & 6.667  & 2.510
 TCN1 (jamendo)         & 3.351  & 0.547  & 2.480  & 166.480    & 3.495  & 1.609
 TCN2 (jamendo)         & 3.320  & 0.559  & 2.485  & 164.577    & 3.136  & 1.501
 SPSA (jamendo)         & 3.123  & 0.712  & 3.203  & 220.400    & 3.003  & 1.396
 proxy0 (jamendo)       & 2.941  & 0.773  & 2.965  & 221.462    & 3.118  & 1.482
 Proxy1 (jamendo)       & 2.828  & 1.074  & 7.014  & 423.571    & 6.143  & 2.961
 Proxy2 (jamendo)       & 2.793  & 1.002  & 5.180  & 291.825    & 3.480  & 1.702
 Autodiff (jamendo)     & 3.348  & 0.500  & 2.426  & 154.307    & 2.451  & 1.098
 Baseline (jamendo)     & 2.839  & 0.878  & 4.285  & 254.154    & 3.750  & 1.531
--------------------------------
Evaluation complete.


-------------------------------- with music proxies
1000

 Corrupt                PESQ: 2.927  MRSTFT: 1.198  MSD: 6.088  SCE: 646.464  CFE: 5.754  RMS: 6.695  LUFS: 2.518  
 TCN1 (jamendo)         PESQ: 3.402  MRSTFT: 0.547  MSD: 2.294  SCE: 160.940  CFE: 2.598  RMS: 3.261  LUFS: 1.483  
 TCN2 (jamendo)         PESQ: 3.390  MRSTFT: 0.548  MSD: 2.278  SCE: 152.314  CFE: 2.494  RMS: 2.951  LUFS: 1.397  
 SPSA (jamendo)         PESQ: 3.173  MRSTFT: 0.716  MSD: 3.024  SCE: 210.077  CFE: 4.549  RMS: 2.809  LUFS: 1.344  
 proxy0 (jamendo)       PESQ: 2.926  MRSTFT: 0.787  MSD: 2.838  SCE: 221.322  CFE: 4.426  RMS: 2.785  LUFS: 1.390  
 proxy1 (jamendo)       PESQ: 2.819  MRSTFT: 1.092  MSD: 6.791  SCE: 395.326  CFE: 6.543  RMS: 6.276  LUFS: 3.032  
 proxy2 (jamendo)       PESQ: 2.833  MRSTFT: 1.016  MSD: 5.005  SCE: 280.831  CFE: 6.675  RMS: 3.377  LUFS: 1.634  
 proxy0m (jamendo)      PESQ: 2.765  MRSTFT: 0.845  MSD: 3.211  SCE: 255.230  CFE: 4.972  RMS: 3.227  LUFS: 1.608  
 proxy1m (jamendo)      PESQ: 2.532  MRSTFT: 1.166  MSD: 7.070  SCE: 591.900  CFE: 11.163  RMS: 5.660  LUFS: 2.593  
 proxy2m (jamendo)      PESQ: 2.648  MRSTFT: 1.137  MSD: 6.368  SCE: 605.618  CFE: 10.520  RMS: 5.903  LUFS: 2.587  
 Autodiff (jamendo)     PESQ: 3.355  MRSTFT: 0.488  MSD: 2.149  SCE: 144.740  CFE: 4.373  RMS: 2.167  LUFS: 1.005  
 Baseline (jamendo)     PESQ: 2.849  MRSTFT: 0.925  MSD: 4.422  SCE: 263.193  CFE: 4.148  RMS: 4.254  LUFS: 1.706  
--------------------------------
Evaluation complete.

 Corrupt                & 2.927  & 1.198  & 6.088  & 646.464  & 6.695  & 2.518  
 TCN1 (jamendo)         & 3.402  & 0.547  & 2.294  & 160.940  & 3.261  & 1.483  
 TCN2 (jamendo)         & 3.390  & 0.548  & 2.278  & 152.314  & 2.951  & 1.397  
 SPSA (jamendo)         & 3.173  & 0.716  & 3.024  & 210.077  & 2.809  & 1.344  
 proxy0 (jamendo)       & 2.926  & 0.787  & 2.838  & 221.322  & 2.785  & 1.390  
 proxy1 (jamendo)       & 2.819  & 1.092  & 6.791  & 395.326  & 6.276  & 3.032  
 proxy2 (jamendo)       & 2.833  & 1.016  & 5.005  & 280.831  & 3.377  & 1.634  
 proxy0m (jamendo)      & 2.765  & 0.845  & 3.211  & 255.230  & 3.227  & 1.608  
 proxy1m (jamendo)      & 2.532  & 1.166  & 7.070  & 591.900  & 5.660  & 2.593  
 proxy2m (jamendo)      & 2.648  & 1.137  & 6.368  & 605.618  & 5.903  & 2.587  
 Autodiff (jamendo)     & 3.355  & 0.488  & 2.149  & 144.740  & 2.167  & 1.005  
 Baseline (jamendo)     & 2.849  & 0.925  & 4.422  & 263.193  & 4.254  & 1.706  

## MUSDB18 @ 44.1kHz (train)

CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/ \
--root_dir /import/c4dm-datasets/deepafx_st \
--gpu \
--dataset musdb18_44100 \
--dataset_dir musdb18_44100/ \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset train \
--length 262144 \
--save \
--ext wav \
--output /import/c4dm-datasets/deepafx_st/

1000

 Corrupt                PESQ: 2.900  MRSTFT: 1.252  MSD: 4.342  SCE: 1088.327  CFE: 5.158  RMS: 5.940  LUFS: 2.312  
 TCN1 (musdb18_44100)   PESQ: 3.121  MRSTFT: 0.896  MSD: 2.956  SCE: 730.446  CFE: 3.594  RMS: 4.548  LUFS: 2.231  
 TCN2 (musdb18_44100)   PESQ: 3.107  MRSTFT: 0.917  MSD: 2.986  SCE: 749.454  CFE: 3.584  RMS: 4.208  LUFS: 2.061  
 SPSA (musdb18_44100)   PESQ: 3.126  MRSTFT: 0.789  MSD: 2.321  SCE: 574.392  CFE: 4.198  RMS: 2.925  LUFS: 1.394  
 proxy0 (musdb18_44100) PESQ: 2.804  MRSTFT: 0.950  MSD: 2.778  SCE: 742.068  CFE: 4.561  RMS: 3.835  LUFS: 1.921  
 proxy1 (musdb18_44100) PESQ: 2.853  MRSTFT: 1.165  MSD: 4.852  SCE: 1005.729  CFE: 7.319  RMS: 6.451  LUFS: 3.269  
 proxy2 (musdb18_44100) PESQ: 2.857  MRSTFT: 1.045  MSD: 3.809  SCE: 617.259  CFE: 6.184  RMS: 3.932  LUFS: 1.971  
 proxy0m (musdb18_44100) PESQ: 2.791  MRSTFT: 0.946  MSD: 2.800  SCE: 757.082  CFE: 4.798  RMS: 4.209  LUFS: 2.127  
 proxy1m (musdb18_44100) PESQ: 2.493  MRSTFT: 1.198  MSD: 5.090  SCE: 1021.863  CFE: 11.674  RMS: 5.585  LUFS: 2.731  
 proxy2m (musdb18_44100) PESQ: 2.575  MRSTFT: 1.225  MSD: 5.450  SCE: 1172.493  CFE: 12.245  RMS: 5.973  LUFS: 2.913  
 Autodiff (musdb18_44100) PESQ: 3.396  MRSTFT: 0.608  MSD: 1.695  SCE: 456.131  CFE: 4.170  RMS: 2.559  LUFS: 1.197  
 Baseline (musdb18_44100) PESQ: 2.994  MRSTFT: 0.821  MSD: 3.052  SCE: 379.400  CFE: 3.871  RMS: 4.078  LUFS: 1.665  
--------------------------------
Evaluation complete.

 Corrupt                & 2.900  & 1.252  & 4.342  & 1088.327    & 5.940  & 2.312  
 TCN1 (musdb18_44100)   & 3.121  & 0.896  & 2.956  & 730.446    & 4.548  & 2.231  
 TCN2 (musdb18_44100)   & 3.107  & 0.917  & 2.986  & 749.454   & 4.208  & 2.061  
 SPSA (musdb18_44100)   & 3.126  & 0.789  & 2.321  & 574.392   & 2.925  & 1.394  
 proxy0 (musdb18_44100) & 2.804  & 0.950  & 2.778  & 742.068   & 3.835  & 1.921  
 proxy1 (musdb18_44100) & 2.853  & 1.165  & 4.852  & 1005.729   & 6.451  & 3.269  
 proxy2 (musdb18_44100) & 2.857  & 1.045  & 3.809  & 617.259   & 3.932  & 1.971  
 proxy0m (musdb18_44100) & 2.791  & 0.946  & 2.800  & 757.082   & 4.209  & 2.127  
 proxy1m (musdb18_44100) & 2.493  & 1.198  & 5.090  & 1021.863    & 5.585  & 2.731  
 proxy2m (musdb18_44100) & 2.575  & 1.225  & 5.450  & 1172.493   & 5.973  & 2.913  
 Autodiff (musdb18_44100) & 3.396  & 0.608  & 1.695  & 456.131    & 2.559  & 1.197  
 Baseline (musdb18_44100) & 2.994  & 0.821  & 3.052  & 379.400   & 4.078 & 1.665  


## MUSDB18 @ 24kHz (train)
quota
CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/ \
--root_dir /import/c4dm-datasets/deepafx_st \
--gpu \
--dataset musdb18_24000 \
--dataset_dir musdb18_24000/ \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset train \
--length 131072 \
--save \
--ext wav \
--output /import/c4dm-datasets/deepafx_st/

1000

 Corrupt                PESQ: 2.925  MRSTFT: 1.237  MSD: 6.452  SCE: 865.043  CFE: 4.969  RMS: 5.658  LUFS: 2.119  
 TCN1 (musdb18_24000)   PESQ: 3.506  MRSTFT: 0.501  MSD: 2.564  SCE: 205.675  CFE: 2.251  RMS: 3.483  LUFS: 1.657  
 TCN2 (musdb18_24000)   PESQ: 3.474  MRSTFT: 0.520  MSD: 2.588  SCE: 194.676  CFE: 2.338  RMS: 3.098  LUFS: 1.533  
 SPSA (musdb18_24000)   PESQ: 3.290  MRSTFT: 0.690  MSD: 3.340  SCE: 258.120  CFE: 3.590  RMS: 2.851  LUFS: 1.417  
 proxy0 (musdb18_24000) PESQ: 3.083  MRSTFT: 0.709  MSD: 3.075  SCE: 280.187  CFE: 3.672  RMS: 3.188  LUFS: 1.577  
 Proxy1 (musdb18_24000) PESQ: 2.936  MRSTFT: 1.085  MSD: 7.042  SCE: 536.682  CFE: 5.264  RMS: 5.913  LUFS: 2.865  
 Proxy2 (musdb18_24000) PESQ: 2.972  MRSTFT: 1.036  MSD: 5.544  SCE: 361.326  CFE: 5.247  RMS: 3.547  LUFS: 1.780  
 Autodiff (musdb18_24000) PESQ: 3.522  MRSTFT: 0.460  MSD: 2.269  SCE: 194.226  CFE: 3.316  RMS: 2.309  LUFS: 1.116  
 Baseline (musdb18_24000) PESQ: 3.059  MRSTFT: 0.811  MSD: 4.069  SCE: 261.277  CFE: 3.133  RMS: 3.309  LUFS: 1.352  
--------------------------------
Evaluation complete.


## Jamendo @ 44.1kHz (test)

CUDA_VISIBLE_DEVICES=4 python scripts/eval.py \
/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/ \
--root_dir /import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/ \
--gpu \
--dataset jamendo_44100 \
--dataset_dir mtg-jamendo_44100/ \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset test \
--length 262144 \
--save \
--ext wav \
--output /import/c4dm-datasets/deepafx_st/

1000

 Corrupt                PESQ: 2.874  MRSTFT: 1.109  MSD: 4.454  SCE: 767.664  CFE: 5.587  RMS: 6.793  LUFS: inf  
 TCN1 (jamendo_44100)   PESQ: 3.168  MRSTFT: 0.876  MSD: 2.921  SCE: 494.368  CFE: 3.459  RMS: 4.372  LUFS: 2.070  
 TCN2 (jamendo_44100)   PESQ: 3.123  MRSTFT: 0.903  MSD: 2.973  SCE: 517.350  CFE: 3.417  RMS: 4.084  LUFS: 1.899  
 SPSA (jamendo_44100)   PESQ: 3.172  MRSTFT: 0.759  MSD: 2.458  SCE: 386.229  CFE: 4.634  RMS: 2.839  LUFS: 1.311  
 proxy0 (jamendo_44100) PESQ: 2.764  MRSTFT: 1.033  MSD: 2.869  SCE: 488.273  CFE: 4.398  RMS: 3.710  LUFS: 1.824  
 proxy1 (jamendo_44100) PESQ: 2.865  MRSTFT: 1.101  MSD: 5.194  SCE: 689.232  CFE: 7.534  RMS: 6.792  LUFS: 3.365  
 proxy2 (jamendo_44100) PESQ: 2.888  MRSTFT: 0.977  MSD: 3.883  SCE: 429.862  CFE: 7.022  RMS: 3.480  LUFS: 1.709  
 proxy0m (jamendo_44100) PESQ: 2.699  MRSTFT: 1.042  MSD: 2.928  SCE: 497.022  CFE: 4.798  RMS: 3.942  LUFS: 1.961  
 proxy1m (jamendo_44100) PESQ: 2.512  MRSTFT: 1.148  MSD: 5.385  SCE: 854.932  CFE: 12.615  RMS: 5.940  LUFS: 2.740  
 proxy2m (jamendo_44100) PESQ: 2.625  MRSTFT: 1.133  MSD: 5.512  SCE: 844.595  CFE: 12.405  RMS: 6.417  LUFS: 2.876  
 Autodiff (jamendo_44100) PESQ: 3.400  MRSTFT: 0.585  MSD: 1.824  SCE: 304.393  CFE: 4.126  RMS: 2.425  LUFS: 1.106  
 Baseline (jamendo_44100) PESQ: 2.931  MRSTFT: 0.887  MSD: 3.355  SCE: 341.957  CFE: 4.474  RMS: 4.749  LUFS: 1.882  
--------------------------------
Evaluation complete.


 Corrupt                & 2.874  & 1.109  & 4.454  & 767.664  & 6.793  & inf  
 TCN1 (jamendo_44100)   & 3.168  & 0.876  & 2.921  & 494.368    & 4.372  & 2.070  
 TCN2 (jamendo_44100)   & 3.123  & 0.903  & 2.973  & 517.350   & 4.084  & 1.899  
 SPSA (jamendo_44100)   & 3.172  & 0.759  & 2.458  & 386.229   & 2.839  & 1.311  
 proxy0 (jamendo_44100) & 2.764  & 1.033  & 2.869  & 488.273  & 3.710  & 1.824  
 proxy1 (jamendo_44100) & 2.865  & 1.101  & 5.194  & 689.232    & 6.792  & 3.365  
 proxy2 (jamendo_44100) & 2.888  & 0.977  & 3.883  & 429.862    & 3.480  & 1.709  
 proxy0m (jamendo_44100) & 2.699  & 1.042  & 2.928  & 497.022    & 3.942  & 1.961  
 proxy1m (jamendo_44100) & 2.512  & 1.148  & 5.385  & 854.932   & 5.940  & 2.740  
 proxy2m (jamendo_44100) & 2.625  & 1.133  & 5.512  & 844.595   & 6.417  & 2.876  
 Autodiff (jamendo_44100) & 3.400  & 0.585  & 1.824  & 304.393    & 2.425  & 1.106  
 Baseline (jamendo_44100) & 2.931  & 0.887  & 3.355  & 341.957   & 4.749  & 1.882  


## Probes 
CUDA_VISIBLE_DEVICES=0 python scripts/eval_probes.py \
--ckpt_dir /import/c4dm-datasets/deepafx_st/logs/probes_new/ \
--eval_dataset /import/c4dm-datasets/deepafx_st/daps_24000_styles_1000_diverse/ \
--subset test \
--output_dir probes \
--gpu \

true acc: 100.00%  f1: 1.00
              precision    recall  f1-score   support

   broadcast       1.00      1.00      1.00       100
   telephone       1.00      1.00      1.00       100
     neutral       1.00      1.00      1.00       100
      bright       1.00      1.00      1.00       100
        warm       1.00      1.00      1.00       100

    accuracy                           1.00       500
   macro avg       1.00      1.00      1.00       500
weighted avg       1.00      1.00      1.00       500

cdpam-mlp acc: 100.00%  f1: 1.00
              precision    recall  f1-score   support

   broadcast       1.00      1.00      1.00       100
   telephone       1.00      1.00      1.00       100
     neutral       1.00      1.00      1.00       100
      bright       1.00      1.00      1.00       100
        warm       1.00      1.00      1.00       100

    accuracy                           1.00       500
   macro avg       1.00      1.00      1.00       500
weighted avg       1.00      1.00      1.00       500

deepafx_st-linear acc: 97.60%  f1: 0.98
              precision    recall  f1-score   support

   broadcast       0.90      0.99      0.94       100
   telephone       1.00      1.00      1.00       100
     neutral       0.99      0.89      0.94       100
      bright       1.00      1.00      1.00       100
        warm       1.00      1.00      1.00       100

    accuracy                           0.98       500
   macro avg       0.98      0.98      0.98       500
weighted avg       0.98      0.98      0.98       500

openl3-mlp acc: 45.60%  f1: 0.40
              precision    recall  f1-score   support

   broadcast       0.38      0.30      0.33       100
   telephone       0.36      0.65      0.46       100
     neutral       0.30      0.27      0.28       100
      bright       0.71      1.00      0.83       100
        warm       1.00      0.06      0.11       100

    accuracy                           0.46       500
   macro avg       0.55      0.46      0.40       500
weighted avg       0.55      0.46      0.40       500

random_mel-mlp acc: 81.40%  f1: 0.79
              precision    recall  f1-score   support

   broadcast       0.53      0.89      0.66       100
   telephone       1.00      1.00      1.00       100
     neutral       0.69      0.22      0.33       100
      bright       0.97      1.00      0.99       100
        warm       0.99      0.96      0.97       100

    accuracy                           0.81       500
   macro avg       0.84      0.81      0.79       500
weighted avg       0.84      0.81      0.79       500

openl3-linear acc: 42.00%  f1: 0.37
              precision    recall  f1-score   support

   broadcast       0.28      0.18      0.22       100
   telephone       0.33      0.58      0.42       100
     neutral       0.26      0.27      0.26       100
      bright       0.67      1.00      0.80       100
        warm       1.00      0.07      0.13       100

    accuracy                           0.42       500
   macro avg       0.51      0.42      0.37       500
weighted avg       0.51      0.42      0.37       500

random_mel-linear acc: 40.00%  f1: 0.23
              precision    recall  f1-score   support

   broadcast       0.00      0.00      0.00       100
   telephone       0.46      1.00      0.63       100
     neutral       0.00      0.00      0.00       100
      bright       0.36      1.00      0.52       100
        warm       0.00      0.00      0.00       100

    accuracy                           0.40       500
   macro avg       0.16      0.40      0.23       500
weighted avg       0.16      0.40      0.23       500

cdpam-linear acc: 64.20%  f1: 0.58
              precision    recall  f1-score   support

   broadcast       0.49      0.57      0.53       100
   telephone       1.00      1.00      1.00       100
     neutral       0.41      0.65      0.50       100
      bright       0.80      0.99      0.88       100
        warm       0.00      0.00      0.00       100

    accuracy                           0.64       500
   macro avg       0.54      0.64      0.58       500
weighted avg       0.54      0.64      0.58       500

deepafx_st-mlp acc: 98.20%  f1: 0.98
              precision    recall  f1-score   support

   broadcast       0.92      1.00      0.96       100
   telephone       1.00      1.00      1.00       100
     neutral       1.00      0.91      0.95       100
      bright       1.00      1.00      1.00       100
        warm       1.00      1.00      1.00       100

    accuracy                           0.98       500
   macro avg       0.98      0.98      0.98       500
weighted avg       0.98      0.98      0.98       500



# Updated style case study with averages

broadcast-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:38<00:00,  3.83s/it]
spsa       MSD: 7.570  SCE: 119.476  RMS: 1.916  LUFS: 0.530  
autodiff   MSD: 7.330  SCE: 203.758  RMS: 2.988  LUFS: 1.229  
proxy0     MSD: 7.854  SCE: 373.396  RMS: 4.277  LUFS: 1.942  
Baseline   MSD: 7.783  SCE: 359.935  RMS: 4.892  LUFS: 2.014  
Corrupt    MSD: 8.454  SCE: 282.362  RMS: 4.441  LUFS: 1.709  

broadcast-->telephone
100%|███████████████████████████████████████████| 10/10 [00:42<00:00,  4.28s/it]
spsa       MSD: 6.390  SCE: 153.945  RMS: 12.168  LUFS: 5.746  
autodiff   MSD: 5.990  SCE: 103.907  RMS: 7.123  LUFS: 3.099  
proxy0     MSD: 7.952  SCE: 250.072  RMS: 4.795  LUFS: 2.135  
Baseline   MSD: 6.179  SCE: 204.214  RMS: 5.164  LUFS: 2.596  
Corrupt    MSD: 10.323  SCE: 423.743  RMS: 5.038  LUFS: 2.358  

broadcast-->neutral
100%|███████████████████████████████████████████| 10/10 [00:30<00:00,  3.05s/it]
spsa       MSD: 8.632  SCE: 189.139  RMS: 3.200  LUFS: 1.323  
autodiff   MSD: 8.419  SCE: 261.488  RMS: 5.080  LUFS: 1.973  
proxy0     MSD: 8.577  SCE: 398.681  RMS: 3.362  LUFS: 1.203  
Baseline   MSD: 8.358  SCE: 455.740  RMS: 6.112  LUFS: 2.810  
Corrupt    MSD: 8.877  SCE: 326.650  RMS: 4.517  LUFS: 2.076  

broadcast-->bright
100%|███████████████████████████████████████████| 10/10 [00:28<00:00,  2.85s/it]
spsa       MSD: 4.498  SCE: 1007.458  RMS: 7.614  LUFS: 3.487  
autodiff   MSD: 4.369  SCE: 766.035  RMS: 12.674  LUFS: 5.192  
proxy0     MSD: 5.549  SCE: 1264.155  RMS: 12.701  LUFS: 7.485  
Baseline   MSD: 4.642  SCE: 793.576  RMS: 4.322  LUFS: 2.077  
Corrupt    MSD: 9.014  SCE: 2366.369  RMS: 10.363  LUFS: 2.558  

broadcast-->warm
100%|███████████████████████████████████████████| 10/10 [00:36<00:00,  3.65s/it]
spsa       MSD: 9.387  SCE: 150.020  RMS: 2.981  LUFS: 3.146  
autodiff   MSD: 8.551  SCE: 237.002  RMS: 6.258  LUFS: 3.585  
proxy0     MSD: 9.239  SCE: 327.282  RMS: 20.747  LUFS: 12.137  
Baseline   MSD: 11.344  SCE: 428.374  RMS: 1.965  LUFS: 0.915  
Corrupt    MSD: 10.688  SCE: 911.725  RMS: 13.515  LUFS: 5.340  

telephone-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:24<00:00,  2.42s/it]
spsa       MSD: 8.833  SCE: 247.258  RMS: 2.139  LUFS: 0.943  
autodiff   MSD: 7.513  SCE: 215.320  RMS: 1.897  LUFS: 0.671  
proxy0     MSD: 9.566  SCE: 1528.270  RMS: 6.313  LUFS: 1.872  
Baseline   MSD: 9.376  SCE: 477.766  RMS: 8.829  LUFS: 3.934  
Corrupt    MSD: 10.533  SCE: 340.476  RMS: 6.092  LUFS: 1.597  

telephone-->telephone
100%|███████████████████████████████████████████| 10/10 [00:23<00:00,  2.34s/it]
spsa       MSD: 5.719  SCE: 96.262  RMS: 7.819  LUFS: 3.660  
autodiff   MSD: 6.300  SCE: 93.168  RMS: 3.613  LUFS: 1.572  
proxy0     MSD: 7.124  SCE: 116.400  RMS: 7.718  LUFS: 3.172  
Baseline   MSD: 6.289  SCE: 140.480  RMS: 3.183  LUFS: 1.599  
Corrupt    MSD: 6.701  SCE: 164.703  RMS: 2.821  LUFS: 1.302  

telephone-->neutral
100%|███████████████████████████████████████████| 10/10 [00:22<00:00,  2.22s/it]
spsa       MSD: 9.668  SCE: 268.287  RMS: 1.866  LUFS: 0.936  
autodiff   MSD: 10.733  SCE: 257.832  RMS: 3.852  LUFS: 1.688  
proxy0     MSD: 10.094  SCE: 920.902  RMS: 13.909  LUFS: 5.293  
Baseline   MSD: 10.038  SCE: 222.337  RMS: 5.335  LUFS: 2.283  
Corrupt    MSD: 11.023  SCE: 394.824  RMS: 7.220  LUFS: 2.392  

telephone-->bright
100%|███████████████████████████████████████████| 10/10 [00:27<00:00,  2.74s/it]
spsa       MSD: 5.705  SCE: 1768.279  RMS: 2.928  LUFS: 1.505  
autodiff   MSD: 6.130  SCE: 731.311  RMS: 6.528  LUFS: 2.418  
proxy0     MSD: 6.674  SCE: 1471.950  RMS: 12.486  LUFS: 6.438  
Baseline   MSD: 6.124  SCE: 1207.425  RMS: 5.672  LUFS: 2.707  
Corrupt    MSD: 8.100  SCE: 2675.410  RMS: 7.645  LUFS: 2.821  

telephone-->warm
100%|███████████████████████████████████████████| 10/10 [00:35<00:00,  3.50s/it]
spsa       MSD: 11.877  SCE: 289.413  RMS: 9.670  LUFS: 4.235  
autodiff   MSD: 9.313  SCE: 337.220  RMS: 11.359  LUFS: 5.737  
proxy0     MSD: 11.498  SCE: 1026.376  RMS: 50.331  LUFS: 23.492  
Baseline   MSD: 13.685  SCE: 574.240  RMS: 5.507  LUFS: 2.282  
Corrupt    MSD: 14.749  SCE: 413.867  RMS: 15.833  LUFS: 4.805  

neutral-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:18<00:00,  1.87s/it]
spsa       MSD: 6.754  SCE: 164.936  RMS: 2.478  LUFS: 1.107  
autodiff   MSD: 6.303  SCE: 178.091  RMS: 2.242  LUFS: 0.990  
proxy0     MSD: 7.266  SCE: 681.963  RMS: 2.686  LUFS: 1.476  
Baseline   MSD: 8.260  SCE: 481.361  RMS: 7.533  LUFS: 3.396  
Corrupt    MSD: 8.817  SCE: 343.222  RMS: 9.012  LUFS: 3.951  

neutral-->telephone
100%|███████████████████████████████████████████| 10/10 [00:26<00:00,  2.68s/it]
spsa       MSD: 6.912  SCE: 126.802  RMS: 11.842  LUFS: 5.800  
autodiff   MSD: 6.790  SCE: 103.199  RMS: 5.143  LUFS: 1.898  
proxy0     MSD: 9.230  SCE: 219.290  RMS: 9.954  LUFS: 4.780  
Baseline   MSD: 7.051  SCE: 183.061  RMS: 3.425  LUFS: 1.520  
Corrupt    MSD: 11.376  SCE: 362.114  RMS: 8.370  LUFS: 2.681  

neutral-->neutral
100%|███████████████████████████████████████████| 10/10 [00:24<00:00,  2.47s/it]
spsa       MSD: 9.350  SCE: 337.903  RMS: 3.897  LUFS: 1.398  
autodiff   MSD: 9.818  SCE: 424.667  RMS: 4.416  LUFS: 1.761  
proxy0     MSD: 10.904  SCE: 374.466  RMS: 8.422  LUFS: 3.045  
Baseline   MSD: 9.088  SCE: 494.032  RMS: 3.246  LUFS: 1.274  
Corrupt    MSD: 9.717  SCE: 479.381  RMS: 4.625  LUFS: 2.025  

neutral-->bright
100%|███████████████████████████████████████████| 10/10 [00:19<00:00,  1.95s/it]
spsa       MSD: 5.332  SCE: 674.385  RMS: 5.128  LUFS: 2.732  
autodiff   MSD: 5.217  SCE: 382.314  RMS: 8.524  LUFS: 3.459  
proxy0     MSD: 6.210  SCE: 572.359  RMS: 9.692  LUFS: 4.994  
Baseline   MSD: 6.106  SCE: 739.821  RMS: 4.730  LUFS: 1.907  
Corrupt    MSD: 10.707  SCE: 2353.734  RMS: 14.699  LUFS: 4.520  

neutral-->warm
100%|███████████████████████████████████████████| 10/10 [00:28<00:00,  2.84s/it]
spsa       MSD: 9.096  SCE: 229.472  RMS: 4.478  LUFS: 2.479  
autodiff   MSD: 8.925  SCE: 523.876  RMS: 10.179  LUFS: 4.529  
proxy0     MSD: 8.739  SCE: 439.980  RMS: 16.191  LUFS: 7.227  
Baseline   MSD: 10.488  SCE: 563.679  RMS: 4.369  LUFS: 1.626  
Corrupt    MSD: 11.037  SCE: 546.397  RMS: 9.678  LUFS: 3.205  

bright-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:20<00:00,  2.06s/it]
spsa       MSD: 9.104  SCE: 419.588  RMS: 2.438  LUFS: 1.230  
autodiff   MSD: 8.564  SCE: 289.857  RMS: 1.341  LUFS: 0.803  
proxy0     MSD: 10.200  SCE: 484.384  RMS: 5.106  LUFS: 2.056  
Baseline   MSD: 9.807  SCE: 845.671  RMS: 2.994  LUFS: 1.114  
Corrupt    MSD: 10.234  SCE: 2210.149  RMS: 11.535  LUFS: 2.551  

bright-->telephone
100%|███████████████████████████████████████████| 10/10 [00:32<00:00,  3.21s/it]
spsa       MSD: 6.794  SCE: 352.609  RMS: 8.139  LUFS: 3.858  
autodiff   MSD: 6.149  SCE: 84.044  RMS: 4.881  LUFS: 2.044  
proxy0     MSD: 7.840  SCE: 129.550  RMS: 5.039  LUFS: 1.975  
Baseline   MSD: 6.521  SCE: 242.590  RMS: 3.510  LUFS: 1.372  
Corrupt    MSD: 8.814  SCE: 2333.702  RMS: 7.755  LUFS: 2.493  

bright-->neutral
100%|███████████████████████████████████████████| 10/10 [00:24<00:00,  2.47s/it]
spsa       MSD: 9.321  SCE: 273.250  RMS: 2.907  LUFS: 0.970  
autodiff   MSD: 9.645  SCE: 157.400  RMS: 5.455  LUFS: 2.232  
proxy0     MSD: 9.179  SCE: 184.585  RMS: 18.288  LUFS: 7.764  
Baseline   MSD: 9.479  SCE: 375.173  RMS: 2.159  LUFS: 0.696  
Corrupt    MSD: 9.293  SCE: 1956.294  RMS: 12.616  LUFS: 3.883  

bright-->bright
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.15s/it]
spsa       MSD: 4.704  SCE: 328.348  RMS: 4.520  LUFS: 1.959  
autodiff   MSD: 5.305  SCE: 226.538  RMS: 8.824  LUFS: 4.115  
proxy0     MSD: 5.780  SCE: 727.537  RMS: 9.035  LUFS: 4.352  
Baseline   MSD: 5.243  SCE: 774.760  RMS: 4.432  LUFS: 1.875  
Corrupt    MSD: 5.752  SCE: 691.583  RMS: 4.774  LUFS: 1.930  

bright-->warm
100%|███████████████████████████████████████████| 10/10 [00:29<00:00,  2.97s/it]
spsa       MSD: 10.198  SCE: 539.897  RMS: 7.841  LUFS: 3.296  
autodiff   MSD: 8.774  SCE: 305.210  RMS: 8.212  LUFS: 4.072  
proxy0     MSD: 11.001  SCE: 1614.079  RMS: 47.078  LUFS: 19.311  
Baseline   MSD: 11.247  SCE: 701.682  RMS: 4.272  LUFS: 1.730  
Corrupt    MSD: 11.401  SCE: 2321.016  RMS: 21.029  LUFS: 6.913  

warm-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:18<00:00,  1.86s/it]
spsa       MSD: 8.570  SCE: 191.909  RMS: 4.233  LUFS: 1.809  
autodiff   MSD: 7.915  SCE: 166.873  RMS: 1.694  LUFS: 0.883  
proxy0     MSD: 15.288  SCE: 1290.383  RMS: 6.962  LUFS: 4.256  
Baseline   MSD: 9.103  SCE: 527.103  RMS: 6.431  LUFS: 2.825  
Corrupt    MSD: 11.239  SCE: 555.232  RMS: 12.235  LUFS: 5.123  

warm-->telephone
100%|███████████████████████████████████████████| 10/10 [00:24<00:00,  2.42s/it]
spsa       MSD: 6.636  SCE: 124.104  RMS: 13.568  LUFS: 7.087  
autodiff   MSD: 6.827  SCE: 90.656  RMS: 4.065  LUFS: 1.877  
proxy0     MSD: 7.621  SCE: 297.015  RMS: 12.667  LUFS: 5.786  
Baseline   MSD: 8.053  SCE: 358.172  RMS: 6.659  LUFS: 2.548  
Corrupt    MSD: 13.733  SCE: 533.204  RMS: 17.164  LUFS: 5.107  

warm-->neutral
100%|███████████████████████████████████████████| 10/10 [00:20<00:00,  2.10s/it]
spsa       MSD: 8.977  SCE: 153.949  RMS: 4.014  LUFS: 1.645  
autodiff   MSD: 10.304  SCE: 331.416  RMS: 5.010  LUFS: 2.185  
proxy0     MSD: 19.354  SCE: 1478.394  RMS: 10.675  LUFS: 6.258  
Baseline   MSD: 10.615  SCE: 546.162  RMS: 7.670  LUFS: 2.573  
Corrupt    MSD: 11.586  SCE: 328.435  RMS: 9.091  LUFS: 3.133  

warm-->bright
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.14s/it]
spsa       MSD: 6.275  SCE: 771.123  RMS: 3.469  LUFS: 2.794  
autodiff   MSD: 6.093  SCE: 625.806  RMS: 7.622  LUFS: 3.542  
proxy0     MSD: 10.266  SCE: 767.358  RMS: 15.424  LUFS: 8.371  
Baseline   MSD: 8.410  SCE: 1063.015  RMS: 12.834  LUFS: 4.710  
Corrupt    MSD: 12.126  SCE: 2773.822  RMS: 21.162  LUFS: 6.825  

warm-->warm
100%|███████████████████████████████████████████| 10/10 [00:27<00:00,  2.75s/it]
spsa       MSD: 8.804  SCE: 228.748  RMS: 2.723  LUFS: 1.755  
autodiff   MSD: 8.988  SCE: 350.737  RMS: 3.947  LUFS: 1.725  
proxy0     MSD: 8.415  SCE: 289.968  RMS: 5.926  LUFS: 2.945  
Baseline   MSD: 9.734  SCE: 314.220  RMS: 3.452  LUFS: 1.352  
Corrupt    MSD: 9.752  SCE: 395.311  RMS: 5.167  LUFS: 2.343  

----- Averages ---- DAPS
autodiff   MSD: 7.611  SCE: 297.909  RMS: 5.717  LUFS: 2.531  
spsa       MSD: 7.804  SCE: 368.262  RMS: 5.359  LUFS: 2.617  
proxy0     MSD: 9.257  SCE: 689.152  RMS: 12.791  LUFS: 5.991  
Baseline   MSD: 8.521  SCE: 522.984  RMS: 5.148  LUFS: 2.149  
Corrupt    MSD: 10.162  SCE: 1059.349  RMS: 9.856  LUFS: 3.346

Global seed set to 16

broadcast-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:33<00:00,  3.33s/it]
spsa       MSD: 7.371  SCE: 172.457  RMS: 1.926  LUFS: 1.073  
autodiff   MSD: 7.017  SCE: 218.527  RMS: 3.325  LUFS: 1.112  
proxy0     MSD: 7.861  SCE: 398.562  RMS: 2.983  LUFS: 1.336  
Baseline   MSD: 8.148  SCE: 402.896  RMS: 6.979  LUFS: 3.053  
Corrupt    MSD: 9.118  SCE: 214.410  RMS: 9.975  LUFS: 4.143  

broadcast-->telephone
100%|███████████████████████████████████████████| 10/10 [00:33<00:00,  3.39s/it]
spsa       MSD: 5.549  SCE: 104.162  RMS: 9.593  LUFS: 4.933  
autodiff   MSD: 7.373  SCE: 132.311  RMS: 7.241  LUFS: 2.998  
proxy0     MSD: 11.958  SCE: 244.265  RMS: 15.485  LUFS: 7.202  
Baseline   MSD: 7.585  SCE: 166.084  RMS: 3.754  LUFS: 1.647  
Corrupt    MSD: 10.272  SCE: 491.861  RMS: 5.794  LUFS: 2.461  

broadcast-->neutral
100%|███████████████████████████████████████████| 10/10 [00:32<00:00,  3.25s/it]
spsa       MSD: 8.019  SCE: 204.897  RMS: 2.908  LUFS: 1.362  
autodiff   MSD: 8.441  SCE: 264.969  RMS: 3.701  LUFS: 1.647  
proxy0     MSD: 8.328  SCE: 368.036  RMS: 4.054  LUFS: 1.705  
Baseline   MSD: 7.784  SCE: 270.785  RMS: 7.047  LUFS: 3.248  
Corrupt    MSD: 7.904  SCE: 371.207  RMS: 6.007  LUFS: 2.700  

broadcast-->bright
100%|███████████████████████████████████████████| 10/10 [00:26<00:00,  2.63s/it]
spsa       MSD: 6.225  SCE: 490.014  RMS: 2.952  LUFS: 1.385  
autodiff   MSD: 5.833  SCE: 318.524  RMS: 5.973  LUFS: 2.489  
proxy0     MSD: 7.602  SCE: 630.675  RMS: 9.428  LUFS: 5.563  
Baseline   MSD: 6.505  SCE: 892.677  RMS: 4.263  LUFS: 2.216  
Corrupt    MSD: 9.665  SCE: 1928.327  RMS: 8.414  LUFS: 2.670  

broadcast-->warm
100%|███████████████████████████████████████████| 10/10 [00:45<00:00,  4.58s/it]
spsa       MSD: 11.026  SCE: 170.762  RMS: 4.956  LUFS: 3.221  
autodiff   MSD: 10.065  SCE: 147.897  RMS: 9.530  LUFS: 5.482  
proxy0     MSD: 10.475  SCE: 246.926  RMS: 15.965  LUFS: 9.458  
Baseline   MSD: 10.470  SCE: 667.165  RMS: 9.631  LUFS: 4.567  
Corrupt    MSD: 10.871  SCE: 471.331  RMS: 12.628  LUFS: 5.300  

telephone-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:19<00:00,  1.99s/it]
spsa       MSD: 8.465  SCE: 279.417  RMS: 1.311  LUFS: 0.910  
autodiff   MSD: 7.611  SCE: 218.598  RMS: 2.633  LUFS: 0.895  
proxy0     MSD: 9.983  SCE: 972.418  RMS: 8.302  LUFS: 2.970  
Baseline   MSD: 10.099  SCE: 452.687  RMS: 7.886  LUFS: 3.718  
Corrupt    MSD: 10.465  SCE: 413.987  RMS: 6.029  LUFS: 2.001  

telephone-->telephone
100%|███████████████████████████████████████████| 10/10 [00:20<00:00,  2.05s/it]
spsa       MSD: 4.506  SCE: 138.731  RMS: 4.951  LUFS: 2.380  
autodiff   MSD: 4.956  SCE: 61.731  RMS: 3.884  LUFS: 1.293  
proxy0     MSD: 6.587  SCE: 132.168  RMS: 8.864  LUFS: 3.503  
Baseline   MSD: 6.005  SCE: 119.214  RMS: 5.546  LUFS: 2.056  
Corrupt    MSD: 6.553  SCE: 280.447  RMS: 4.834  LUFS: 1.827  

telephone-->neutral
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.11s/it]
spsa       MSD: 9.945  SCE: 313.686  RMS: 3.363  LUFS: 1.403  
autodiff   MSD: 9.247  SCE: 308.933  RMS: 4.673  LUFS: 1.794  
proxy0     MSD: 10.991  SCE: 864.742  RMS: 19.190  LUFS: 7.352  
Baseline   MSD: 10.158  SCE: 460.452  RMS: 5.010  LUFS: 2.011  
Corrupt    MSD: 10.700  SCE: 425.533  RMS: 9.457  LUFS: 3.232  

telephone-->bright
100%|███████████████████████████████████████████| 10/10 [00:27<00:00,  2.79s/it]
spsa       MSD: 5.898  SCE: 1389.797  RMS: 3.033  LUFS: 1.591  
autodiff   MSD: 5.869  SCE: 429.690  RMS: 4.644  LUFS: 2.356  
proxy0     MSD: 7.527  SCE: 1012.915  RMS: 13.229  LUFS: 5.968  
Baseline   MSD: 5.564  SCE: 818.532  RMS: 5.503  LUFS: 2.504  
Corrupt    MSD: 8.339  SCE: 2389.284  RMS: 4.828  LUFS: 1.808  

telephone-->warm
100%|███████████████████████████████████████████| 10/10 [00:28<00:00,  2.85s/it]
spsa       MSD: 10.833  SCE: 312.101  RMS: 9.283  LUFS: 3.787  
autodiff   MSD: 9.216  SCE: 374.449  RMS: 8.670  LUFS: 4.076  
proxy0     MSD: 12.483  SCE: 1148.659  RMS: 49.559  LUFS: 21.345  
Baseline   MSD: 13.266  SCE: 470.901  RMS: 4.924  LUFS: 1.677  
Corrupt    MSD: 13.197  SCE: 522.449  RMS: 18.464  LUFS: 5.796  

neutral-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:18<00:00,  1.87s/it]
spsa       MSD: 6.641  SCE: 231.272  RMS: 2.405  LUFS: 1.172  
autodiff   MSD: 6.455  SCE: 303.714  RMS: 1.542  LUFS: 0.726  
proxy0     MSD: 6.944  SCE: 446.685  RMS: 2.408  LUFS: 1.497  
Baseline   MSD: 8.370  SCE: 677.089  RMS: 5.512  LUFS: 2.459  
Corrupt    MSD: 8.919  SCE: 283.764  RMS: 7.543  LUFS: 3.408  

neutral-->telephone
100%|███████████████████████████████████████████| 10/10 [00:23<00:00,  2.37s/it]
spsa       MSD: 6.548  SCE: 125.255  RMS: 10.246  LUFS: 5.055  
autodiff   MSD: 6.691  SCE: 140.547  RMS: 5.201  LUFS: 2.269  
proxy0     MSD: 9.147  SCE: 200.192  RMS: 8.658  LUFS: 3.882  
Baseline   MSD: 6.413  SCE: 176.637  RMS: 3.044  LUFS: 1.157  
Corrupt    MSD: 11.548  SCE: 253.729  RMS: 10.898  LUFS: 3.087  

neutral-->neutral
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.11s/it]
spsa       MSD: 8.229  SCE: 397.836  RMS: 4.964  LUFS: 2.163  
autodiff   MSD: 8.446  SCE: 547.878  RMS: 4.720  LUFS: 2.147  
proxy0     MSD: 9.892  SCE: 583.782  RMS: 10.960  LUFS: 4.535  
Baseline   MSD: 8.831  SCE: 546.597  RMS: 3.727  LUFS: 1.621  
Corrupt    MSD: 9.035  SCE: 628.128  RMS: 5.517  LUFS: 2.584  

neutral-->bright
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.77s/it]
spsa       MSD: 5.138  SCE: 773.308  RMS: 5.914  LUFS: 3.552  
autodiff   MSD: 5.038  SCE: 454.673  RMS: 9.380  LUFS: 4.732  
proxy0     MSD: 6.190  SCE: 747.949  RMS: 8.446  LUFS: 4.713  
Baseline   MSD: 6.132  SCE: 777.004  RMS: 5.666  LUFS: 1.866  
Corrupt    MSD: 10.296  SCE: 2148.154  RMS: 13.907  LUFS: 3.789  

neutral-->warm
100%|███████████████████████████████████████████| 10/10 [00:40<00:00,  4.05s/it]
spsa       MSD: 12.036  SCE: 303.386  RMS: 6.242  LUFS: 3.284  
autodiff   MSD: 11.478  SCE: 468.910  RMS: 12.385  LUFS: 5.592  
proxy0     MSD: 12.177  SCE: 326.679  RMS: 20.926  LUFS: 9.999  
Baseline   MSD: 12.819  SCE: 929.818  RMS: 9.187  LUFS: 3.835  
Corrupt    MSD: 12.108  SCE: 535.255  RMS: 13.302  LUFS: 5.392  

bright-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:22<00:00,  2.25s/it]
spsa       MSD: 8.609  SCE: 425.804  RMS: 2.318  LUFS: 1.060  
autodiff   MSD: 7.712  SCE: 273.757  RMS: 2.447  LUFS: 0.752  
proxy0     MSD: 9.258  SCE: 650.227  RMS: 5.816  LUFS: 1.639  
Baseline   MSD: 9.889  SCE: 658.042  RMS: 7.105  LUFS: 3.023  
Corrupt    MSD: 9.141  SCE: 2054.307  RMS: 10.704  LUFS: 2.987  

bright-->telephone
100%|███████████████████████████████████████████| 10/10 [00:25<00:00,  2.54s/it]
spsa       MSD: 6.847  SCE: 297.782  RMS: 7.416  LUFS: 3.305  
autodiff   MSD: 6.949  SCE: 80.701  RMS: 4.010  LUFS: 1.586  
proxy0     MSD: 8.813  SCE: 100.100  RMS: 6.955  LUFS: 3.087  
Baseline   MSD: 6.842  SCE: 158.974  RMS: 3.036  LUFS: 1.131  
Corrupt    MSD: 9.380  SCE: 2468.890  RMS: 5.970  LUFS: 2.489  

bright-->neutral
100%|███████████████████████████████████████████| 10/10 [00:21<00:00,  2.14s/it]
spsa       MSD: 11.284  SCE: 419.588  RMS: 2.265  LUFS: 1.078  
autodiff   MSD: 12.047  SCE: 301.471  RMS: 2.768  LUFS: 1.481  
proxy0     MSD: 10.578  SCE: 389.600  RMS: 9.342  LUFS: 3.756  
Baseline   MSD: 10.776  SCE: 445.448  RMS: 2.500  LUFS: 1.274  
Corrupt    MSD: 11.276  SCE: 2091.734  RMS: 10.198  LUFS: 2.402  

bright-->bright
100%|███████████████████████████████████████████| 10/10 [00:18<00:00,  1.89s/it]
spsa       MSD: 5.786  SCE: 219.477  RMS: 2.651  LUFS: 1.053  
autodiff   MSD: 6.340  SCE: 233.348  RMS: 4.010  LUFS: 1.949  
proxy0     MSD: 6.906  SCE: 654.335  RMS: 8.258  LUFS: 4.230  
Baseline   MSD: 6.289  SCE: 474.055  RMS: 2.687  LUFS: 0.994  
Corrupt    MSD: 7.115  SCE: 601.199  RMS: 5.821  LUFS: 1.852  

bright-->warm
100%|███████████████████████████████████████████| 10/10 [00:26<00:00,  2.63s/it]
spsa       MSD: 9.844  SCE: 752.189  RMS: 9.814  LUFS: 4.016  
autodiff   MSD: 8.481  SCE: 267.815  RMS: 8.754  LUFS: 4.293  
proxy0     MSD: 10.927  SCE: 1941.049  RMS: 50.826  LUFS: 20.030  
Baseline   MSD: 11.895  SCE: 437.921  RMS: 3.092  LUFS: 1.207  
Corrupt    MSD: 12.235  SCE: 2533.021  RMS: 20.768  LUFS: 6.394  

warm-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:16<00:00,  1.66s/it]
spsa       MSD: 8.253  SCE: 144.919  RMS: 4.567  LUFS: 1.956  
autodiff   MSD: 7.501  SCE: 255.792  RMS: 2.247  LUFS: 0.889  
proxy0     MSD: 10.815  SCE: 1238.939  RMS: 3.606  LUFS: 2.852  
Baseline   MSD: 10.156  SCE: 567.506  RMS: 12.300  LUFS: 5.272  
Corrupt    MSD: 11.231  SCE: 549.288  RMS: 14.310  LUFS: 6.126  

warm-->telephone
100%|███████████████████████████████████████████| 10/10 [00:18<00:00,  1.86s/it]
spsa       MSD: 6.870  SCE: 166.390  RMS: 11.550  LUFS: 6.152  
autodiff   MSD: 6.178  SCE: 114.869  RMS: 2.939  LUFS: 1.617  
proxy0     MSD: 7.863  SCE: 203.042  RMS: 10.660  LUFS: 4.645  
Baseline   MSD: 9.449  SCE: 471.936  RMS: 8.080  LUFS: 2.670  
Corrupt    MSD: 14.019  SCE: 252.479  RMS: 19.388  LUFS: 6.550  

warm-->neutral
100%|███████████████████████████████████████████| 10/10 [00:19<00:00,  1.96s/it]
spsa       MSD: 9.106  SCE: 166.963  RMS: 4.564  LUFS: 1.977  
autodiff   MSD: 9.523  SCE: 302.502  RMS: 4.357  LUFS: 1.775  
proxy0     MSD: 20.448  SCE: 1060.644  RMS: 10.820  LUFS: 6.420  
Baseline   MSD: 10.240  SCE: 496.880  RMS: 4.583  LUFS: 1.665  
Corrupt    MSD: 11.977  SCE: 709.600  RMS: 6.823  LUFS: 2.232  

warm-->bright
100%|███████████████████████████████████████████| 10/10 [00:17<00:00,  1.72s/it]
spsa       MSD: 6.002  SCE: 751.138  RMS: 5.238  LUFS: 3.321  
autodiff   MSD: 5.313  SCE: 334.639  RMS: 8.967  LUFS: 3.842  
proxy0     MSD: 9.097  SCE: 914.759  RMS: 12.348  LUFS: 6.958  
Baseline   MSD: 8.472  SCE: 1110.614  RMS: 9.309  LUFS: 3.231  
Corrupt    MSD: 13.787  SCE: 2904.344  RMS: 23.853  LUFS: 7.778  

warm-->warm
100%|███████████████████████████████████████████| 10/10 [00:30<00:00,  3.03s/it]
spsa       MSD: 10.273  SCE: 266.991  RMS: 3.600  LUFS: 1.592  
autodiff   MSD: 10.117  SCE: 396.560  RMS: 2.964  LUFS: 1.410  
proxy0     MSD: 11.400  SCE: 437.774  RMS: 5.956  LUFS: 2.307  
Baseline   MSD: 10.142  SCE: 279.552  RMS: 5.683  LUFS: 2.484  
Corrupt    MSD: 10.802  SCE: 518.633  RMS: 4.915  LUFS: 2.210  

----- Averages ---- DAPS
spsa       MSD: 7.972  SCE: 360.733  RMS: 5.121  LUFS: 2.511  
autodiff   MSD: 7.756  SCE: 278.112  RMS: 5.239  LUFS: 2.368  
proxy0     MSD: 9.770  SCE: 636.605  RMS: 12.922  LUFS: 5.878  
Baseline   MSD: 8.892  SCE: 517.179  RMS: 5.842  LUFS: 2.423  
Corrupt    MSD: 10.398  SCE: 1041.654  RMS: 10.414  LUFS: 3.649 

## Style case study on MUSDB18 @ 44.1 kHz

CUDA_VISIBLE_DEVICES=1 python scripts/style_case_study.py \
--ckpt_paths \
"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt" \
--style_audio "/import/c4dm-datasets/deepafx_st/musdb18_44100_styles_100/train" \
--output_dir "/import/c4dm-datasets/deepafx_st/style_case_study_musdb18" \
--sample_rate 44100 \
--gpu \
--save \
--plot \
broadcast-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:38<00:00,  3.87s/it]
autodiff   MSD: 5.268  SCE: 764.455  RMS: 1.486  LUFS: 0.620  
spsa       MSD: 5.717  SCE: 742.577  RMS: 2.132  LUFS: 1.101  
proxy0     MSD: 7.515  SCE: 954.857  RMS: 6.580  LUFS: 3.552  
Baseline   MSD: 6.712  SCE: 324.917  RMS: 6.104  LUFS: 2.953  
Corrupt    MSD: 7.248  SCE: 857.312  RMS: 7.288  LUFS: 3.347  

broadcast-->telephone
100%|███████████████████████████████████████████| 10/10 [01:28<00:00,  8.88s/it]
autodiff   MSD: 6.153  SCE: 165.554  RMS: 3.281  LUFS: 1.483  
spsa       MSD: 7.145  SCE: 414.354  RMS: 9.122  LUFS: 4.119  
proxy0     MSD: 7.849  SCE: 303.187  RMS: 6.100  LUFS: 2.475  
Baseline   MSD: 8.953  SCE: 261.259  RMS: 15.381  LUFS: 6.724  
Corrupt    MSD: 11.741  SCE: 1618.816  RMS: 11.667  LUFS: 5.312  

broadcast-->neutral
100%|███████████████████████████████████████████| 10/10 [01:11<00:00,  7.14s/it]
autodiff   MSD: 6.254  SCE: 721.382  RMS: 2.862  LUFS: 1.141  
spsa       MSD: 6.460  SCE: 825.238  RMS: 2.443  LUFS: 1.190  
proxy0     MSD: 8.555  SCE: 1317.455  RMS: 6.520  LUFS: 3.426  
Baseline   MSD: 7.031  SCE: 238.390  RMS: 7.928  LUFS: 3.514  
Corrupt    MSD: 7.839  SCE: 1029.224  RMS: 5.390  LUFS: 2.837  

broadcast-->bright
100%|███████████████████████████████████████████| 10/10 [00:38<00:00,  3.83s/it]
autodiff   MSD: 2.377  SCE: 1834.532  RMS: 4.029  LUFS: 1.704  
spsa       MSD: 3.088  SCE: 2325.082  RMS: 6.319  LUFS: 2.218  
proxy0     MSD: 3.582  SCE: 1312.444  RMS: 12.568  LUFS: 5.890  
Baseline   MSD: 3.043  SCE: 985.595  RMS: 8.910  LUFS: 3.665  
Corrupt    MSD: 7.679  SCE: 4955.671  RMS: 23.504  LUFS: 7.104  

broadcast-->warm
100%|███████████████████████████████████████████| 10/10 [00:49<00:00,  4.92s/it]
autodiff   MSD: 3.891  SCE: 748.354  RMS: 3.421  LUFS: 1.569  
spsa       MSD: 4.541  SCE: 769.316  RMS: 3.495  LUFS: 1.453  
proxy0     MSD: 4.705  SCE: 1007.030  RMS: 6.546  LUFS: 3.328  
Baseline   MSD: 5.870  SCE: 705.780  RMS: 6.706  LUFS: 2.960  
Corrupt    MSD: 7.728  SCE: 1744.439  RMS: 6.603  LUFS: 2.843  

telephone-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:42<00:00,  4.29s/it]
autodiff   MSD: 5.457  SCE: 795.807  RMS: 1.699  LUFS: 0.687  
spsa       MSD: 6.593  SCE: 589.769  RMS: 2.581  LUFS: 1.151  
proxy0     MSD: 8.515  SCE: 3452.242  RMS: 15.419  LUFS: 6.475  
Baseline   MSD: 9.751  SCE: 1464.298  RMS: 11.973  LUFS: 5.026  
Corrupt    MSD: 11.900  SCE: 1627.915  RMS: 8.630  LUFS: 4.750  

telephone-->telephone
100%|███████████████████████████████████████████| 10/10 [01:06<00:00,  6.63s/it]
autodiff   MSD: 5.490  SCE: 122.907  RMS: 3.295  LUFS: 1.602  
spsa       MSD: 6.032  SCE: 91.678  RMS: 3.282  LUFS: 1.391  
proxy0     MSD: 6.820  SCE: 357.135  RMS: 7.806  LUFS: 3.530  
Baseline   MSD: 6.814  SCE: 253.513  RMS: 6.370  LUFS: 2.765  
Corrupt    MSD: 7.155  SCE: 328.757  RMS: 8.424  LUFS: 3.690  

telephone-->neutral
100%|███████████████████████████████████████████| 10/10 [00:48<00:00,  4.84s/it]
autodiff   MSD: 6.228  SCE: 563.005  RMS: 2.564  LUFS: 1.157  
spsa       MSD: 7.018  SCE: 622.279  RMS: 2.742  LUFS: 1.198  
proxy0     MSD: 8.846  SCE: 2406.702  RMS: 10.261  LUFS: 4.415  
Baseline   MSD: 8.775  SCE: 1237.468  RMS: 4.171  LUFS: 2.020  
Corrupt    MSD: 9.914  SCE: 1318.137  RMS: 6.742  LUFS: 2.769  

telephone-->bright
100%|███████████████████████████████████████████| 10/10 [02:26<00:00, 14.63s/it]
autodiff   MSD: 3.270  SCE: 3387.736  RMS: 8.958  LUFS: 4.675  
spsa       MSD: 3.858  SCE: 4975.919  RMS: 5.703  LUFS: 3.585  
proxy0     MSD: 4.210  SCE: 2814.990  RMS: 5.121  LUFS: 2.485  
Baseline   MSD: 4.113  SCE: 2005.442  RMS: 25.054  LUFS: 11.366  
Corrupt    MSD: 8.285  SCE: 7409.718  RMS: 15.362  LUFS: 5.134  

telephone-->warm
100%|███████████████████████████████████████████| 10/10 [00:30<00:00,  3.07s/it]
autodiff   MSD: 4.053  SCE: 798.839  RMS: 6.723  LUFS: 3.331  
spsa       MSD: 5.598  SCE: 652.353  RMS: 11.710  LUFS: 4.834  
proxy0     MSD: 5.526  SCE: 2542.052  RMS: 48.147  LUFS: 18.399  
Baseline   MSD: 10.660  SCE: 1727.024  RMS: 4.498  LUFS: 2.267  
Corrupt    MSD: 12.201  SCE: 1363.306  RMS: 8.944  LUFS: 3.653  

neutral-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:51<00:00,  5.14s/it]
autodiff   MSD: 5.633  SCE: 473.112  RMS: 1.688  LUFS: 0.884  
spsa       MSD: 6.232  SCE: 620.590  RMS: 2.096  LUFS: 1.177  
proxy0     MSD: 7.190  SCE: 973.757  RMS: 8.096  LUFS: 3.795  
Baseline   MSD: 6.703  SCE: 408.505  RMS: 7.695  LUFS: 3.240  
Corrupt    MSD: 7.213  SCE: 697.631  RMS: 5.424  LUFS: 2.277  

neutral-->telephone
100%|███████████████████████████████████████████| 10/10 [01:12<00:00,  7.22s/it]
autodiff   MSD: 5.293  SCE: 170.508  RMS: 2.995  LUFS: 1.343  
spsa       MSD: 6.183  SCE: 302.104  RMS: 5.546  LUFS: 2.798  
proxy0     MSD: 6.489  SCE: 206.780  RMS: 5.344  LUFS: 2.495  
Baseline   MSD: 7.031  SCE: 291.637  RMS: 6.276  LUFS: 3.036  
Corrupt    MSD: 9.642  SCE: 1073.369  RMS: 7.456  LUFS: 3.109  

neutral-->neutral
100%|███████████████████████████████████████████| 10/10 [01:13<00:00,  7.34s/it]
autodiff   MSD: 6.086  SCE: 596.216  RMS: 2.165  LUFS: 0.993  
spsa       MSD: 6.659  SCE: 817.107  RMS: 2.225  LUFS: 0.850  
proxy0     MSD: 8.871  SCE: 1231.226  RMS: 4.540  LUFS: 2.858  
Baseline   MSD: 6.545  SCE: 396.184  RMS: 5.075  LUFS: 2.517  
Corrupt    MSD: 7.287  SCE: 1037.534  RMS: 3.076  LUFS: 2.261  

neutral-->bright
100%|███████████████████████████████████████████| 10/10 [02:10<00:00, 13.01s/it]
autodiff   MSD: 3.766  SCE: 2586.719  RMS: 5.256  LUFS: 2.830  
spsa       MSD: 3.988  SCE: 2741.861  RMS: 5.056  LUFS: 2.633  
proxy0     MSD: 5.062  SCE: 1127.514  RMS: 3.931  LUFS: 2.326  
Baseline   MSD: 4.613  SCE: 1388.867  RMS: 20.508  LUFS: 9.528  
Corrupt    MSD: 9.180  SCE: 5913.314  RMS: 10.544  LUFS: 3.077  

neutral-->warm
100%|███████████████████████████████████████████| 10/10 [00:32<00:00,  3.26s/it]
autodiff   MSD: 3.895  SCE: 839.517  RMS: 5.932  LUFS: 2.306  
spsa       MSD: 4.252  SCE: 777.918  RMS: 3.545  LUFS: 1.911  
proxy0     MSD: 4.595  SCE: 1163.291  RMS: 9.814  LUFS: 5.204  
Baseline   MSD: 6.711  SCE: 421.702  RMS: 6.794  LUFS: 3.228  
Corrupt    MSD: 10.006  SCE: 1059.995  RMS: 6.357  LUFS: 4.262  

bright-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:29<00:00,  2.94s/it]
autodiff   MSD: 5.007  SCE: 1785.158  RMS: 1.634  LUFS: 0.619  
spsa       MSD: 5.712  SCE: 3068.999  RMS: 3.311  LUFS: 1.212  
proxy0     MSD: 7.551  SCE: 3772.967  RMS: 7.368  LUFS: 3.364  
Baseline   MSD: 7.888  SCE: 918.202  RMS: 4.133  LUFS: 1.998  
Corrupt    MSD: 7.071  SCE: 5855.776  RMS: 12.743  LUFS: 3.426  

bright-->telephone
100%|███████████████████████████████████████████| 10/10 [01:36<00:00,  9.63s/it]
autodiff   MSD: 6.150  SCE: 245.520  RMS: 2.700  LUFS: 1.190  
spsa       MSD: 7.275  SCE: 715.758  RMS: 6.034  LUFS: 2.993  
proxy0     MSD: 9.355  SCE: 269.956  RMS: 5.673  LUFS: 2.097  
Baseline   MSD: 8.040  SCE: 231.228  RMS: 9.257  LUFS: 4.368  
Corrupt    MSD: 10.568  SCE: 4923.880  RMS: 14.471  LUFS: 5.918  

bright-->neutral
100%|███████████████████████████████████████████| 10/10 [00:34<00:00,  3.50s/it]
autodiff   MSD: 6.739  SCE: 1748.574  RMS: 6.367  LUFS: 2.415  
spsa       MSD: 8.106  SCE: 2257.716  RMS: 4.058  LUFS: 1.124  
proxy0     MSD: 10.557  SCE: 2358.070  RMS: 7.158  LUFS: 3.306  
Baseline   MSD: 9.158  SCE: 903.038  RMS: 2.445  LUFS: 1.018  
Corrupt    MSD: 8.662  SCE: 5363.465  RMS: 14.566  LUFS: 3.676  

bright-->bright
100%|███████████████████████████████████████████| 10/10 [00:53<00:00,  5.39s/it]
autodiff   MSD: 2.583  SCE: 842.237  RMS: 3.639  LUFS: 1.679  
spsa       MSD: 2.942  SCE: 1069.083  RMS: 3.446  LUFS: 1.603  
proxy0     MSD: 3.404  SCE: 1263.463  RMS: 7.770  LUFS: 3.352  
Baseline   MSD: 3.194  SCE: 1182.327  RMS: 8.312  LUFS: 3.665  
Corrupt    MSD: 3.611  SCE: 1865.160  RMS: 7.540  LUFS: 3.165  

bright-->warm
100%|███████████████████████████████████████████| 10/10 [00:41<00:00,  4.11s/it]
autodiff   MSD: 3.720  SCE: 918.005  RMS: 5.844  LUFS: 2.801  
spsa       MSD: 4.430  SCE: 1835.436  RMS: 11.297  LUFS: 4.988  
proxy0     MSD: 5.261  SCE: 4984.333  RMS: 29.548  LUFS: 10.549  
Baseline   MSD: 8.866  SCE: 911.462  RMS: 4.441  LUFS: 2.750  
Corrupt    MSD: 8.584  SCE: 5182.931  RMS: 16.600  LUFS: 4.017  

warm-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:36<00:00,  3.66s/it]
autodiff   MSD: 4.536  SCE: 462.483  RMS: 1.149  LUFS: 0.397  
spsa       MSD: 5.336  SCE: 1403.954  RMS: 2.841  LUFS: 1.359  
proxy0     MSD: 9.801  SCE: 857.491  RMS: 8.365  LUFS: 5.247  
Baseline   MSD: 5.814  SCE: 1245.149  RMS: 7.553  LUFS: 2.836  
Corrupt    MSD: 7.006  SCE: 1214.527  RMS: 10.749  LUFS: 4.726  

warm-->telephone
100%|███████████████████████████████████████████| 10/10 [00:56<00:00,  5.65s/it]
autodiff   MSD: 6.247  SCE: 306.690  RMS: 7.223  LUFS: 3.547  
spsa       MSD: 7.720  SCE: 976.353  RMS: 8.565  LUFS: 3.589  
proxy0     MSD: 8.222  SCE: 800.051  RMS: 9.571  LUFS: 4.325  
Baseline   MSD: 10.837  SCE: 814.371  RMS: 13.657  LUFS: 4.271  
Corrupt    MSD: 12.381  SCE: 1682.943  RMS: 11.264  LUFS: 4.622  

warm-->neutral
100%|███████████████████████████████████████████| 10/10 [00:44<00:00,  4.45s/it]
autodiff   MSD: 6.247  SCE: 759.577  RMS: 1.195  LUFS: 0.692  
spsa       MSD: 6.635  SCE: 1124.939  RMS: 2.152  LUFS: 0.646  
proxy0     MSD: 12.783  SCE: 910.850  RMS: 7.942  LUFS: 4.809  
Baseline   MSD: 7.825  SCE: 1237.193  RMS: 4.960  LUFS: 1.748  
Corrupt    MSD: 8.902  SCE: 1197.048  RMS: 4.863  LUFS: 1.927  

warm-->bright
100%|███████████████████████████████████████████| 10/10 [00:20<00:00,  2.04s/it]
autodiff   MSD: 2.768  SCE: 1369.470  RMS: 3.846  LUFS: 2.007  
spsa       MSD: 3.445  SCE: 1910.909  RMS: 1.579  LUFS: 1.279  
proxy0     MSD: 4.475  SCE: 802.250  RMS: 5.425  LUFS: 1.860  
Baseline   MSD: 5.136  SCE: 2548.782  RMS: 13.753  LUFS: 2.340  
Corrupt    MSD: 7.634  SCE: 6192.568  RMS: 21.637  LUFS: 4.702  

warm-->warm
100%|███████████████████████████████████████████| 10/10 [00:44<00:00,  4.49s/it]
autodiff   MSD: 3.388  SCE: 683.876  RMS: 4.456  LUFS: 1.592  
spsa       MSD: 3.400  SCE: 794.539  RMS: 3.175  LUFS: 0.996  
proxy0     MSD: 4.359  SCE: 626.161  RMS: 4.856  LUFS: 2.238  
Baseline   MSD: 3.495  SCE: 782.423  RMS: 4.227  LUFS: 1.738  
Corrupt    MSD: 3.910  SCE: 1368.821  RMS: 5.118  LUFS: 2.271  

----- Averages ---- MUSDB18
autodiff   MSD: 4.820  SCE: 947.609  RMS: 3.776  LUFS: 1.731  
spsa       MSD: 5.535  SCE: 1297.033  RMS: 4.578  LUFS: 2.056  
proxy0     MSD: 6.964  SCE: 1512.650  RMS: 10.019  LUFS: 4.472  
Baseline   MSD: 6.943  SCE: 915.390  RMS: 8.647  LUFS: 3.662  
Corrupt    MSD: 8.534  SCE: 2675.290  RMS: 10.199  LUFS: 3.795 

 ------

Global seed set to 16
Proxy Processor: peq @ fs=24000 Hz
TCN receptive field: 7021 samples  or 292.542 ms
Proxy Processor: comp @ fs=24000 Hz
TCN receptive field: 7021 samples  or 292.542 ms
broadcast-->broadcast
100%|███████████████████████████████████████████| 10/10 [01:18<00:00,  7.87s/it]
autodiff   MSD: 5.214  SCE: 539.719  RMS: 1.776  LUFS: 0.896  
spsa       MSD: 5.499  SCE: 532.192  RMS: 3.119  LUFS: 1.426  
proxy0     MSD: 6.473  SCE: 1390.071  RMS: 7.491  LUFS: 4.578  
Baseline   MSD: 6.449  SCE: 611.495  RMS: 13.743  LUFS: 5.791  
Corrupt    MSD: 6.739  SCE: 1188.642  RMS: 9.744  LUFS: 4.377  

broadcast-->telephone
100%|███████████████████████████████████████████| 10/10 [01:00<00:00,  6.00s/it]
autodiff   MSD: 5.974  SCE: 151.923  RMS: 2.261  LUFS: 1.189  
spsa       MSD: 6.501  SCE: 260.794  RMS: 5.734  LUFS: 2.850  
proxy0     MSD: 7.920  SCE: 273.202  RMS: 4.547  LUFS: 1.867  
Baseline   MSD: 7.449  SCE: 280.922  RMS: 7.111  LUFS: 2.715  
Corrupt    MSD: 10.834  SCE: 1312.504  RMS: 8.379  LUFS: 2.868  

broadcast-->neutral
100%|███████████████████████████████████████████| 10/10 [01:55<00:00, 11.54s/it]
autodiff   MSD: 5.757  SCE: 625.424  RMS: 2.377  LUFS: 1.079  
spsa       MSD: 6.238  SCE: 598.486  RMS: 1.932  LUFS: 0.808  
proxy0     MSD: 7.995  SCE: 1080.859  RMS: 5.854  LUFS: 3.147  
Baseline   MSD: 6.528  SCE: 305.291  RMS: 15.089  LUFS: 6.711  
Corrupt    MSD: 6.470  SCE: 1340.071  RMS: 7.338  LUFS: 3.598  

broadcast-->bright
100%|███████████████████████████████████████████| 10/10 [01:28<00:00,  8.86s/it]
autodiff   MSD: 3.124  SCE: 1826.629  RMS: 6.154  LUFS: 3.618  
spsa       MSD: 3.612  SCE: 2296.033  RMS: 4.474  LUFS: 2.919  
proxy0     MSD: 4.519  SCE: 1239.892  RMS: 5.006  LUFS: 2.656  
Baseline   MSD: 3.695  SCE: 968.008  RMS: 11.576  LUFS: 5.373  
Corrupt    MSD: 8.048  SCE: 5699.976  RMS: 13.067  LUFS: 3.319  

broadcast-->warm
100%|███████████████████████████████████████████| 10/10 [00:25<00:00,  2.52s/it]
autodiff   MSD: 3.129  SCE: 665.939  RMS: 3.023  LUFS: 0.865  
spsa       MSD: 3.759  SCE: 966.383  RMS: 3.144  LUFS: 1.248  
proxy0     MSD: 4.048  SCE: 1058.204  RMS: 6.544  LUFS: 3.036  
Baseline   MSD: 5.674  SCE: 566.713  RMS: 6.871  LUFS: 3.654  
Corrupt    MSD: 8.552  SCE: 1016.717  RMS: 6.865  LUFS: 4.339  

telephone-->broadcast
100%|███████████████████████████████████████████| 10/10 [01:54<00:00, 11.46s/it]
autodiff   MSD: 6.452  SCE: 780.362  RMS: 2.015  LUFS: 1.112  
spsa       MSD: 7.557  SCE: 869.791  RMS: 2.794  LUFS: 1.316  
proxy0     MSD: 9.715  SCE: 2663.854  RMS: 11.838  LUFS: 4.143  
Baseline   MSD: 9.111  SCE: 1393.222  RMS: 17.848  LUFS: 7.498  
Corrupt    MSD: 11.864  SCE: 1874.800  RMS: 7.548  LUFS: 2.650  

telephone-->telephone
100%|███████████████████████████████████████████| 10/10 [01:14<00:00,  7.49s/it]
autodiff   MSD: 5.115  SCE: 112.800  RMS: 2.275  LUFS: 1.213  
spsa       MSD: 5.915  SCE: 155.356  RMS: 3.879  LUFS: 1.523  
proxy0     MSD: 6.797  SCE: 232.531  RMS: 9.969  LUFS: 4.578  
Baseline   MSD: 7.631  SCE: 180.769  RMS: 11.585  LUFS: 4.900  
Corrupt    MSD: 7.546  SCE: 269.839  RMS: 10.740  LUFS: 4.552  

telephone-->neutral
100%|███████████████████████████████████████████| 10/10 [00:44<00:00,  4.42s/it]
autodiff   MSD: 6.357  SCE: 429.069  RMS: 2.349  LUFS: 0.944  
spsa       MSD: 7.289  SCE: 294.847  RMS: 2.050  LUFS: 0.745  
proxy0     MSD: 9.393  SCE: 2912.440  RMS: 12.004  LUFS: 5.530  
Baseline   MSD: 9.369  SCE: 1105.188  RMS: 2.573  LUFS: 1.102  
Corrupt    MSD: 10.772  SCE: 1039.346  RMS: 4.271  LUFS: 1.896  

telephone-->bright
100%|███████████████████████████████████████████| 10/10 [02:11<00:00, 13.12s/it]
autodiff   MSD: 3.781  SCE: 2569.188  RMS: 7.647  LUFS: 3.893  
spsa       MSD: 4.613  SCE: 3651.535  RMS: 3.470  LUFS: 1.885  
proxy0     MSD: 5.099  SCE: 1862.924  RMS: 4.515  LUFS: 1.621  
Baseline   MSD: 5.135  SCE: 1852.895  RMS: 29.542  LUFS: 13.255  
Corrupt    MSD: 8.587  SCE: 5692.969  RMS: 13.567  LUFS: 4.993  

telephone-->warm
100%|███████████████████████████████████████████| 10/10 [00:35<00:00,  3.57s/it]
autodiff   MSD: 3.965  SCE: 947.003  RMS: 11.701  LUFS: 4.596  
spsa       MSD: 5.228  SCE: 1044.573  RMS: 15.173  LUFS: 5.606  
proxy0     MSD: 6.266  SCE: 2187.456  RMS: 45.108  LUFS: 17.355  
Baseline   MSD: 9.049  SCE: 1669.726  RMS: 6.980  LUFS: 3.745  
Corrupt    MSD: 11.205  SCE: 1524.191  RMS: 5.889  LUFS: 4.271  

neutral-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:38<00:00,  3.83s/it]
autodiff   MSD: 5.919  SCE: 930.650  RMS: 1.039  LUFS: 0.554  
spsa       MSD: 6.176  SCE: 1030.785  RMS: 3.445  LUFS: 1.300  
proxy0     MSD: 7.642  SCE: 955.156  RMS: 6.284  LUFS: 3.115  
Baseline   MSD: 6.963  SCE: 584.665  RMS: 6.691  LUFS: 2.750  
Corrupt    MSD: 7.364  SCE: 1393.375  RMS: 5.646  LUFS: 2.405  

neutral-->telephone
100%|███████████████████████████████████████████| 10/10 [01:15<00:00,  7.55s/it]
autodiff   MSD: 5.892  SCE: 174.921  RMS: 1.639  LUFS: 0.751  
spsa       MSD: 6.844  SCE: 205.956  RMS: 6.119  LUFS: 2.741  
proxy0     MSD: 7.938  SCE: 220.270  RMS: 4.373  LUFS: 1.958  
Baseline   MSD: 7.668  SCE: 243.061  RMS: 7.975  LUFS: 3.246  
Corrupt    MSD: 10.047  SCE: 922.323  RMS: 5.995  LUFS: 3.302  

neutral-->neutral
100%|███████████████████████████████████████████| 10/10 [00:42<00:00,  4.24s/it]
autodiff   MSD: 4.376  SCE: 479.564  RMS: 1.821  LUFS: 0.728  
spsa       MSD: 5.174  SCE: 700.180  RMS: 2.980  LUFS: 1.256  
proxy0     MSD: 6.352  SCE: 410.978  RMS: 6.181  LUFS: 2.945  
Baseline   MSD: 5.412  SCE: 588.237  RMS: 4.344  LUFS: 2.115  
Corrupt    MSD: 6.826  SCE: 944.537  RMS: 3.581  LUFS: 2.232  

neutral-->bright
100%|███████████████████████████████████████████| 10/10 [01:39<00:00,  9.97s/it]
autodiff   MSD: 3.813  SCE: 2155.787  RMS: 5.397  LUFS: 2.366  
spsa       MSD: 4.466  SCE: 2451.703  RMS: 3.662  LUFS: 1.580  
proxy0     MSD: 5.212  SCE: 1388.196  RMS: 4.916  LUFS: 2.789  
Baseline   MSD: 4.552  SCE: 856.215  RMS: 13.886  LUFS: 6.292  
Corrupt    MSD: 8.861  SCE: 5333.085  RMS: 13.093  LUFS: 4.045  

neutral-->warm
100%|███████████████████████████████████████████| 10/10 [00:26<00:00,  2.64s/it]
autodiff   MSD: 3.852  SCE: 1447.159  RMS: 3.055  LUFS: 1.282  
spsa       MSD: 4.310  SCE: 1533.351  RMS: 3.364  LUFS: 1.599  
proxy0     MSD: 4.953  SCE: 1366.485  RMS: 8.061  LUFS: 3.729  
Baseline   MSD: 6.034  SCE: 851.415  RMS: 5.795  LUFS: 2.972  
Corrupt    MSD: 7.863  SCE: 1160.896  RMS: 5.189  LUFS: 2.998  

bright-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:48<00:00,  4.82s/it]
autodiff   MSD: 5.281  SCE: 1287.706  RMS: 2.544  LUFS: 0.997  
spsa       MSD: 6.023  SCE: 2018.120  RMS: 3.074  LUFS: 1.097  
proxy0     MSD: 7.068  SCE: 2807.467  RMS: 6.503  LUFS: 3.095  
Baseline   MSD: 7.483  SCE: 465.684  RMS: 6.261  LUFS: 2.503  
Corrupt    MSD: 7.669  SCE: 4878.829  RMS: 12.448  LUFS: 4.104  

bright-->telephone
100%|███████████████████████████████████████████| 10/10 [01:31<00:00,  9.16s/it]
autodiff   MSD: 5.727  SCE: 294.151  RMS: 2.993  LUFS: 1.275  
spsa       MSD: 6.517  SCE: 822.264  RMS: 5.998  LUFS: 2.810  
proxy0     MSD: 8.026  SCE: 314.646  RMS: 6.618  LUFS: 2.634  
Baseline   MSD: 6.978  SCE: 297.031  RMS: 5.284  LUFS: 2.069  
Corrupt    MSD: 9.277  SCE: 5712.550  RMS: 12.289  LUFS: 4.404  

bright-->neutral
100%|███████████████████████████████████████████| 10/10 [01:02<00:00,  6.28s/it]
autodiff   MSD: 5.694  SCE: 1402.436  RMS: 2.791  LUFS: 1.241  
spsa       MSD: 6.566  SCE: 2049.418  RMS: 2.237  LUFS: 0.772  
proxy0     MSD: 7.959  SCE: 2064.933  RMS: 6.404  LUFS: 3.087  
Baseline   MSD: 7.848  SCE: 706.934  RMS: 1.997  LUFS: 0.614  
Corrupt    MSD: 8.448  SCE: 4981.422  RMS: 15.911  LUFS: 4.215  

bright-->bright
100%|███████████████████████████████████████████| 10/10 [01:30<00:00,  9.01s/it]
autodiff   MSD: 3.347  SCE: 1051.300  RMS: 2.393  LUFS: 1.379  
spsa       MSD: 3.728  SCE: 909.130  RMS: 3.405  LUFS: 1.348  
proxy0     MSD: 4.121  SCE: 836.498  RMS: 6.039  LUFS: 2.327  
Baseline   MSD: 4.790  SCE: 980.283  RMS: 11.279  LUFS: 4.587  
Corrupt    MSD: 5.055  SCE: 2172.028  RMS: 12.907  LUFS: 4.884  

bright-->warm
100%|███████████████████████████████████████████| 10/10 [00:36<00:00,  3.65s/it]
autodiff   MSD: 3.569  SCE: 1351.627  RMS: 5.278  LUFS: 2.889  
spsa       MSD: 4.743  SCE: 2563.007  RMS: 13.225  LUFS: 5.640  
proxy0     MSD: 5.911  SCE: 5251.111  RMS: 27.244  LUFS: 10.150  
Baseline   MSD: 9.030  SCE: 846.539  RMS: 4.628  LUFS: 2.327  
Corrupt    MSD: 8.576  SCE: 5245.425  RMS: 18.363  LUFS: 3.916  

warm-->broadcast
100%|███████████████████████████████████████████| 10/10 [00:58<00:00,  5.90s/it]
autodiff   MSD: 5.288  SCE: 510.650  RMS: 1.278  LUFS: 0.772  
spsa       MSD: 5.692  SCE: 1178.561  RMS: 2.900  LUFS: 1.194  
proxy0     MSD: 9.389  SCE: 988.039  RMS: 7.070  LUFS: 4.741  
Baseline   MSD: 6.917  SCE: 1349.983  RMS: 11.752  LUFS: 4.370  
Corrupt    MSD: 7.793  SCE: 1454.567  RMS: 8.747  LUFS: 3.901  

warm-->telephone
100%|███████████████████████████████████████████| 10/10 [01:18<00:00,  7.82s/it]
autodiff   MSD: 4.924  SCE: 127.741  RMS: 3.423  LUFS: 1.673  
spsa       MSD: 6.156  SCE: 592.691  RMS: 8.652  LUFS: 4.304  
proxy0     MSD: 5.924  SCE: 564.229  RMS: 12.786  LUFS: 5.267  
Baseline   MSD: 9.343  SCE: 813.276  RMS: 10.653  LUFS: 3.503  
Corrupt    MSD: 11.138  SCE: 983.834  RMS: 5.029  LUFS: 2.949  

warm-->neutral
100%|███████████████████████████████████████████| 10/10 [01:10<00:00,  7.07s/it]
autodiff   MSD: 5.059  SCE: 474.697  RMS: 1.627  LUFS: 0.606  
spsa       MSD: 5.559  SCE: 979.341  RMS: 2.158  LUFS: 0.808  
proxy0     MSD: 8.685  SCE: 860.294  RMS: 4.988  LUFS: 3.829  
Baseline   MSD: 7.061  SCE: 1128.742  RMS: 13.217  LUFS: 5.004  
Corrupt    MSD: 7.815  SCE: 1242.012  RMS: 5.997  LUFS: 2.878  

warm-->bright
100%|███████████████████████████████████████████| 10/10 [01:16<00:00,  7.67s/it]
autodiff   MSD: 3.362  SCE: 1751.219  RMS: 5.462  LUFS: 2.800  
spsa       MSD: 4.090  SCE: 2119.844  RMS: 5.119  LUFS: 2.240  
proxy0     MSD: 5.014  SCE: 1057.485  RMS: 4.544  LUFS: 1.985  
Baseline   MSD: 5.832  SCE: 2804.064  RMS: 19.218  LUFS: 7.381  
Corrupt    MSD: 8.620  SCE: 6288.224  RMS: 17.063  LUFS: 4.609  

warm-->warm
100%|███████████████████████████████████████████| 10/10 [00:40<00:00,  4.07s/it]
autodiff   MSD: 3.549  SCE: 646.522  RMS: 4.088  LUFS: 2.016  
spsa       MSD: 3.863  SCE: 699.310  RMS: 4.146  LUFS: 1.984  
proxy0     MSD: 4.896  SCE: 857.195  RMS: 7.609  LUFS: 3.748  
Baseline   MSD: 4.317  SCE: 323.523  RMS: 6.576  LUFS: 3.416  
Corrupt    MSD: 4.534  SCE: 1513.056  RMS: 4.237  LUFS: 2.441  

----- Averages ---- MUSDB18
autodiff   MSD: 4.741  SCE: 909.368  RMS: 3.456  LUFS: 1.629  
spsa       MSD: 5.445  SCE: 1220.946  RMS: 4.650  LUFS: 2.040  
proxy0     MSD: 6.693  SCE: 1393.777  RMS: 9.300  LUFS: 4.156  
Baseline   MSD: 6.813  SCE: 870.955  RMS: 10.099  LUFS: 4.316  
Corrupt    MSD: 8.420  SCE: 2607.409  RMS: 9.356  LUFS: 3.606