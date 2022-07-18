# Machine
sandle
Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz (16 core)
GeForce GTX 1080 Ti

# 100

dsp_infer           : sec/step 0.0177    0.0035 RTF
autodiff_cpu_infer  : sec/step 0.0256    0.0051 RTF
autodiff_gpu_infer  : sec/step 0.0047    0.0009 RTF
tcn1_cpu_infer      : sec/step 0.7828    0.1566 RTF
tcn2_cpu_infer      : sec/step 1.3870    0.2774 RTF
tcn1_gpu_infer      : sec/step 0.0116    0.0023 RTF
tcn2_gpu_infer      : sec/step 0.0222    0.0044 RTF
autodiff_gpu_grad   : sec/step 0.3009    0.0602 RTF
np_norm_gpu_grad    : sec/step 0.3880    0.0776 RTF
np_hh_gpu_grad      : sec/step 0.4226    0.0845 RTF
np_fh_gpu_grad      : sec/step 0.4319    0.0864 RTF
tcn1_gpu_grad       : sec/step 0.4323    0.0865 RTF
tcn2_gpu_grad       : sec/step 0.6371    0.1274 RTF
spsa_gpu_grad       : sec/step 0.3945    0.0789 RTF

# 1000

rb_infer : sec/step 0.0186    0.0037 RTF
dsp_infer : sec/step 0.0172    0.0034 RTF
autodiff_cpu_infer : sec/step 0.0295    0.0059 RTF
autodiff_gpu_infer : sec/step 0.0049    0.0010 RTF
tcn1_cpu_infer : sec/step 0.6580    0.1316 RTF
tcn2_cpu_infer : sec/step 1.3409    0.2682 RTF
tcn1_gpu_infer : sec/step 0.0114    0.0023 RTF
tcn2_gpu_infer : sec/step 0.0223    0.0045 RTF
autodiff_gpu_grad : sec/step 0.3086    0.0617 RTF
np_norm_gpu_grad : sec/step 0.4346    0.0869 RTF
np_hh_gpu_grad : sec/step 0.4379    0.0876 RTF
np_fh_gpu_grad : sec/step 0.4339    0.0868 RTF
tcn1_gpu_grad : sec/step 0.4382    0.0876 RTF
tcn2_gpu_grad : sec/step 0.6424    0.1285 RTF
spsa_gpu_grad : sec/step 0.4132    0.0826 RTF