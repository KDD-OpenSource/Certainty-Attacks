## quality
import pandas as pd
distances_jaffe_inception = pd.read_csv('distancesimagenet_50000_imagenet_inception.csv')
selected_metrics = distances_jaffe_inception[["rmse_new_strat", "vifp_new_strat", "ssim_new_strat", "rmse_deepfool", "vifp_deepfool", "ssim_deepfool", "rmse_fgsm", "vifp_fgsm", "ssim_fgsm", "rmse_BIM", "vifp_BIM", "ssim_BIM"]]
print(selected_metrics.mean())
