from functions_file import *
import os
import matplotlib.pyplot as plt
from sewar.full_ref import uqi, rmse, psnrb, scc, sam, vifp
from skimage.metrics import structural_similarity as ssim


"""
This file is run after success_confidence_deepf_newstg and success_confidence_fgsm_bim have been run, it produces a csv with all distance measures from each strategy and its adversaries. Ignore the column rmse_subset_method 
"""

model = input("Model to use with .pt extension: ")
name_model = os.path.splitext(model)[0]
number_images = int(
    input("Number of images to test, write the same amount as for the success test: ")
)
nb_neigh_new = int(input("Neighbors for LIME: "))
value_dec = float(input("Decay factor for new strategy: "))
# batch_size = int(input("batch size: "))
data_set = input("Data set to use, can be FER or JAFFE: ")
# loading model previously trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = torch.load(model, map_location=device)
model_ft.eval()

## this test runs after we got the correct predictions from success_confidence_deepf_newstg
data_table = pd.read_csv(
    "correct_predictions" + "_" + name_model + "_" + str(number_images) + ".csv"
).iloc[:number_images, :]


## prepare data
if data_set == "FER":
    all_data = pd.read_csv("FER2013_data_set.csv", delimiter=";")
    array_data = [
        np.fromstring(all_data.iloc[i, 1], dtype=np.uint8, sep=" ")
        for i in range(number_images)
    ]
    image_first_reshape = [
        np.reshape(array_data[i], (48, 48)) for i in range(len(array_data))
    ]
    array_imgs = [create_image(i) for i in image_first_reshape]
    labels = all_data.iloc[:number_images, 0].values.tolist()
elif data_set == "JAFFE":
    files_jaffe = [
        f
        for f in os.listdir("jaffedbase")
        if os.path.isfile(os.path.join("jaffedbase", f))
    ]
    files_jaffe.remove(".DS_Store")
    emotion_raw = [files_jaffe[i].split(".")[1][:2] for i in range(number_images)]
    dict_emotions = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}
    emotions = [*map(dict_emotions.get, emotion_raw)]
    all_data = pd.DataFrame(emotions)
    labels = all_data.iloc[:number_images, 0].values.tolist()
    array_imgs = [
        create_image(plt.imread("jaffedbase/" + files_jaffe[i]))
        for i in range(number_images)
    ]
else:
    print("please choose JAFFE or FER:")


## steps for LIME
def prob_function_for_lime(array_images, model=model_ft):
    x = torch.stack(tuple(to_tensor_img(i) for i in array_images), dim=0)
    output = predict_tensor(model, x)
    return output


seed = 1234
explainer = lime_image.LimeImageExplainer(
    verbose=False, feature_selection="lasso_path", random_state=seed
)
segmenter = SegmentationAlgorithm(
    "quickshift", compactness=1, sigma=1, random_seed=seed
)


indexes = data_table["original_index"].values.tolist()
indexes = [i for i in indexes if i < number_images]

## my way of transforming back to an array, maybe not the best
def tensor_to_img(tensor_fd):
    back_array = np.swapaxes(tensor_fd.squeeze(0).detach().cpu().numpy(), 0, 2)
    back_array = np.swapaxes(back_array, 1, 0)
    return back_array


def distances_calc(explanation, adversarial, full_image):
    subset_adv = np.multiply(explanation, adversarial)
    subset_original_tensor = np.multiply(explanation, full_image)
    uqi_measure = uqi(full_image, adversarial)
    rmse_measure = rmse(full_image, adversarial)
    vifp_measure = vifp(full_image, adversarial)
    ssim_measure = ssim(adversarial, full_image, channel_axis=2)
    rmse_subset = rmse(subset_adv, subset_original_tensor)
    return (uqi_measure, rmse_measure, vifp_measure, ssim_measure, rmse_subset)


data = {}
list_methods = ["new_strat", "deepfool", "fgsm", "BIM"]
for i in indexes:
    ##first calculate the things for the explanation so this can be used for comparing the subset image with the subset region perturbed by each strategy (this was an additional experiment but wwe will not include it)
    advs_vals_white_0_001_new = generation_adversarial_lime_new(
        array_imgs[i],
        model_ft,
        device=device,
        iter=10,
        alpha_original=0.001,
        dec=value_dec,
        proba_func=prob_function_for_lime,
        num_samples=nb_neigh_new,
        real_label=labels[i],
        explainer=explainer,
        segmenter=segmenter,
    )
    explanation_final = advs_vals_white_0_001_new[7] / 255
    adversarial_output = tensor_to_img(advs_vals_white_0_001_new[6])
    original_image = tensor_to_img(torch.clamp(to_tensor_img(array_imgs[i]), 0, 1))
    explanation_final[explanation_final > 0] = 1
    all_measures = distances_calc(explanation_final, adversarial_output, original_image)
    data.setdefault("index", []).append(i)
    data.setdefault("original_label", []).append(labels[i])
    data.setdefault("label_new", []).append(advs_vals_white_0_001_new[0])
    data.setdefault("confidence_new", []).append(advs_vals_white_0_001_new[1])
    for j in list_methods:
        if j == "new_strat":
            all_measures = distances_calc(
                explanation_final, adversarial_output, original_image
            )
        if j == "fgsm":
            sample = [(to_tensor_img(array_imgs[i]), labels[i])]
            dataloaders = {
                "find_adver": DataLoader(sample, batch_size=1, shuffle=False)
            }
            advers_tuple = test_fgsm(model_ft, device, dataloaders, 0.001, "find_adver")
            advers_fgsm = tensor_to_img(advers_tuple[0][0])
            all_measures = distances_calc(
                explanation_final, advers_fgsm, original_image
            )
        if j == "BIM":
            sample_bim = [(to_tensor_img(array_imgs[i]), labels[i])]
            dataloaders_bim = {
                "find_adver": DataLoader(sample_bim, batch_size=1, shuffle=False)
            }
            advers_tuple_bim = fgsm_iterative(
                model_ft, device, dataloaders_bim, 0.006, 0.001, 10, "find_adver"
            )
            advers_bim = tensor_to_img(advers_tuple_bim[0][0])
            all_measures = distances_calc(explanation_final, advers_bim, original_image)
        if j == "deepfool":
            a, b, c, d, e, f = deepfool(
                to_tensor_img(array_imgs[i]), model_ft, max_iter=12
            )
            adversarial_deepfool = tensor_to_img(f)
            all_measures = distances_calc(
                explanation_final, adversarial_deepfool, original_image
            )
        data.setdefault("uqi_" + j, []).append(all_measures[0])
        data.setdefault("rmse_" + j, []).append(all_measures[1])
        data.setdefault("vifp_" + j, []).append(all_measures[2])
        data.setdefault("ssim_" + j, []).append(all_measures[3])
        data.setdefault("rmse_subset_" + j, []).append(all_measures[4])

results = pd.DataFrame(data)

results.to_csv(
    "distances" + data_set + "_" + str(number_images) + "_" + name_model + ".csv",
    encoding="utf-8",
    index=False,
)
