from functions_file import *
import os
import matplotlib.pyplot as plt


"""
This file will generate three csv, the first one containing the original predictions for the correctly classified images with as well as the classification provided after perturbing with the new strategy and deepfool. The second file provides the output of the new strategy containing the label of the adversarial "label_adv_tens", the confidence_adversarial, the perturbation size used alpha, the number of iterations needed "count", the portion of the image used "similarity_imgs", if the image was used with important or unimportant features "gi", the tensor "complete_adv_tensor", and the original explanation "temp". The third document is the output from deepfool r_tot, loop_i, the label of the adversarial "label", k_i, the confidence of the adversarial "confidence_i" 
"""


model = input("model to load with .pt extension: ")
name_model = os.path.splitext(model)[0]
number_images = int(input("how many images to test: "))
nb_neigh_new = int(input("how many neighbors for LIME: "))
value_dec = float(input("how much decay factor for new strategy: "))
# batch_size = int(input("batch size: "))
data_set = input("Name of dataset, can be JAFFE or FER: ")
# loading model previously trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = torch.load(model, map_location=device)
model_ft.eval()


# preparing data given methodoly to extract it

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
elif data_set == "JAFFE":
    files_jaffe = [
        f
        for f in os.listdir("jaffedbase")
        if os.path.isfile(os.path.join("jaffedbase", f))
    ]
    files_jaffe.remove(".DS_Store")
    emotion_raw = [files_jaffe[i].split(".")[1][:2] for i in range(len(files_jaffe))]
    dict_emotions = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}
    ## list of original class for each image
    emotions = [*map(dict_emotions.get, emotion_raw)]
    all_data = pd.DataFrame(emotions)

    ## array of images to provide to model
    array_imgs = [
        create_image(plt.imread("jaffedbase/" + files_jaffe[i]))
        for i in range(len(files_jaffe))
    ]
else:
    print("please choose JAFFE or FER")


##producing prob function for LIME
## read LIME documentation, probability function needs to be able to handle multiple array of images and provide the provide as output the array of probabilities of each class
def prob_function_for_lime(array_images, model=model_ft):
    x = torch.stack(tuple(to_tensor_img(i) for i in array_images), dim=0)
    output = predict_tensor(model, x)
    return output


# producing original predictions

list_predictions = [pipeline_prediction(i, model_ft) for i in array_imgs]

table_predictions = pd.DataFrame(list_predictions)

table_predictions["real_label"] = all_data.iloc[:number_images, 0]

table_predictions["model_label"] = table_predictions[0].apply(pd.Series)[1]

## filtering only for images correctly classify, this is what will be passed to the algorithm
correct_predictions = table_predictions[
    table_predictions["real_label"] == table_predictions["model_label"]
]

index_correct = correct_predictions.index
index_correct = index_correct.values.tolist()

correct_predictions["original_index"] = index_correct
l_label = table_predictions["model_label"].values.tolist()


##new strategy

seed = 1348
explainer = lime_image.LimeImageExplainer(
    verbose=False, feature_selection="lasso_path", random_state=seed
)
segmenter = SegmentationAlgorithm(
    "quickshift", compactness=1, sigma=1, random_seed=seed
)

advs_vals_white_0_001_new = [
    generation_adversarial_lime_new(
        array_imgs[i],
        model_ft,
        device=device,
        iter=10,
        alpha_original=0.001,
        dec=value_dec,
        proba_func=prob_function_for_lime,
        num_samples=nb_neigh_new,
        real_label=l_label[i],
        explainer=explainer,
        segmenter=segmenter,
    )
    for i in index_correct
]

new_strat = pd.DataFrame(advs_vals_white_0_001_new)
correct_predictions["new_strategy_label"] = [
    advs_vals_white_0_001_new[i][0] for i in range(len(advs_vals_white_0_001_new))
]
correct_predictions["group_adv"] = [
    advs_vals_white_0_001_new[i][5] for i in range(len(advs_vals_white_0_001_new))
]

new_strat.to_csv(
    "Adversarials_new_strat_" + name_model + "_" + str(number_images) + ".csv",
    encoding="utf-8",
    index=False,
)

# Deepfool

deep_fool_advs = [
    deepfool(to_tensor_img(array_imgs[i]), model_ft, num_classes=7, max_iter=12)
    for i in index_correct
]
deep_fool_df = pd.DataFrame(deep_fool_advs)
correct_predictions["deep_fool_label"] = [
    deep_fool_advs[i][3] for i in range(len(deep_fool_advs))
]


# print(correct_predictions)
correct_predictions.to_csv(
    "correct_predictions_" + name_model + "_" + str(number_images) + ".csv",
    encoding="utf-8",
    index=False,
)
deep_fool_df.to_csv(
    "Adversarials_deepfool_" + name_model + "_" + str(number_images) + ".csv",
    encoding="utf-8",
    index=False,
)
