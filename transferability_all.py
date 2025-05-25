from functions_file import *
import os
import matplotlib.pyplot as plt

"""
This file produces one csv file containing the labels produced by each strategy using the adversarials produced by the selected model in two different target models. The from_model is the model you put as input to run this file, the remaining models are the ones used as targets. You can check the list of models_to_test and add more models if necessary.  

"""
model = input("Model to use with .pt extension: ")
name_model = os.path.splitext(model)[0]
number_images = int(
    input("Number of images (write the same as in the docs already produced): ")
)
# batch_size = int(input("batch size: "))
data_set = input("Data set to use: ")
nb_neigh_new = int(input("Neighbors for LIME: "))
value_dec = float(input("Decay factor value: "))
strategies_to_check = input(
    "Provide strategies split by comma as new_strategy, BIM, fgsm, deepfool: "
)
# loading model previously trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = torch.load(model, map_location=device)
model_ft.eval()
data_table = pd.read_csv(
    "correct_predictions" + "_" + name_model + "_" + str(number_images) + ".csv"
).iloc[:number_images, :]


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
    emotion_raw = [files_jaffe[i].split(".")[1][:2] for i in range(number_images)]
    dict_emotions = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}
    emotions = [*map(dict_emotions.get, emotion_raw)]
    all_data = pd.DataFrame(emotions)
    array_imgs = [
        create_image(plt.imread("jaffedbase/" + files_jaffe[i]))
        for i in range(number_images)
    ]
else:
    print("please choose JAFFE or FER:")

labels = all_data.iloc[:number_images, 0].values.tolist()


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
list_strategies = strategies_to_check.split(",")


if data_set == "FER":
    models_to_test = ["inceptionv3.pt", "VGG19.pt", "Resnet18_model_two.pt"]
else:
    models_to_test = [
        "inceptionv3_JAFFE_two.pt",
        "VGG19_JAFFE_two.pt",
        "Resnet18_model_JAFFE_two.pt",
    ]
models_to_test.remove(model)

for strategy in list_strategies:
    if strategy == "new_strategy":
        advs_to_test_new = [
            generation_adversarial_lime_new(
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
            )[6].unsqueeze(0)
            for i in indexes
        ]

    elif strategy == "BIM":
        gen_tensors_labels = [
            pipeline_prediction(array_imgs[i], model_ft)[0] for i in indexes
        ]
        dataloaders = {
            "find_adver": DataLoader(gen_tensors_labels, batch_size=1, shuffle=False)
        }
        list_adversarials_BIM = fgsm_iterative(
            model_ft, device, dataloaders, 0.006, 0.001, 10, "find_adver"
        )
        advs_to_test_new = [
            list_adversarials_BIM[i][0] for i in range(len(list_adversarials_BIM))
        ]

    elif strategy == "fgsm":
        gen_tensors_labels = [
            pipeline_prediction(array_imgs[i], model_ft)[0] for i in indexes
        ]
        dataloaders = {
            "find_adver": DataLoader(gen_tensors_labels, batch_size=1, shuffle=False)
        }
        list_adversarials_fgsm = test_fgsm(
            model_ft, device, dataloaders, 0.001, "find_adver"
        )
        advs_to_test_new = [
            list_adversarials_fgsm[i][0] for i in range(len(list_adversarials_fgsm))
        ]

    elif strategy == "deepfool":
        advs_to_test_new = [
            deepfool(
                to_tensor_img(array_imgs[i]), model_ft, num_classes=7, max_iter=12
            )[5]
            for i in indexes
        ]

    for y in models_to_test:
        model_c = torch.load(y, map_location=device)
        model_c.eval()
        data_table[strategy + name_model + "_to_" + y] = [
            np.argmax(predict_tensor(model_c, advs_to_test_new[i]))
            for i in range(len(advs_to_test_new))
        ]


data_table.to_csv(
    "transferability_"
    + "from_"
    + name_model
    + "_nb_strat"
    + str(len(list_strategies))
    + ".csv",
    encoding="utf-8",
    index=False,
)
