from functions_file import *
import os

"""
This file will generate two csv, the first one containing the information from BIM: "adversarial, label and confidence" and the second one of FGSM "adversarial, label and confidence"
"""


model = input("model to load with .pt extension: ")
name_model = os.path.splitext(model)[0]
number_images = int(input("how many images same as in success test: "))
batch_size = int(input("batch size: "))
data_set = input("Name of dataset, can be JAFFE or FER: ")
# loading model previously trained
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = torch.load(model, map_location=device)
model_ft.eval()


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
    emotions = [*map(dict_emotions.get, emotion_raw)]
    all_data = pd.DataFrame(emotions)
    array_imgs = [
        create_image(plt.imread("jaffedbase/" + files_jaffe[i]))
        for i in range(len(files_jaffe))
    ]
else:
    print("please choose JAFFE or FER")

list_predictions = [pipeline_prediction(i, model_ft) for i in array_imgs]

table_predictions = pd.DataFrame(list_predictions)

table_predictions["real_label"] = all_data.iloc[:number_images, 0]

table_predictions["model_label"] = table_predictions[0].apply(pd.Series)[1]

correct_predictions = table_predictions[
    table_predictions["real_label"] == table_predictions["model_label"]
]

index_correct = correct_predictions.index
index_correct = index_correct.values.tolist()

correct_predictions["original_index"] = index_correct
l_label = table_predictions["model_label"].values.tolist()


##BIM AND FGSM running in small batches (images in test_loaders)
batch_init = np.arange(0, number_images, batch_size)
batch_init = np.append(batch_init, number_images)


list_adversarials_BIM = []
for i in range(len(batch_init) - 1):
    preprocess_loader = correct_predictions.iloc[
        batch_init[i] : batch_init[i + 1], 0
    ].values.tolist()
    dataloaders = {
        "find_adver": DataLoader(preprocess_loader, batch_size=1, shuffle=False)
    }
    list_adversarials_BIM.append(
        fgsm_iterative(model_ft, device, dataloaders, 0.006, 0.001, 10, "find_adver")
    )

list_adversarials_BIM = [item for sublist in list_adversarials_BIM for item in sublist]
# correct_predictions['BIM_label'] = [list_adversarials_BIM[i][2] for i in range(len(list_adversarials_BIM))]
adversarials_BIM = pd.DataFrame(list_adversarials_BIM)

# print(adversarials_BIM)


adversarials_BIM.to_csv(
    "Adversarials_bim_" + name_model + "_" + str(number_images) + ".csv",
    encoding="utf-8",
    index=False,
)

# FGSM
list_adversarials = []
for i in range(len(batch_init) - 1):
    # print(list(range(i,i+400)))
    preprocess_loader = correct_predictions.iloc[
        batch_init[i] : batch_init[i + 1], 0
    ].values.tolist()
    dataloaders = {
        "find_adver": DataLoader(preprocess_loader, batch_size=1, shuffle=False)
    }
    list_adversarials.append(
        test_fgsm(model_ft, device, dataloaders, 0.001, "find_adver")
    )

list_adversarials = [item for sublist in list_adversarials for item in sublist]
adversarials_fgsm = pd.DataFrame(list_adversarials)
# correct_predictions['fgsm_label'] = [list_adversarials[i][2] for i in range(len(list_adversarials))]

adversarials_fgsm.to_csv(
    "Adversarials_fgsm_" + name_model + "_" + str(number_images) + ".csv",
    encoding="utf-8",
    index=False,
)

# print(adversarials_fgsm)
