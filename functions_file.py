from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
from PIL import Image
import torch.backends.cudnn as cudnn
import numpy as np

# import torchvision
# import torch.optim as optim
# from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd

cudnn.benchmark = True
plt.ion()
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd.gradcheck
import copy


from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


import matplotlib.pyplot as plt


#### needed depending on the model output you may need it

transf_img = transforms.Compose([transforms.Resize((256, 256))])


def create_image(pixels):
    img = Image.fromarray(pixels)
    img = img.convert("RGB")
    img = transf_img(img)
    return img


def get_preprocess_transform():

    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transf


to_tensor_img = get_preprocess_transform()


def predict_tensor(model, input_tensor):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_tensor)
        probs_adv = F.softmax(output, dim=1)
        probs_adv = probs_adv.detach().cpu().numpy()
        return probs_adv


trans = transforms.Compose([transforms.ToTensor()])


def pipeline_prediction(array_image_to_predict, model):
    image_tensor = to_tensor_img(array_image_to_predict)
    input_batch = image_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model WHEN IS ONLY ONE
    probs = predict_tensor(model, input_batch)
    confidence = np.max(probs)
    label_img = np.argmax(probs)
    sample = image_tensor, label_img
    return sample, confidence


#### FUNCTIONS FOR STRATEGY

explainer = lime_image.LimeImageExplainer(verbose=False)
segmenter = SegmentationAlgorithm("quickshift", compactness=1, sigma=1)


def fgsm_momentum_attack(image, alpha, gt, data_grad, dec):
    """This function computes the perturbation and velocity vector to produce the adversarial, it is used by the function ensemble_adv"""
    # Collect the element-wise sign of the data gradient
    gt = gt * dec + data_grad / torch.mean(abs(data_grad), [1, 2, 3], keepdim=True)
    sign_data_grad = gt.sign()
    # Create the perturbed image by adjusting each pixel of the input image

    perturbed_image = image + alpha * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image, gt


def ensamble_adv(sample_raw, model, device, alpha, dec, gt):
    """This function is a one step procedure (does not work on batch data but individual samples), calculating for the current iteration the gradient and using fgsm_momentum_attack to calculate the adversarial

    sample_raw: tuple formed by the subset image as a tensor and its data label (subset class)
    model: neural network to execute
    alpha: step size of perturbation, set to 0.001 usually
    dec: decay factor for momentum set to 0.006
    gt: velocity vector calculated in previous iterations, initially this is 0
    """
    dat = {"correct": DataLoader([sample_raw], batch_size=1, shuffle=False)}

    for i, (input, label) in enumerate(dat["correct"]):
        input = input.to(device)
        label = label.to(device)
        input.requires_grad = True
        output = model(input)
        probs = F.softmax(output, dim=1)
        probs = probs.detach().cpu().numpy()
        init_pred = np.argmax(probs)
        loss = F.nll_loss(output, label)
        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = input.grad.data
        perturbed_data, gt = fgsm_momentum_attack(
            input, alpha=alpha, gt=gt, data_grad=data_grad, dec=dec
        )

        # Re-classify the perturbed image
        outputs_adv = model(perturbed_data)
        # transform image to array so it can be used on next iteration
        perturbed_data = perturbed_data.squeeze(0).detach().cpu().numpy()
        adv_arr = np.swapaxes(perturbed_data, 0, 1)
        adv_arr = np.swapaxes(adv_arr, 1, 2)
        adv_tens = trans(adv_arr)
    return adv_tens, gt, init_pred, alpha


def adversarial_generation_reverse(
    original_img, label, temp, model, device, iter, gt_new=0, dec=1, alpha=0.001
):
    """This function (Algorithm 1 in the paper) iterates through the alpha values and computes the perturbation for the image
    checking if the label has been change and terminate the algorithm or continuing since all perturbation values have been used
    original_img: full image as numpy array
    label: original prediction of the full image
    temp: explanation provided by lime as an numpy array, i.e, the subset image containing relevant features
    model: model to use
    iter: number of iterations
    gt_new: initial velocit vector
    dec: decay factor
    alpha: initial perturbation size
    """
    first_perturbed = to_tensor_img(temp)
    input_batch_first = first_perturbed.unsqueeze(0)
    label_first_img = np.argmax(predict_tensor(model, input_batch_first))
    sample_raw = first_perturbed, label_first_img
    advs_gen = []
    preds_change = []
    # iterating over alpha to increase perturbance
    range_alpha = np.linspace(alpha, alpha * iter, iter)
    for i in range(iter):
        # print(range_alpha[i])
        adv_tensor, gt, pred_before, alpha = ensamble_adv(
            sample_raw, model, device, gt=gt_new, dec=dec, alpha=range_alpha[i]
        )
        gt_new = gt
        preds_change.append(pred_before)
        sample_raw = adv_tensor, preds_change[0]
        missing_feat_img = original_img - temp
        missing_feat_tensor = to_tensor_img(missing_feat_img)
        missing_feat_tensor = torch.clamp(missing_feat_tensor, 0, 1)
        complete_adv_tensor = adv_tensor + missing_feat_tensor
        input_batch_complete = complete_adv_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model WHEN IS ONLY ONE
        output_adv_tens = predict_tensor(model, input_batch_complete)
        label_adv_tens = np.argmax(output_adv_tens)
        # print(label_adv_tens)
        sample = complete_adv_tensor, label_adv_tens, output_adv_tens, alpha
        advs_gen.append(sample)
        if label != label_adv_tens:
            break

    return advs_gen[-1]


def generation_adversarial_lime_new(
    original_img,
    model,
    device,
    iter,
    alpha_original,
    proba_func,
    num_samples,
    dec,
    real_label,
    segmenter,
    explainer,
):

    """This function is the equivalent of algorithm 2 (in the paper) it increases the number of features in the explanation, i.e increasing the regions of the subset image to use to compute the perturbation. If all features available in the explanation are used, it proceeds to calculate the perturbation with the unimportant features.
    original_img: full image represented as a numpy array
    model: classifier to use
    iter: number of iterations for Algorithm 1
    alpha_original: initial perturbation size
    proba_func: probability function of the classifier for LIME
    num_samples: number of samples in the neighborhood of X to be used by LIME
    dec: decay factor for velocity vector
    real_label: original label of clean image
    segmenter and explainer are parameters for LIME


    """
    # initial values
    img = original_img.copy()
    label = real_label
    nb_feat = 10
    zer = np.zeros((256, 256, 3))
    explanation = explainer.explain_instance(
        np.array(img),
        proba_func,
        top_labels=7,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmenter,
    )
    max_distance = np.sqrt(np.sum((np.array(img) / 255 - zer) ** 2))
    distance_perturbed_img = max_distance
    label_adv_tens = label
    count = 0
    temp_imgs = [zer]
    while distance_perturbed_img > max_distance * 0.01:
        # print("is greater")
        if label == label_adv_tens:
            # print('is same')
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=nb_feat,
                hide_rest=True,
            )
            temp = temp.astype(np.uint8)
            distance_perturbed_img = np.sqrt(
                np.sum((np.array(img) / 255 - temp / 255) ** 2)
            )
            nb_feat = int(nb_feat + 0.1 * num_samples)
            # plt.imshow(temp)
            if (temp_imgs[-1] == temp).all():

                ##if no adversarial has been found on the last iteration, try with remaining features (not important ones)
                temp_reverse = img - temp
                k = adversarial_generation_reverse(
                    img,
                    label,
                    temp_reverse,
                    model=model,
                    device=device,
                    iter=iter,
                    dec=dec,
                    alpha=alpha_original,
                )
                complete_adv_tensor, label_adv_tens, probs, alpha = k
                confidence_adversarial = np.max(probs)
                distance_perturbed_img = np.sqrt(
                    np.sum((np.array(img) / 255 - temp_reverse / 255) ** 2)
                )
                similarity_imgs = 100 - round(
                    (distance_perturbed_img / max_distance) * 100, 4
                )
                gi = 2
                break
            else:
                temp_imgs.append(temp)
                k = adversarial_generation_reverse(
                    img,
                    label,
                    temp,
                    model=model,
                    device=device,
                    iter=iter,
                    dec=dec,
                    alpha=alpha_original,
                )
                complete_adv_tensor, label_adv_tens, probs, alpha = k
                similarity_imgs = 100 - round(
                    (distance_perturbed_img / max_distance) * 100, 4
                )
                count = count + 1
                confidence_adversarial = np.max(probs)
                gi = 1
                # print(count)
                # print(similarity_imgs)
        else:
            # print("label is different")
            break

    # print("number of features used are: ", nb_feat)
    # print("percentage of image chosen is: ", round((distance_perturbed_img/max_distance)*100,4))
    # print("alpha used is: ", round(alpha,2))
    return (
        label_adv_tens,
        confidence_adversarial,
        alpha,
        count,
        similarity_imgs,
        gi,
        complete_adv_tensor,
        temp,
    )


### FGSM


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test_fgsm(model, device, test_loader, epsilon, phase="correct"):
    """test_loader: tuple (x,y) of clean instance. Can be passed as a batch.
    epsilon: perturbation size
    """
    adversarial_gen = []

    # Loop over all examples in test set
    # for data, target in test_loader:

    #     # Send the data and label to the device
    #     data, target = data.to(device), target.to(device)

    #     # Set requires_grad attribute of tensor. Important for Attack
    #     data.requires_grad = True

    for i, (inputs, labels) in enumerate(test_loader[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)

        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # print(i)
        # print(probs)
        # print(init_pred)
        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #    continue
        # print(outputs.requires_grad)
        # Calculate the loss
        loss = F.nll_loss(outputs, labels)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = inputs.grad.data
        # print(data_grad.sign())
        # Call FGSM Attack
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

        # Re-classify the perturbed image
        outputs_adv = model(perturbed_data)
        # perturbed_pred = outputs_adv.max(1, keepdim=True)[1]
        probs_adv = F.softmax(outputs_adv, dim=1)
        confidence = np.max(probs_adv.detach().cpu().numpy())
        label_adv = np.argmax(probs_adv.detach().cpu().numpy())
        # print(label_adv)
        result = (perturbed_data, confidence, label_adv)
        adversarial_gen.append(result)

    return adversarial_gen


### BIM


def bim(image, epsilon, alpha, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    a = torch.clamp(image - epsilon, min=0)
    # b = max{a, X'}
    b = (perturbed_image >= a).float() * perturbed_image + (
        a > perturbed_image
    ).float() * a
    # c = min{X+eps, b}
    c = (b > image + epsilon).float() * (image + epsilon) + (
        image + epsilon >= b
    ).float() * b
    perturbed_image = torch.clamp(c, max=1)
    # Return the perturbed image
    return perturbed_image


def fgsm_iterative(model, device, test_loader, epsilon, alpha, iters, phase="correct"):

    adversarial_gen = []

    for i, (inputs, labels) in enumerate(test_loader[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        for i in range(iters):
            inputs.requires_grad = True
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            loss = F.nll_loss(outputs, labels)

            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect datagrad
            data_grad = inputs.grad.data
            # print(data_grad.sign())
            # Call FGSM Attack
            perturbed_data = bim(inputs, epsilon, alpha, data_grad)
            inputs = perturbed_data.detach_()

        probs_adv = F.softmax(outputs, dim=1)
        confidence = np.max(probs_adv.detach().cpu().numpy())
        label_adv = np.argmax(probs_adv.detach().cpu().numpy())
        result = (perturbed_data, confidence, label_adv)
        adversarial_gen.append(result)

    return adversarial_gen


###https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
##DEEPFOOL
def deepfool(image, net, num_classes=7, overshoot=0.02, max_iter=50):

    """
    :param image: Image of size HxWx3
    :param net: network (input: images, output: values of activation **BEFORE** softmax).
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter: maximum number of iterations for deepfool (default = 50)
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image = (
        net.forward(Variable(image[None, :, :, :], requires_grad=True))
        .data.cpu()
        .numpy()
        .flatten()
    )
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            if x.grad is not None:
                x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        probs_adv = F.softmax(net(x), dim=1)
        confidence_i = np.max(probs_adv.detach().cpu().numpy())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, confidence_i, torch.clamp(pert_image, 0, 1)


def fgsm_momentum_attack_minimize(image, alpha, gt, data_grad, dec):
    # Collect the element-wise sign of the data gradient
    gt = gt * dec + data_grad / torch.mean(abs(data_grad), [1, 2, 3], keepdim=True)
    sign_data_grad = gt.sign()
    # Create the perturbed image by adjusting each pixel of the input image

    perturbed_image = image - alpha * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image, gt
