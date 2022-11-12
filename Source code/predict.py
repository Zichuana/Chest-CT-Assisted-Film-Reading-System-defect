import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet50


class Predict:
    def __init__(self, path):
        self.img_path = path

    def predict_3(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 标准化参数
        # transforms.Normalize([0.5, ], [0.5, ])])

        # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(self.img_path)
        # plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)  # 扩充维度（添加到前面）  banch, 高度, 宽度, 深度
        # plt.show()

        # 读取索引对应的类别名称
        json_path = './class3_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = resnet50(num_classes=3).to(device)
        # load model weights
        weights_path = "./NET50/ResNet50_64_3.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

        # prediction
        model.eval()
        with torch.no_grad():  # 禁止pytorch跟踪变量损失梯度
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)  # 概率分布
            predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处

        print_res = "该胸片最佳预测结果为：{}   概率值: {:.5}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
        # print(print_res)
        # print()
        res = []
        plt.title(print_res)
        for i in range(len(predict)):
            # print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
            #                                           predict[i].numpy()))
            res.append("class: {:10}   prob: {:.5}".format(class_indict[str(i)], predict[i].numpy()))
        return print_res, res

    def predict_2(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 标准化参数
        # transforms.Normalize([0.5, ], [0.5, ])])

        # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(self.img_path)
        # plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)  # 扩充维度（添加到前面）  banch, 高度, 宽度, 深度
        # plt.show()

        # 读取索引对应的类别名称
        json_path = './class2_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = resnet50(num_classes=2).to(device)
        # load model weights
        weights_path = "./NET50/ResNet50_32_2.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():  # 禁止pytorch跟踪变量损失梯度
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)  # 概率分布
            predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处

        print_res = "该胸片最佳预测结果为：{}   概率值: {:.5}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
        # print(print_res)
        # print()
        plt.title(print_res)
        res = []
        for i in range(len(predict)):
            res.append("class: {:10}   problem: {:.5}".format(class_indict[str(i)], predict[i].numpy()))
        return print_res, res

