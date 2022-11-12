import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json


class DrawCradCams:

    def __init__(self, path, dif):
        self.path = path
        # print(path)
        # 存放梯度和特征图
        self.dif = dif
        self.grad_block = list()
        self.fmap_block = list()

    # 图片预处理
    def img_preprocess(self, img_in):
        img = img_in.copy()
        img = img[:, :, ::-1]   				# 1
        img = np.ascontiguousarray(img)			# 2
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = img.unsqueeze(0)					# 3
        return img

    # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    # 计算grad-cam并可视化
    def cam_show_img(self, img, feature_map, grads, out_dir):
        H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
        grads = grads.reshape([grads.shape[0], -1])					# 5
        weights = np.mean(grads, axis=1)							# 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							# 7
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        path_cam_img = os.path.join(out_dir + "/2CData/pic_{}_cam.jpg".format(self.dif))
        cv2.imwrite(path_cam_img, cam_img)
        return path_cam_img

    def DRAW(self):
        # json_path = './labels.json'
        output_dir = '.'


        # 图片读取；网络加载
        img = cv2.imread(self.path, 1)
        # cv2.imshow(' ', img)
        img_input = self.img_preprocess(img)


        net = models.resnet50(pretrained=True)
        pthfile = './resnet50-pre.pth'
        net.load_state_dict(torch.load(pthfile, map_location='cpu'), strict=False)


        net.eval()  # 8
        # print(net)

        # resnet 50
        net.layer4[-1].register_forward_hook(self.farward_hook)
        net.layer4[-1].register_full_backward_hook(self.backward_hook)

        # forward
        output = net(img_input)
        idx = np.argmax(output.cpu().data.numpy())
        # print(idx)
        # print("predict: {}".format(classes[idx]))

        # backward
        net.zero_grad()
        class_loss = output[0, idx]
        class_loss.backward()

        # 生成cam
        grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
        fmap = self.fmap_block[0].cpu().data.numpy().squeeze()

        # 保存cam图片
        save_path = self.cam_show_img(img, fmap, grads_val, output_dir)

        return save_path
