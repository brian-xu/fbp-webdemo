import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from cv2.data import haarcascades
from torch import nn

import Nets


class Predictor:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    classifier_xml = os.path.join(haarcascades, 'haarcascade_frontalface_default.xml')
    face_classifier = cv2.CascadeClassifier(classifier_xml)

    def __init__(self, gpu: bool):
        self.net = None
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')

    def use_alexnet(self) -> nn.Module:
        self.net = torch.load('./models/alexnet.pth', map_location=self.device)
        self.net.eval()
        return Nets.AlexNet

    def use_resnet(self) -> nn.Module:
        self.net = torch.load('./models/resnet18.pth', map_location=self.device)
        self.net.eval()
        return Nets.ResNet

    def predict(self, img: np.array) -> [(np.array, torch.Tensor)]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = Predictor.face_classifier.detectMultiScale(gray, 1.3, 4)

        with torch.no_grad():
            pred = []
            for (x, y, w, h) in faces:
                face_img = img[y:y + h, x:x + w]
                face = Predictor.transform(Image.fromarray(face_img))
                face = face.unsqueeze(0)
                output = self.net(face).squeeze(1)
                pred.append((face_img, output.cpu()[0].item()))

        return pred
