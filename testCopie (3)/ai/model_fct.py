from io import BytesIO

import cv2
from PIL import Image
from ultralytics import YOLO

import mysql.connector
from veid.load_model import load_model_from_opts
import torch
from torchvision import transforms
import numpy as np
import requests



class AIParkingModule:

    device = "cuda"
    plate_detector = None
    ocr_model = None
    model_veid = None
    data_transforms = None
    FLASK_URL = 'http://127.0.0.1:5000/api/'

    def __init__(self):
        print("Initializing models")
        self.plate_detector = YOLO("../models/yolov11/results/40_epochs_yolox/weights/last.pt")
        self.ocr_model = YOLO("../models/yolov11/results/50_epochs_full_yolol/weights/best.pt")
        self.model_veid = load_model_from_opts("veid/model/result/opts.yaml", ckpt="veid/model/result/net_29.pth",
                                               remove_classifier=True)
        self.model_veid.eval()
        self.model_veid.to(self.device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Initialized !")

    def extract_feature(self, model, X):
        """Exract the embeddings of a single image tensor X"""
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, 0)
        X = X.to(self.device)
        feature = model(X).reshape(-1)

        X = self.fliplr(X)
        flipped_feature = model(X).reshape(-1)
        feature += flipped_feature

        fnorm = torch.norm(feature, p=2)
        return feature.div(fnorm)

    def fliplr(self, img):
        """flip images horizontally in a batch"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        inv_idx = inv_idx.to(img.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def get_scores(self, query_feature, gallery_features):
        """Calculate the similarity scores of the query and gallery features"""
        query = query_feature.view(-1, 1)
        score = torch.mm(gallery_features, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        return score

    def check_img(self, image):
        image = cv2.resize(image, (2048, 2048))
        return self.plate_detector.predict(image, conf=0.5)

    def start(self):
        while True:
            img = self.request_frame()
            image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (2048, 2048))
                results_1 = self.plate_detector(image, conf=0.5)
                roi_image = None

                if results_1 is None:
                    self.change_state("Aucune voiture détectée", "En attente...", "pending")
                else:
                    print("car detected")
                    self.change_state("Voiture détectée", "Lecture de la plaque...", "detected")

                    for r in results_1:
                        for box in r.boxes:
                            if self.plate_detector.names[int(box.cls)] == 'license-plate':
                                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                                roi_image = image[y1:y2, x1:x2]
                                break

                    if roi_image is None:
                        self.change_state("Voiture détectée", "Pas de plaque détectée", "detected")
                    else:
                        roi_image = cv2.resize(roi_image, (2048, 2048))
                        self.change_state("Voiture détectée", "Récupération des chiffres de la plaque...", "detected")
                        results_2 = self.ocr_model.predict(roi_image, conf=0.38)
                        class_names = "".join(
                            self.ocr_model.names[int(box.cls)]
                            for r in results_2 for box in sorted(r.boxes, key=lambda x: x.xyxy[0].mean().item())
                        )
                        self.change_state("Voiture détectée", "Plaque n°"+class_names, "detected")

                        if not class_names:
                            self.change_state("Plaque illisible", "En attente...", "pending")

                        result = self.check_license_from_server(class_names)

                        gallery_images = []
                        if result:
                            self.change_state("Plaque valide", "Plaque n°" + class_names, "success")

                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_image)
                            X_query = torch.unsqueeze(self.data_transforms(pil_image), 0).to(self.device)

                            print(self.get_image_list())
                            for filename in self.get_image_list():
                                if filename.endswith(".jpg") or filename.endswith(".png"):
                                    gallery_image = self.get_image_from_server(filename.replace("/api/images/", ""))
                                    gallery_images.append(gallery_image)

                            X_gallery = torch.stack([self.data_transforms(img) for img in gallery_images]).to(self.device)
                            f_query = self.extract_feature(self.model_veid, X_query).detach().cpu()
                            f_gallery = [self.extract_feature(self.model_veid, X) for X in X_gallery]
                            f_gallery = torch.stack(f_gallery).detach().cpu()
                            scores = self.get_scores(f_query, f_gallery)
                            sorted_idx = np.argsort(scores)[::-1]
                            gallery_images_sorted = [gallery_images[i] for i in sorted_idx]

                            if scores[sorted_idx[0]] > 0.7:
                                best_image = gallery_images_sorted[0]
                                self.change_state("Voiture valide", "Image trouvée", "success")
                            else:
                                self.change_state("Plaque valide uniquement", "Pas d'image trouvée", "success")

                        else:
                            self.change_state("Plaque non-valide", "Plaque n°" + class_names, "detected")
    def change_state(self, result, instruct, state):
        data = {
            'content':[result, instruct, state]
        }
        try:
            requests.post(self.FLASK_URL+"state_update", json=data)
        except ConnectionError:
            print("Cannot update interface")

    def request_frame(self):
        try:
            return requests.get(self.FLASK_URL+'vframe').content
        except ConnectionError:
            print("Cannot get image")
            return None

    def get_image_list(self):
        response = requests.get(self.FLASK_URL+"images")
        if response.status_code == 200:
            data = response.json()
            return data["images"]
        else:
            print(f"Failed to fetch image list: {response.status_code}")
            return []

    def get_image_from_server(self, image):
        response = requests.get(self.FLASK_URL+"images/"+image)

        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img = Image.open(img_data)
            return img
        else:
            print(f"Failed to fetch image: {response.status_code}")
            return None

    def check_license_from_server(self, data):
        result = requests.get(self.FLASK_URL+"check_license/"+data).json()
        return result["result"]


if __name__ == '__main__':
    module = AIParkingModule()
    module.start()


