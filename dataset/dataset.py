from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import cv2
from torchvision.transforms import ToTensor
import torch


class BUSIDataset_image_addition3(Dataset):
    def __init__(self, image_path, addition1_path, addition2_path, addition3_path, mapping_path, transform):
        super().__init__()
        self.image_path = image_path
        self.addition1_path = addition1_path
        self.addition2_path = addition2_path
        self.addition3_path = addition3_path
        self.mapping_path = mapping_path
        self.transform = transform

        self.image_names = []
        self.labels = []
        # label_transform = {
        #         "benign": 0,  # 良性
        #         "normal": 1,  # 正常
        #         "malignant": 2,  # 恶性
        #     }
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("L")
        addition1 = Image.open(os.path.join(self.addition1_path, image_name)).convert("L")
        addition2 = Image.open(os.path.join(self.addition2_path, image_name)).convert("L")
        addition3 = Image.open(os.path.join(self.addition3_path, image_name)).convert("L")
        image = self.transform(image)
        addition1 = self.transform(addition1)
        addition2 = self.transform(addition2)
        addition3 = self.transform(addition3)


        image = torch.concat((image, addition1, addition2, addition3), 0)

        return image, label



class BUSIDataset_image_addition1(Dataset):
    def __init__(self, image_path, addition_path, mapping_path, transform):
        super().__init__()
        self.image_path = image_path
        self.addition_path = addition_path
        self.mapping_path = mapping_path
        self.transform = transform
        
        self.image_names = []
        self.labels = []
        # label_transform = {
        #         "benign": 0,  # 良性
        #         "normal": 1,  # 正常
        #         "malignant": 2,  # 恶性
        #     }
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        addition = Image.open(os.path.join(self.addition_path, image_name)).convert("L")
        image = self.transform(image)
        addition = self.transform(addition)

        image = torch.concat((image, addition), 0)

        return image, label



class BUSIDataset(Dataset):
    def __init__(self, image_path, mapping_path, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mapping_path = mapping_path
        self.transform = transform

        self.image_names = []
        self.labels = []
        # label_transform = {
        #         "benign": 0,  # 良性
        #         "normal": 1,  # 正常
        #         "malignant": 2,  # 恶性
        #     }
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label



class BUSIDataset_image_mask(Dataset):
    def __init__(self, image_path, mask_path, mapping_path, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mapping_path = mapping_path
        self.transform = transform

        self.image_names = []
        self.labels = []
        # label_transform = {
        #         "benign": 0,  # 良性
        #         "normal": 1,  # 正常
        #         "malignant": 2,  # 恶性
        #     }
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        img = cv2.imread(self.image_path + image_name, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + image_name, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, img.shape[:2][::-1])

        T = self.transform(image=img, mask=mask)
        img, mask = T["image"], T["mask"]

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = ToTensor()(Image.fromarray(mask).convert("L"))

        return img, mask, label



class BUSIDataset_image_mask_name(Dataset):
    def __init__(self, image_path, mask_path, mapping_path, transform=None):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.mapping_path = mapping_path
        self.transform = transform

        self.image_names = []
        self.labels = []
        # label_transform = {
        #         "benign": 0,  # 良性
        #         "normal": 1,  # 正常
        #         "malignant": 2,  # 恶性
        #     }
        label_transform = {
                "Benign": 0,  # 良性
                "Malignant": 1,  # 恶性
            }
        
        with open(self.mapping_path, "r") as f:
            info = f.read().strip().split("\n")
            for line in info:
                name, label, diagnosis = line.split()
                self.image_names.append(name)
                self.labels.append(label_transform[label])

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        img = cv2.imread(self.image_path + image_name, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + image_name, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, img.shape[:2][::-1])

        T = self.transform(image=img, mask=mask)
        img, mask = T["image"], T["mask"]

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = ToTensor()(Image.fromarray(mask).convert("L"))

        return img, mask, label, image_name
