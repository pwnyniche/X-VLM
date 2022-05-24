import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class cosmos_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 50
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['img_local_path'])

        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)


        caption1 = ann['caption1'][0]
        caption2 = ann['caption2'][0]

        if ann['context_label'][0] == True:
            label = 1

        elif ann['context_label'][0] == False:
            label = 0

        else:
            raise ValueError(f"unsupported label: {ann['context_label'][0]}")

        return image, caption1, caption2, label
