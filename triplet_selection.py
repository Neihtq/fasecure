# Triplet Selection for the training process
# from the following paper: https://arxiv.org/pdf/1902.11007.pdf
# there were tested various triplet selections and tested on the LWF Dataset
# the highest accuracy got the Batch Min Min and Batch Min Max


#open to do:
#after each epoch: save Model
#before training every new epoch: load model from last epoch,
#call calculateEmbedding,
#and then make a new dataset with the dataloader

#the following class is the class for the Dataset


class LFWDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        # returns amount of classes
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        img_path = os.path.join(folder, listdir(folder)[0])
        anchor = self.get_image(img_path)
        # get positive which is farest away from anchor
        img_path = os.path.join(folder, listdir(folder)[1])
        positive = self.get_image(img_path)
        negative = self.get_negative(idx)

        return label, anchor, positive, negative

    def get_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

    def get_positive(self, idx):
        # get positive which is most far away from L2 distance
        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        difference = 0
        diff_tmp = difference
        curr_path = ""
        for embedding in self.embeddings[label]:
            distance = np.linalg.norm(self.embeddings[label][0][0] - embedding[0])
            difference = max(distance, difference)
            if difference != curr_path:
                curr_path = embedding[1]
        return self.get_image(curr_path)

    def get_negative(self, idx):
        # get negative which is most closest from L2 distance
        label = self.labels[idx]
        folder = os.path.join(self.root, label)

        for l in labels:
            if l!=label:
                for embedding in self.embeddings[l]:
                    distance = np.linalg.norm(self.embeddings[label][0][0] - embedding[0])
                    difference = min(distance, difference)
                    if difference != curr_path:
                        curr_path = embedding[1]
        return self.get_image(curr_path)

    def calculateEmbedding(self, idx):
        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        self.embeddings = {}

        for i in listdir(folder):
            img_path = os.path.join(folder, i)
            embedding = embedding_model(img_path)
            if label not in embeddings:
                embeddings[label] =
            self.embeddings[label].append((embedding,img_path))
