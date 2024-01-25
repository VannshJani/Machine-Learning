import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import pandas as pd
import umap.umap_ as umap

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial.distance import hamming

# ASTRA
from astra.torch.data import load_cifar_10
from astra.torch.utils import train_fn
from astra.torch.models import EfficientNet, MLP, MLPClassifier, EfficientNetClassifier

from itertools import permutations,product
# selecting gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading CIFAR10

dataset = load_cifar_10()

# Resizing images
X = F.interpolate(dataset.data, size=(33, 33), mode='bilinear', align_corners=False)

plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(torch.einsum("chw->hwc", X[i].cpu()))
    plt.axis('off')
    plt.title(dataset.classes[dataset.targets[i]])
plt.tight_layout()

# Splitting data into train and test
n_train = 10000
n_test = 40000

y = dataset.targets
print(X[0].shape)
print(X.shape, X.dtype)
print(X.min(), X.max())
print(y.shape, y.dtype)

torch.manual_seed(0)
idx = torch.randperm(len(X))
train_idx = idx[:n_train]
pool_idx = idx[n_train:-n_test]
test_idx = idx[-n_test:]
print(len(train_idx), len(pool_idx), len(test_idx))

ecf = EfficientNetClassifier(n_classes=10).to(device) 

def get_accuracy(net, X, y):
    with torch.no_grad():
        logits_pred = net(X)
        y_pred = logits_pred.argmax(dim=1)
        acc = (y_pred == y).float().mean()
        return y_pred, acc

def predict(net, classes, plot_confusion_matrix=False):
    with torch.no_grad():
        for i, (name, idx) in enumerate(zip(("train", "pool", "test"), [train_idx, pool_idx, test_idx])):
            X_dataset = X[idx].to(device)
            y_dataset = y[idx].to(device)
            y_pred, acc = get_accuracy(net, X_dataset, y_dataset)
            print(f'{name} set accuracy: {acc*100:.2f}%')
            if plot_confusion_matrix:
                cm = confusion_matrix(y_dataset.cpu(), y_pred.cpu())
                cm_display = ConfusionMatrixDisplay(cm, display_labels=classes).plot(values_format='d' , cmap='Blues')
                # Rotate the labels on x-axis to make them readable
                _ = plt.xticks(rotation=90)
                plt.show()


# iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X[train_idx], y[train_idx], lr=3e-4,
#                                      batch_size=128, epochs=30, verbose=True)

# plt.plot(iter_losses)
# plt.xlabel("Iteration")
# plt.ylabel("Training loss")
                
# predict(ecf, dataset.classes, plot_confusion_matrix=True)
                
### Train on train + pool
# train_plus_pool_idx = torch.cat([train_idx, pool_idx])
# iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X[train_plus_pool_idx], y[train_plus_pool_idx],
#                                      lr=3e-4,
#                                         batch_size=1024, epochs=30)

# plt.plot(iter_losses)
# plt.xlabel("Iteration")
# plt.ylabel("Training loss")
                
# predict(ecf, dataset.classes, plot_confusion_matrix=True)
                
X_pool = X[pool_idx]
y_pool = y[pool_idx]

# X_pool.shape, y_pool.shape

# Divinding image into 9 patches
def divide_into_patches(images):                 
  patches = torch.split(images,11,dim=3)
  # print(patches[0].shape)
  patches = [torch.unsqueeze(patch,1) for patch in patches]
  # print(patches[2].shape,len(patches))
  patches = torch.cat(patches,dim=1)
  patches = torch.split(patches,11,dim=3)
  patches = torch.cat(patches,dim=1)
  return patches


images_with_patches = divide_into_patches(X_pool)
# images_with_patches.shape

# Viusalizing patches
sample_image = images_with_patches[0]
plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(torch.einsum("chw->hwc", sample_image[i]))
    plt.axis('off')
    plt.title(f"Patch: {i+1}")
plt.tight_layout()

# Original image
print(X_pool[0].shape)
plt.imshow(torch.einsum("chw->hwc", X_pool[0]))
plt.tight_layout()


def top_k_hamming(k):
    # find k permutations with highest hamming distance
    permuts_list = list(permutations(range(9)))
    permuts_array = np.array(permuts_list)
    no_permuts = len(permuts_list)

    permuts_to_take = k
    set_of_taken = set()
    cnt_iterations = 0
    while True:
        cnt_iterations += 1
        x = np.random.randint(0, no_permuts )
        y = np.random.randint(0, no_permuts )
        permut_1 = permuts_array[x]
        permut_2 = permuts_array[y]
        hd = hamming(permut_1, permut_2)

        if hd > 0.9 and (not x in set_of_taken) and (not y in set_of_taken):
            set_of_taken.add(x)
            set_of_taken.add(y)

            if len(set_of_taken) == permuts_to_take:
                break

        if cnt_iterations % 100 == 0:
            print ("Already performed count of iterations with pairs of jigsaw permutations", cnt_iterations)
            print ("Length of set of taken: ",len(set_of_taken))

    print ("No of iterations it took to build top - {} permutations array = {}".format(permuts_to_take, cnt_iterations))
    print ("No of permutations", len(set_of_taken))
    selected_permuts = []
    for ind, perm_id in enumerate(set_of_taken):
        selected_permuts.append(permuts_array[perm_id])
    selected_permuts = np.array(selected_permuts)
    selected_permuts = torch.tensor(selected_permuts)
    return selected_permuts
    


# def get_ranks(a):
#     val = 0
#     ranked_indices = torch.argsort(a)
#     ranks = torch.zeros_like(ranked_indices)
#     for i in range(len(ranked_indices)-1):
#         if i==len(ranked_indices)-2:
#             if a[ranked_indices[i]] == a[ranked_indices[i+1]]:
#                 ranks[ranked_indices[i]] = val
#                 ranks[ranked_indices[i+1]] = val
#             else:
#                 ranks[ranked_indices[i]] = val
#                 ranks[ranked_indices[i+1]] = val+1
#             break
#         ranks[ranked_indices[i]] = val
#         if a[ranked_indices[i]] != a[ranked_indices[i+1]]:
#             val += 1
#             continue
#         else:
#             continue
#     return ranks


def temp_func(image):
    # image shape is 9,3,11,11
    selected_perms = top_k_hamming(k=64)
    p_image = torch.unsqueeze(image,0)  # includes the image with no perm
    for i in range(len(selected_perms)):
        p_image = torch.cat([p_image,torch.unsqueeze(image[selected_perms[i]],0)],dim=0)
    p_image = p_image[1:]     # excluding the image with no perm
    y_labels = torch.arange(len(selected_perms))
    return p_image,y_labels


def permute_image(images,number_of_permutations):
    p_image,y_labels = torch.vmap(temp_func)(images)
    # if type == 'random':
    #     idxs = torch.randint(low = 0,high = np.factorial(9),size=(number_of_permutations,))
    #     p_image = p_image[:,idxs]
    #     y_labels = y_labels[:,idxs]

    return p_image,y_labels
        

def permute_patches(images,number_of_permutations=64):
    
    all_permuted_images,y_labels = permute_image(images,number_of_permutations)
    all_permuted_images = all_permuted_images.reshape(-1,9,3,11,11)
    y_labels = y_labels.reshape(-1)
    # y_labels = get_ranks(y_labels)
    return all_permuted_images,y_labels

permuted_images,y_labels = permute_patches(images_with_patches,64)
# permuted_images.shape,y_labels.shape

class SSL_Model(nn.Module):
    def __init__(self,number_of_permutations=64):
        super().__init__()
        self.model = EfficientNet()
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(2304, 100)
        self.fc3 = nn.Linear(100, number_of_permutations)
        self.softmax = nn.Softmax(dim=1)
 
        
    def forward(self, p1,p2,p3,p4,p5,p6,p7,p8,p9):
        p1 = self.model(p1)  # (batch, 1280)
        p2 = self.model(p2)
        p3 = self.model(p3)
        p4 = self.model(p4)
        p5 = self.model(p5)
        p6 = self.model(p6)
        p7 = self.model(p7)
        p8 = self.model(p8)
        p9 = self.model(p9)
        p1 = F.relu(self.fc1(p1)) # (batch, 256)
        p2 = F.relu(self.fc1(p2))
        p3 = F.relu(self.fc1(p3))
        p4 = F.relu(self.fc1(p4))
        p5 = F.relu(self.fc1(p5))
        p6 = F.relu(self.fc1(p6))
        p7 = F.relu(self.fc1(p7))
        p8 = F.relu(self.fc1(p8))
        p9 = F.relu(self.fc1(p9))
        x = torch.cat([p1,p2,p3,p4,p5,p6,p7,p8,p9],dim=1) # (batch, 2304)
        x = F.relu(self.fc2(x))  # (batch, 100)
        x = self.softmax(self.fc3(x))  # (batch, 64)
        return x

def train_ssl_model(model, loss_fn, images, labels, lr=3e-4, batch_size=512, epochs=30, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    iter_losses = []
    epoch_losses = []

    for epoch in range(epochs):
        for i in range(0, len(images), batch_size):
            model.train()
            optimizer.zero_grad()
            p1, p2, p3, p4,p5,p6,p7,p8,p9 = images[i:i + batch_size][:, 0], images[i:i + batch_size][:, 1], images[i:i + batch_size][:, 2], images[i:i + batch_size][:, 3],images[i:i + batch_size][:, 4],images[i:i + batch_size][:, 5],images[i:i + batch_size][:, 6],images[i:i + batch_size][:, 7],images[i:i + batch_size][:, 8]
            p1,p2,p3,p4,p5,p6,p7,p8,p9 = p1.to(device),p2.to(device),p3.to(device),p4.to(device),p5.to(device),p6.to(device),p7.to(device),p8.to(device),p9.to(device)
            flat_labels = labels[i:i + batch_size]
            y_pred_prob = model(p1, p2, p3, p4,p5,p6,p7,p8,p9).to(device)
            loss = loss_fn(y_pred_prob, flat_labels.to(device))
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())

        epoch_losses.append(loss.item())
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

    return iter_losses, epoch_losses

# predict permutations and get accuracy

def predict_permutations(model,images,labels):
    with torch.no_grad():
        model.eval()
        p1,p2,p3,p4,p5,p6,p7,p8,p9 = images[:,0],images[:,1],images[:,2],images[:,3],images[:,4],images[:,5],images[:,6],images[:,7],images[:,8]
        p1,p2,p3,p4,p5,p6,p7,p8,p9 = p1.to(device),p2.to(device),p3.to(device),p4.to(device),p5.to(device),p6.to(device),p7.to(device),p8.to(device),p9.to(device)
    
        y_pred_prob = model(p1,p2,p3,p4,p5,p6,p7,p8,p9).to(device)
        y_pred = torch.argmax(y_pred_prob,dim=1)
        return y_pred

def results(y_labels,y_pred):
    y_labels = y_labels.to(device) 
    print(f"Accuracy = {(y_labels == y_pred).float().mean()}")
    cm = confusion_matrix(y_labels.cpu(), y_pred.cpu())
    unique_labels = np.unique(y_labels.cpu())
    cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot(values_format='d', cmap='Blues')
    plt.show()



ssl_model = SSL_Model().to(device)
iter_losses, epoch_losses = train_ssl_model(ssl_model,nn.CrossEntropyLoss(),permuted_images,y_labels,lr=0.01,
                                     batch_size=1024, epochs=50, verbose=True)

plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")

y_pred = predict_permutations(ssl_model,permuted_images,y_labels)
results(y_labels,y_pred)


    


