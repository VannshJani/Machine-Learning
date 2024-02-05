# %% [markdown]
# # Implementing Self-supervised Learning on Brick kilns

# %% [markdown]
# Let's import the necessary libraries

# %%
try:
    from astra.torch.models import ResNetClassifier
    import umap.umap_ as umap
except:
    %pip install git+https://github.com/sustainability-lab/ASTRA
    %pip install umap-learn

# %%
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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import umap.umap_ as umap

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial.distance import hamming

# ASTRA
from astra.torch.data import load_cifar_10
from astra.torch.utils import train_fn
from astra.torch.models import EfficientNet, MLP, MLPClassifier, EfficientNetClassifier

from itertools import permutations,product

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ### Loading Data

# %%
loaded_data = torch.load("data.pt")
index = loaded_data['index']
X_ban = loaded_data['images']
y_ban = loaded_data['labels']
#print shape of tensors
X_ban = X_ban / 255
    # mean normalize
X_ban = (X_ban - X_ban.mean(dim=(0, 2, 3), keepdim=True)) / X_ban.std(dim=(0, 2, 3), keepdim=True)
print(X_ban.shape)
print(y_ban.shape)

# %%
loaded_data = torch.load("test_data.pt")
index = loaded_data['index']
X_del = loaded_data['images']
y_del = loaded_data['labels']
#print shape of tensors
X_del = X_del / 255
    # mean normalize
X_del = (X_del - X_del.mean(dim=(0, 2, 3), keepdim=True)) / X_del.std(dim=(0, 2, 3), keepdim=True)
print(X_del.shape)
print(y_del.shape)

# %%
# Resizing images
X_ban = F.interpolate(X_ban, size=(225, 225), mode='bilinear', align_corners=False)
X_del = F.interpolate(X_del, size=(225, 225), mode='bilinear', align_corners=False)
# X_ban = X_ban[:10000]
# y_ban = y_ban[:10000]
# X_del = X_del[:5000]
# y_del = y_del[:5000]
X_ban.shape, X_del.shape

# %% [markdown]
# ### Plotting some images

# %%
# plt.figure(figsize=(6, 6))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(torch.einsum("chw->hwc", X_del[i].cpu()))
#     plt.axis('off')
#     plt.title("label",y_ban[i])
# plt.tight_layout()

# %% [markdown]
# ### Splitting Data

# %%
# # splitting X_ban into train and test, using Stratified mode
# X_train, X_test, y_train, y_test = train_test_split(X_ban, y_ban, test_size=1/3, random_state=42, stratify=y_ban)

# %%
# splitting X_del into train and test, using Stratified mode
X_train, X_test, y_train, y_test = train_test_split(X_del, y_del, test_size=1/3, random_state=42, stratify=y_ban)
X_train.shape, X_test.shape

# %%
result_dict = {}
# 1% of the training data
train_1_x, _, train_1_y, _ = train_test_split(X_train, y_train, test_size=0.99, random_state=42, stratify=y_train)
# 5% of the training data
train_5_x, _, train_5_y, _ = train_test_split(X_train, y_train, test_size=0.95, random_state=42, stratify=y_train)
# 10% of the training data
train_10_x, _, train_10_y, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42, stratify=y_train)
# 50% of the training data 
train_50_x, _, train_50_y, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)

# %%
def get_metrics(y_pred,y_label):
    with torch.no_grad():
        acc = accuracy_score(y_label,y_pred)
        f1 = f1_score(y_label,y_pred,average='macro')
        precision = precision_score(y_label,y_pred,average='macro')
        recall = recall_score(y_label,y_pred,average='macro')
        return acc,f1,precision,recall

def predict(model,X_test,y_test,percent,ssl,batch_size = 64):
    with torch.no_grad():
        model.eval()
        # incorporating batch size
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        y_pred = []
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            y_pred.append(output)
        y_pred = torch.cat(y_pred)
        y_pred = torch.argmax(y_pred,dim=1)
        y_pred = y_pred.cpu().numpy()
        y_test = y_test.cpu().numpy()
        acc,f1,precision,recall = get_metrics(y_pred,y_test)


        result_dict[percent+" "+ssl] = {"accuracy":acc,"f1":f1,"precision":precision,"recall":recall}

# %% [markdown]
# ### Training model on train set 1%

# %%
iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), train_1_x, train_1_y, lr=3e-4,
                                     batch_size=128, epochs=30, verbose=True)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
predict(ecf,X_test,y_test,"1%","no_SSL")

# %% [markdown]
# ### Training model on train set 5%

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), train_5_x, train_5_y, lr=3e-4,
                                     batch_size=128, epochs=30, verbose=True)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
predict(ecf,X_test,y_test,"5%","no_SSL")

# %% [markdown]
# ### Training model on train set 10%

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), train_10_x, train_10_y, lr=3e-4,
                                     batch_size=128, epochs=30, verbose=True)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
predict(ecf,X_test,y_test,"10%","no_SSL")

# %% [markdown]
# ### Training model on train set 50%

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), train_50_x, train_50_y, lr=3e-4,
                                     batch_size=128, epochs=30, verbose=True)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
predict(ecf,X_test,y_test,"50%","no_SSL")

# %% [markdown]
# ### Training model on train set 100%

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(),X_train, y_train, lr=3e-4,
                                     batch_size=256, epochs=30, verbose=True)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
predict(ecf,X_test,y_test,"100%","no_SSL")

# %%
print(len(X_train),len(X_test))

# %%
df2 = pd.DataFrame(result_dict)
df2

# %% [markdown]
# ### Oracle without SSL

# %%
# ### Train on train + pool
# ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=10)
# ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
# ecf.to(device)
# train_plus_pool_idx = torch.cat([train_idx, pool_idx])
# iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X[train_plus_pool_idx], y[train_plus_pool_idx],
#                                      lr=3e-4,
#                                         batch_size=1024, epochs=30)
# plt.plot(iter_losses)
# plt.xlabel("Iteration")
# plt.ylabel("Training loss")
# predict(ecf,X[test_idx],y[test_idx],"Oracle","no_SSL")

# %% [markdown]
# ## SSL

# %% [markdown]
# Task: Dividing images into patches and permuting them. Predict the permutation number as a classification task

# %%

X_pool = torch.cat([X_ban,X_del])
y_pool = torch.cat([y_ban,y_del])

# X_pool = X
# y_pool = y

X_pool.shape, y_pool.shape

# %%
# Divinding image into 9 patches
def divide_into_patches(images):                 
  patches = torch.split(images,75,dim=3)
  # print(patches[0].shape)
  patches = [torch.unsqueeze(patch,1) for patch in patches]
  # print(patches[2].shape,len(patches))
  patches = torch.cat(patches,dim=1)
  patches = torch.split(patches,75,dim=3)
  patches = torch.cat(patches,dim=1)
  return patches


images_with_patches = divide_into_patches(X_pool)
images_with_patches.shape

# %%
# Viusalizing patches
sample_image = images_with_patches[0]
plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(torch.einsum("chw->hwc", sample_image[i]))
    plt.axis('off')
    plt.title(f"Patch: {i+1}")
plt.tight_layout()

# %%
# # Original image
# print(X_pool[0].shape)
# plt.imshow(torch.einsum("chw->hwc", X_pool[0]))
# plt.tight_layout()

# %%
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

        # if cnt_iterations % 100 == 0:
        #     print ("Already performed count of iterations with pairs of jigsaw permutations", cnt_iterations)
        #     print ("Length of set of taken: ",len(set_of_taken))

    # print ("No of iterations it took to build top - {} permutations array = {}".format(permuts_to_take, cnt_iterations))
    # print ("No of permutations", len(set_of_taken))
    selected_permuts = []
    for ind, perm_id in enumerate(set_of_taken):
        selected_permuts.append(permuts_array[perm_id])
    selected_permuts = np.array(selected_permuts)
    selected_permuts = torch.tensor(selected_permuts)
    return selected_permuts
    
selected_permuts = top_k_hamming(64)
selected_permuts.shape

# %%
def get_rand_perms(nperms,selected_perms):
    rand_idx = torch.randint(0, len(selected_perms), (nperms,))
    return selected_perms[rand_idx],rand_idx

rand_perms,y_labels = get_rand_perms(len(X_del)+len(X_ban),selected_permuts)
rand_perms.shape

# %%
rand_perms[:10]

# %%
# rearrange the patches of images_with_patches for each of the 15000 images based on the random permutations

def rearrange_patches(images_with_patches,random_permutations):

    # images_with_patches = torch.rand(15000,9,3,75,75)
    # random_permutations = rand_perms
    # random_permutations = get_rand_perms(15000,selected_permuts)

    # rearrange the patches of images_with_patches for each of the 15000 images based on the random permutations
    images_with_patches_rearranged = torch.zeros(images_with_patches.shape)
    for i in range(images_with_patches.shape[0]):
        img = images_with_patches[i]
        img_rearranged = img[random_permutations[i]]
        images_with_patches_rearranged[i] = img_rearranged

    return images_with_patches_rearranged

permuted_images = rearrange_patches(images_with_patches,rand_perms)
permuted_images.shape,y_labels.shape

# %%
# def top_k_hamming(k):
#     # find k permutations with highest hamming distance
#     permuts_list = list(permutations(range(9)))
#     permuts_array = np.array(permuts_list)
#     no_permuts = len(permuts_list)

#     permuts_to_take = k
#     set_of_taken = set()
#     cnt_iterations = 0
#     while True:
#         cnt_iterations += 1
#         x = np.random.randint(0, no_permuts )
#         y = np.random.randint(0, no_permuts )
#         permut_1 = permuts_array[x]
#         permut_2 = permuts_array[y]
#         hd = hamming(permut_1, permut_2)

#         if hd > 0.9 and (not x in set_of_taken) and (not y in set_of_taken):
#             set_of_taken.add(x)
#             set_of_taken.add(y)

#             if len(set_of_taken) == permuts_to_take:
#                 break

#         # if cnt_iterations % 100 == 0:
#         #     print ("Already performed count of iterations with pairs of jigsaw permutations", cnt_iterations)
#         #     print ("Length of set of taken: ",len(set_of_taken))

#     # print ("No of iterations it took to build top - {} permutations array = {}".format(permuts_to_take, cnt_iterations))
#     # print ("No of permutations", len(set_of_taken))
#     selected_permuts = []
#     for ind, perm_id in enumerate(set_of_taken):
#         selected_permuts.append(permuts_array[perm_id])
#     selected_permuts = np.array(selected_permuts)
#     selected_permuts = torch.tensor(selected_permuts)
#     return selected_permuts
    

# def temp_func(image):
#     # image shape is 9,3,11,11

#     selected_perms = top_k_hamming(k=64)
#     p_image = torch.unsqueeze(image,0)
#     for i in range(len(selected_perms)):
#         p_image = torch.cat([p_image,torch.unsqueeze(image[selected_perms[i]],0)],dim=0)
#     p_image = p_image[1:]
#     return p_image



# def permute_image(images,number_of_permutations):
    
#     p_images = torch.vmap(temp_func)(images)
#     shape1 = (len(images),9,3,75,75)
#     shape2 = (len(images))
#     final_images = torch.zeros(shape1)
#     y_labels = torch.zeros(shape2)
#     for i in range(0,p_images.shape[0]):
#         rand_idx = torch.randint(0,number_of_permutations,(1,))
#         final_images[i] = torch.squeeze(p_images[i],0)[rand_idx]
#         y_labels[i] =  rand_idx

#     return final_images,y_labels
        

# def permute_patches(images,number_of_permutations=64,batch_size=64):
    
#     # incorporating batch size
#     all_permuted_images = []
#     # y_labels_list = []
#     # for i in range(0,images.shape[0],batch_size):
#     #     permuted_images,y_labels = permute_image(images[i:i+batch_size],number_of_permutations)
#     #     all_permuted_images.append(permuted_images)
#     #     y_labels_list.append(y_labels)
#     all_permuted_images,y_labels = permute_image(images,number_of_permutations)
#     all_permuted_images = all_permuted_images.reshape(-1,9,3,11,11)
#     y_labels = y_labels.reshape(-1)
#     # y_labels = torch.tensor(y_labels_list).reshape(-1)
#     # all_permuted_images = torch.cat(all_permuted_images)
#     # y_labels = torch.cat(y_labels_list)
#     return all_permuted_images,y_labels

# images_with_patches = images_with_patches
# permuted_images,y_labels = permute_patches(images_with_patches,64)
# permuted_images.shape,y_labels.shape
    

# %%
# Viusalizing patches
sample_image = permuted_images[0]
plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(torch.einsum("chw->hwc", sample_image[i]))
    plt.axis('off')
    plt.title(f"Patch: {i+1}")
plt.tight_layout()

# %%
y_labels.unique()

# %%
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
        # for i in range(len(patches)):
        #     patches[i] = self.model(patches[i])  # (batch, 1280)
        #     patches[i] = F.relu(self.fc1(patches[i]))  # (batch, 256)
        x = torch.cat([p1,p2,p3,p4,p5,p6,p7,p8,p9],dim=1) # (batch, 2304)
        x = F.relu(self.fc2(x))  # (batch, 100)
        x = self.fc3(x)  # (batch, 64)
        return x


# %%
model = EfficientNet().to(device)
aggregator = MLPClassifier(1280*9,[1280,1024,256],n_classes=64).to(device)


# %%
def train_ssl_model(model,aggregator, loss_fn, images, labels, lr=3e-4, batch_size=512, epochs=30, verbose=True):
    
    iter_losses = []
    epoch_losses = []

    model.train()
    aggregator.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(list(model.parameters())+list(aggregator.parameters()), lr=lr)
    for epoch in range(epochs):
        for i in range(0, len(images), batch_size):
        
            p1, p2, p3, p4,p5,p6,p7,p8,p9 = images[i:i + batch_size][:, 0], images[i:i + batch_size][:, 1], images[i:i + batch_size][:, 2], images[i:i + batch_size][:, 3],images[i:i + batch_size][:, 4],images[i:i + batch_size][:, 5],images[i:i + batch_size][:, 6],images[i:i + batch_size][:, 7],images[i:i + batch_size][:, 8]
            p1,p2,p3,p4,p5,p6,p7,p8,p9 = p1.to(device),p2.to(device),p3.to(device),p4.to(device),p5.to(device),p6.to(device),p7.to(device),p8.to(device),p9.to(device)
            flat_labels = labels[i:i + batch_size]
            flat_labels = flat_labels.type(torch.LongTensor)
            patches = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
            patch_out = []
            for patch in patches:
                patch_out.append(model(patch))
            patch_out = torch.cat(patch_out,dim=-1)
            y_pred_prob = aggregator(patch_out)
            # y_pred_prob = model(p1,p2,p3,p4,p5,p6,p7,p8,p9)
            loss = loss_fn(y_pred_prob, flat_labels.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_losses.append(loss.item())


        epoch_losses.append(loss.item())
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")
        if epoch % 10 == 0 and epoch != 0 or epoch == epochs - 1:
            # save parameters
            path = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/model_{epoch}_delban.pth"
            path_a = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/aggregator_{epoch}_delban.pth"
            torch.save(model.state_dict(), path)
            torch.save(aggregator.state_dict(), path_a)

    return iter_losses, epoch_losses


# %%
# # predict permutations and get accuracy

# def predict_permutations(model,aggregator,images,labels):
#     with torch.no_grad():
#         model.eval()
#         aggregator.eval()
#         p1,p2,p3,p4,p5,p6,p7,p8,p9 = images[:,0],images[:,1],images[:,2],images[:,3],images[:,4],images[:,5],images[:,6],images[:,7],images[:,8]
#         p1,p2,p3,p4,p5,p6,p7,p8,p9 = p1.to(device),p2.to(device),p3.to(device),p4.to(device),p5.to(device),p6.to(device),p7.to(device),p8.to(device),p9.to(device)
#         patches = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
#         patch_out = []
#         for patch in patches:
#             patch_out.append(model(patch))
#         patch_out = torch.cat(patch_out,dim=-1)
#         y_pred_prob = aggregator(patch_out)
#         # y_pred_prob = model(p1,p2,p3,p4,p5,p6,p7,p8,p9)
#         y_pred = torch.argmax(y_pred_prob,dim=1)
#         return y_pred

# def results(y_labels,y_pred,plot_confusion = False):
#     y_labels = y_labels.to(device) 
#     print(f"Accuracy = {(y_labels == y_pred).float().mean()}")
#     if plot_confusion:
#         cm = confusion_matrix(y_labels.cpu(), y_pred.cpu())
#         unique_labels = np.unique(y_labels.cpu())
#         cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot(values_format='d', cmap='Blues')
#         plt.show()

# %%
permuted_images.shape,y_labels.shape

# %%
iter_losses, epoch_losses = train_ssl_model(model,aggregator,nn.CrossEntropyLoss(),permuted_images.to(device),y_labels.to(device),lr=3e-4,
                                     batch_size=256, epochs=50, verbose=True)

plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")


# %%
# y_pred = predict_permutations(model,aggregator,permuted_images,y_labels)
# results(y_labels,y_pred)

# %%
# using ssl model parameters as pretrained to train efficientnet classifier
num_epochs = 50
for i in range(num_epochs,0,-10):
    if os.path.exists(f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/model_{i}.pth"):
        path = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/model_{i}.pth"
        path_a = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/aggregator_{i}.pth"
        break
if os.path.exists(f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/model_{num_epochs-1}_delban.pth"):
    path = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/model_{num_epochs-1}_delban.pth"
    path_a = f"/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/aggregator_{num_epochs-1}_delban.pth"
print(path,path_a)


# %% [markdown]
# ### Training model on 1% Train set with SSL

# %%
# ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=10).to(device)
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
ecf.featurizer.load_state_dict(torch.load(path))
last_layer = ecf.classifier.classifier
last_layer_aggregator = aggregator.classifier
layers = list(ecf.classifier.children())[:-1]
# set mlp weights as classifier weights
layers_mlp = list(aggregator.children())[:-1]
# removed the input layer in layers_mlp
list(aggregator.children())[0].input_layer = list(aggregator.children())[0].hidden_layer_1
list(aggregator.children())[0].hidden_layer_1 = list(aggregator.children())[0].hidden_layer_2
list(aggregator.children())[0].hidden_layer_2 = nn.Identity()
layers_mlp_matched = list(aggregator.children())[0]

layers = layers_mlp_matched
ecf.classifier = nn.Sequential(*[layers])
aggregator = nn.Sequential(*[layers_mlp_matched])
ecf.classifier.load_state_dict(aggregator.state_dict())
layers_1 = list(ecf.classifier.children())
layers_1.append(last_layer)
ecf.classifier = nn.Sequential(*layers_1)

iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X_ban, y_ban, lr=3e-4,verbose=True,batch_size=512,epochs=30)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
predict(ecf,train_1_x,train_1_y,"1%","yes_SSL")

# %%
aggregator = MLPClassifier(1280*9,[1280,1024,256],n_classes=64).to(device)
aggregator.load_state_dict(torch.load(path_a))

# %% [markdown]
# ### Training model on 5% Train set with SSL

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
ecf.featurizer.load_state_dict(torch.load(path))
last_layer = ecf.classifier.classifier
last_layer_aggregator = aggregator.classifier
layers = list(ecf.classifier.children())[:-1]
# set mlp weights as classifier weights
layers_mlp = list(aggregator.children())[:-1]
# removed the input layer in layers_mlp
list(aggregator.children())[0].input_layer = list(aggregator.children())[0].hidden_layer_1
list(aggregator.children())[0].hidden_layer_1 = list(aggregator.children())[0].hidden_layer_2
list(aggregator.children())[0].hidden_layer_2 = nn.Identity()
layers_mlp_matched = list(aggregator.children())[0]

layers = layers_mlp_matched
ecf.classifier = nn.Sequential(*[layers])
aggregator = nn.Sequential(*[layers_mlp_matched])
ecf.classifier.load_state_dict(aggregator.state_dict())
layers_1 = list(ecf.classifier.children())
layers_1.append(last_layer)
ecf.classifier = nn.Sequential(*layers_1)

iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X_ban, y_ban, lr=3e-4,verbose=True,batch_size=512,epochs=30)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
predict(ecf,train_5_x,train_5_y,"5%","yes_SSL")

# %%
aggregator = MLPClassifier(1280*9,[1280,1024,256],n_classes=64).to(device)
aggregator.load_state_dict(torch.load(path_a))

# %% [markdown]
# ### Training model on 10% Train set with SSL

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
ecf.featurizer.load_state_dict(torch.load(path))
last_layer = ecf.classifier.classifier
last_layer_aggregator = aggregator.classifier
layers = list(ecf.classifier.children())[:-1]
# set mlp weights as classifier weights
layers_mlp = list(aggregator.children())[:-1]
# removed the input layer in layers_mlp
list(aggregator.children())[0].input_layer = list(aggregator.children())[0].hidden_layer_1
list(aggregator.children())[0].hidden_layer_1 = list(aggregator.children())[0].hidden_layer_2
list(aggregator.children())[0].hidden_layer_2 = nn.Identity()
layers_mlp_matched = list(aggregator.children())[0]

layers = layers_mlp_matched
ecf.classifier = nn.Sequential(*[layers])
aggregator = nn.Sequential(*[layers_mlp_matched])
ecf.classifier.load_state_dict(aggregator.state_dict())
layers_1 = list(ecf.classifier.children())
layers_1.append(last_layer)
ecf.classifier = nn.Sequential(*layers_1)

iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X_ban,y_ban, lr=3e-4,verbose=True,batch_size=512,epochs=30)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
predict(ecf,train_10_x,train_10_y,"10%","yes_SSL")

# %%
aggregator = MLPClassifier(1280*9,[1280,1024,256],n_classes=64).to(device)
aggregator.load_state_dict(torch.load(path_a))

# %% [markdown]
# ### Training model on 50% Train set with SSL

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
ecf.featurizer.load_state_dict(torch.load(path))
last_layer = ecf.classifier.classifier
last_layer_aggregator = aggregator.classifier
layers = list(ecf.classifier.children())[:-1]
# set mlp weights as classifier weights
layers_mlp = list(aggregator.children())[:-1]
# removed the input layer in layers_mlp
list(aggregator.children())[0].input_layer = list(aggregator.children())[0].hidden_layer_1
list(aggregator.children())[0].hidden_layer_1 = list(aggregator.children())[0].hidden_layer_2
list(aggregator.children())[0].hidden_layer_2 = nn.Identity()
layers_mlp_matched = list(aggregator.children())[0]

layers = layers_mlp_matched
ecf.classifier = nn.Sequential(*[layers])
aggregator = nn.Sequential(*[layers_mlp_matched])
ecf.classifier.load_state_dict(aggregator.state_dict())
layers_1 = list(ecf.classifier.children())
layers_1.append(last_layer)
ecf.classifier = nn.Sequential(*layers_1)

iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X_ban, y_ban, lr=3e-4,verbose=True,batch_size=512,epochs=30)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
predict(ecf,train_50_x,train_50_y,"50%","yes_SSL")

# %%
aggregator = MLPClassifier(1280*9,[1280,1024,256],n_classes=64).to(device)
aggregator.load_state_dict(torch.load(path_a))

# %% [markdown]
# ### Training model on 100% Train set with SSL

# %%
ecf = EfficientNetClassifier(dense_hidden_dims=[1280,1024,256],n_classes=2)
ecf.featurizer.efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(4, 4),padding=(1,1), bias=False)
ecf.to(device)
ecf.featurizer.load_state_dict(torch.load(path))
last_layer = ecf.classifier.classifier
last_layer_aggregator = aggregator.classifier
layers = list(ecf.classifier.children())[:-1]
# set mlp weights as classifier weights
layers_mlp = list(aggregator.children())[:-1]
# removed the input layer in layers_mlp
list(aggregator.children())[0].input_layer = list(aggregator.children())[0].hidden_layer_1
list(aggregator.children())[0].hidden_layer_1 = list(aggregator.children())[0].hidden_layer_2
list(aggregator.children())[0].hidden_layer_2 = nn.Identity()
layers_mlp_matched = list(aggregator.children())[0]

layers = layers_mlp_matched
ecf.classifier = nn.Sequential(*[layers])
aggregator = nn.Sequential(*[layers_mlp_matched])
ecf.classifier.load_state_dict(aggregator.state_dict())
layers_1 = list(ecf.classifier.children())
layers_1.append(last_layer)
ecf.classifier = nn.Sequential(*layers_1)


iter_losses, epoch_losses = train_fn(ecf,nn.CrossEntropyLoss(), X_ban, y_ban, lr=3e-4,verbose=True,batch_size=512,epochs=30)
plt.plot(iter_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
predict(ecf,X_del,y_del,"100%","yes_SSL")

# %%
df = pd.DataFrame(result_dict)
df

# %%
# saving best
path = "/home/vannsh.jani/brick_kilns/githubrepo/Machine-Learning/best_model_prec.pth"
torch.save(ecf.state_dict(), path)

# %%
93,78,85,74

# %%



