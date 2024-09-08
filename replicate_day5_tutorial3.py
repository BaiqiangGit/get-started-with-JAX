
# Build neural network from scratch
import os
import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import jit, grad, vmap, pmap
from jax import make_jaxpr, tree
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets

"""
jax do not come with a dataloader: use pytorch or tf dl

"""

## MLP training on MNIST
# todo: init MLP and add the predict
# todo: add data loading in pytorch
# toto: add the training loop, loss_fn

def start_new_block(msg = ''):
    print()
    print(">>>----------------------------------------------------")
    if msg: print(msg)
    print()


#------------------------------------------
start_new_block("Init MLP via numpy:")
def init_MLP(layer_widths, scale = 0.01):
    params = [] # weights [out, in]
    for in_width, out_width in zip(layer_widths[:-1], layer_widths[1:]):
        layer = [scale * np.random.randn(out_width, in_width), np.random.randn(out_width)]
        params.append(layer)
    return params

MLP_params = init_MLP([784, 512, 256, 10])
print(jax.tree.map(lambda x:x.shape, MLP_params))


#------------------------------------------
start_new_block("Init MLP via jax random:")
seed = 0
rng  = jax.random.PRNGKey(seed=seed)

def init_MLP(layer_widths, parent_key, scale = 0.01):
    params = []
    keys = jax.random.split(parent_key, len(layer_widths)-1) # get different keys
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key) # split again 
        layer = [scale * jax.random.normal(key=weight_key, shape=(out_width, in_width)),
                 jax.random.normal(key=bias_key, shape=(out_width, ))]
        params.append(layer)
    return params

key = jax.random.PRNGKey(seed=seed)
MLP_params = init_MLP([784, 512, 128, 10], key)
print(jax.tree.map(lambda x:x.shape, MLP_params))

        
#------------------------------------------
start_new_block("Add predict fn:")

def MLP_predict(params, x):
    
    *hidden_layers, last = params
    
    for w, b in hidden_layers:
        x = jnp.dot(w, x) + b
        x = jax.nn.relu(x)
    
    last_w, last_b = last
    logits = jnp.dot(last_w, x) + last_b
    
    # logits: z_i = log(exp(z_i))
    # logsumexp(logits): log(sum_i(exp(z_i)))
    # logits - logsumexp(logits) = log(exp(z_i)/sum_i(exp(z_i)))
    # this is actually softmax
    return logits - logsumexp(logits)

mnist_image_size= [28, 28]
print('mnist image size:\n:', mnist_image_size)
dummy_img_flat = np.random.randn(np.prod(mnist_image_size))
print(dummy_img_flat.shape)
prediction = MLP_predict(MLP_params, dummy_img_flat)
print("prediction:\n", prediction)

#------------------------------------------
start_new_block("Bached MLP predict:")

batched_MLP_predict = jax.vmap(MLP_predict, in_axes=(None, 0))
# small test
dummy_imgs_flat = np.random.randn(16, np.prod(mnist_image_size))
print("batched image shape:", dummy_imgs_flat.shape)
predictions = batched_MLP_predict(MLP_params, dummy_imgs_flat)
print("batched prediction shape:", predictions.shape) 

#------------------------------------------
start_new_block("Add dataloarder in pytorch:")

def custom_transform(x): 
    return np.ravel(np.array(x, dtype=np.float32))
train_dataset = datasets.MNIST(root = './train_mnist', 
                               train = True, 
                               download=True,
                               transform=custom_transform)
test_dataset = datasets.MNIST(root = './train_mnist', 
                               train = False, 
                               download=True,
                               transform=custom_transform)

print('type of dataset: ', type(train_dataset))
print('current work directory: ', os.getcwd())
something = train_dataset[0][0]
print(type(something), something.shape)

#------------------------------------------
start_new_block("Create DataLoader in Pytorch:")
def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))
    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])
    return imgs, labels

train_dataloader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True,
                              collate_fn=custom_collate_fn,
                              drop_last=True)

test_dataloader = DataLoader(test_dataset,
                              batch_size=128,
                              shuffle=False,
                              collate_fn=custom_collate_fn,
                              drop_last=True)

batched_data = next(iter(train_dataloader))
imgs = batched_data[0]
labels = batched_data[1]
print("batched imgs shape: ", imgs.shape)
print("batched labels shape: ", labels.shape)

#------------------------------------------
start_new_block("Add train and test fn:")

lr = 1e-3
n_epoches = 1

def loss_fn(params, imgs, lbls): # lbls: one hot
    predictions = batched_MLP_predict(params, imgs) # (b,10)
    return -jnp.mean(predictions*lbls)

def update(params, imgs, lbls):
    loss, grads = jax.value_and_grad(loss_fn)(params, imgs, lbls)
    return loss, jax.tree.map(lambda p,g: p-lr*g, params, grads)

def accuracy(params, loader):
    acc_sum = 0
    for img, lbs in loader:
        class_pred = jnp.argmax(batched_MLP_predict(params, imgs), axis=1)
        acc_sum += np.sum(class_pred == lbs)
    return acc_sum/ (len(loader) * loader.batch_size)

for epoch in range(n_epoches):
    print(f"Epoch {epoch}, train acc = {accuracy(MLP_params, train_dataloader)}")
    print(f"Epoch {epoch}, test acc =  {accuracy(MLP_params, test_dataloader)}")
    for cnt, (imgs, lbls) in enumerate(train_dataloader):
        lbls = jax.nn.one_hot(lbls, len(datasets.MNIST.classes))
        loss, MLP_params = update(MLP_params, imgs, lbls)
        if cnt % 100 == 0:
            print(f"Epoch: {epoch}, Loss {loss}")

#------------------------------------------
start_new_block("Improve speed, via passing dataset to device memory directly:")
train_imgs  = jnp.array(train_dataset.data.reshape(len(train_dataset), -1))
train_lbls  = jnp.array(train_dataset.targets)

test_imgs  = jnp.array(test_dataset.data.reshape(len(test_dataset), -1))
test_lbls  = jnp.array(test_dataset.targets)

print(train_imgs.shape, train_lbls.shape)

def accuracy(params, dataset_imgs, dataset_lbls):
    class_pred = jnp.argmax(batched_MLP_predict(params, dataset_imgs), axis=1)
    return jnp.mean(dataset_lbls == class_pred)

MLP_params = init_MLP([784, 512, 128, 64, 10], parent_key=key)
n_epoches = 10

for epoch in range(n_epoches):
    for cnt, (imgs, lbls) in enumerate(train_dataloader):
        lbls = jax.nn.one_hot(lbls, len(datasets.MNIST.classes))
        loss, MLP_params = update(MLP_params, imgs, lbls)
        if cnt % 100 == 0: 
            print(f"Epoch {epoch}, Step {cnt}, Loss {loss}")
    print(f"Epoch {epoch}, train acc = {accuracy(MLP_params, train_imgs, train_lbls)}")
    print(f"Epoch {epoch}, train acc = {accuracy(MLP_params, test_imgs, test_lbls)}")
    
#------------------------------------------
start_new_block("Visualize the results:")
def visualize_sample(loader):
    imgs, lbls = next(iter(loader))
    img = imgs[0].reshape(mnist_image_size)
    pred = jnp.argmax(MLP_predict(MLP_params, imgs[0]))
    print("label:", lbls[0])
    print('pred:', pred)
    plt.imshow(img)
    plt.show()

visualize_sample(test_dataloader)

#------------------------------------------
start_new_block("Visualize the weight matrix ")
w = MLP_params[0][0]
print(w.shape)

w_single = w[0,:].reshape(mnist_image_size)
print(w_single.shape)
plt.imshow(w_single)
plt.show()

#------------------------------------------
start_new_block("Visualize embeddings via t-SNE")
from sklearn.manifold import TSNE
def fetch_activations(params, x):
    *hidden, last = params
    for w, b in hidden:
        x = jax.nn.relu(jnp.dot(w, x)) + b
    return x

batched_fetch_activations = jax.vmap(fetch_activations, in_axes=(None, 0))
imgs, lbls = next(iter(test_dataloader))
batch_activations = batched_fetch_activations(MLP_params, imgs)
print(batch_activations[1].shape)

t_sne_embeddings = TSNE(n_components=2, perplexity=30,).fit_transform(batch_activations)
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}

#for class_id in range(10):
#    plt.scatter(t_sne_embeddings[lbls == class_id, 0], t_sne_embeddings[lbls == class_id, 1], s=20, color=cora_label_to_color_map[class_id])
# plt.show()

#------------------------------------------
start_new_block("Find dead neurons:")

def fetch_activations2(params, x):
    *hidden, last = params
    collector = []
    for w, b in hidden:
        x = jax.nn.relu(jnp.dot(w, x)) + b
        collector.append(x)
    return collector

batched_fetch_activations2 = jax.vmap(fetch_activations2, in_axes=(None, 0))
imgs, lbls = next(iter(test_dataloader))
batch_activations = batched_fetch_activations2(MLP_params, imgs)
print(batch_activations[1].shape)

dead_neurons = [np.ones(act.shape[1:]) for act in batch_activations]

for layer_id, activations in enumerate(batch_activations):
    dead_neurons[layer_id] = np.logical_and(dead_neurons[layer_id], (activations==0).all(axis=0))

for li, layers in enumerate(dead_neurons):
    print(f"In layer {li}, there are {np.sum(layers)} dead neurons.")


