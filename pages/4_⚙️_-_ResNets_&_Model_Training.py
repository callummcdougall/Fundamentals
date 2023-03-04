import os
if not os.path.exists("images"):
    os.chdir("../")
from st_dependencies import *
styling()

def section_home():
    st.markdown(r"""
Links to Colab: [**exercises**](https://colab.research.google.com/drive/1N1Cu13q4dk2Z0qYgdy7Cnb6ESAlOu5ge?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1obMRz1Y9iXrJbQBXaYCBS61S-mxOIhWO?usp=sharing).
""")
    st_image("resnet.png", 350)
    st.markdown(r"""
# ResNets & Model Training

## 1️⃣ Building & training a CNN

In part 1, we'll use the modules that we defined in previous exercises to build a basic CNN to classify MNIST images. We'll understand the basics of `Datasets` and `DataLoaders`, see how a basic training loop works, and also measure our model's accuracy on a test set.

This section should take **2-3 hours**.

## 2️⃣ Assembling ResNet

In part 2, we'll start by defining a few more important modules (e.g. `BatchNorm2d` and `Sequential`), building on our work from yesterday. Then we'll build a much more complex architecture - a **residual neural network**, which uses a special type of connection called **skip connections**. 

This section should take approximately **2-3 hours**.

---

Today's exercises are probably the most directly relevant for the rest of the programme out of everything we've done this week. This is because we'll be looking at important concepts like training loops and neural network architectures. Additionally, the task of assembling a complicated neural network architecture from a set of instructions will lead straight into next week, when we'll be building our own transformers! So forming a deep understanding of everything that's going on in today's exercises will be very helpful going forwards.

""")
# ## 3️⃣ Finetuning ResNet (bonus)

# If you get to part 3, this is an opportunity to try finetuning your ResNet. Finetuning involves taking a pretrained model, and training it to perform a slightly different task (sometimes altering the architecture at the end of the model, and only training that part while freezing the rest). We've given much less guidance for this section, since it builds on all the previous sections.

# This section may take quite a long time to finish, and you're encouraged to go further with it over the next couple of days if it seems exciting to you.
 
def section_cnn():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#convnet">ConvNet</a></li>
    <li><a class="contents-el" href="#transforms">Transforms</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#aside-tqdm">Aside - <code>tqdm</code></a></li>
        <li><a class="contents-el" href="#aside-device">Aside - <code>device</code></a></li>
    </li></ul>
    <li><a class="contents-el" href="#training-loop">Training loop</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#aside-dataclasses">Aside - <code>dataclasses</code></a></li>
        <li><a class="contents-el" href="#other-notes-on-this-code">Other notes on this code</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""

# Building & Training a CNN

## Imports

```python
import torch as t
from torch import nn
from einops import rearrange
from dataclasses import dataclass
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import List, Callable, Tuple
from PIL import Image
import plotly.express as px
from IPython.display import display
import torchinfo
import json

from part2_cnns_solutions import ReLU, Conv2d, MaxPool2d, Flatten, Linear

import part4_resnets_utils as utils
import part4_resnets_tests as tests
```

## ConvNet

We'll be attempting to build the following neural network architecture:
""")

    st_image('mnist_diagram.png', width=750)

    st.markdown(r"""
Let's briefly discuss this architecture. We see that it starts with two consecutive stacks of:

* 2D convolution,
* ReLU activation function,
* 2x2 Max pooling

Combining these three in this order (or more generally, convolution + activation function + max pooling) is pretty common, and we'll see it in many more of the architectures we look at going forwards.

Then, we use a `Flatten` (recall the question from yesterday's exercises - we only use `Flatten` after all of our convolutions, because it destroys spatial dependence). Finally, we apply two linear layers to get our output. 

Our network is doing MNIST classification, so this output should represent (in some sense) the strength of our evidence that the input is some particular digit. We can get a prediction by taking the max across this output layer.

We can also represent this network using a diagram:
""")

    st.write(r"""<figure style="max-width:150px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNp9kUFrwzAMhf-K8bmF1RkjhNFLtsEg60rKTnEPaqw2BscOjj0ySv_77GSFlI35IJ7QJz14PtPaCKQZPVnoGlKUXPf-MDWc5psNp1yT8F5151011seDXbN0YOmeLJdrkhv9WSVDMoo4SxipG9AaVR_bLQgh9YmsJrxE5asSi4-pbWHojFEVGxh5g2EbdFzaOSsFEjZBdbjMbjwe7kn-l8dsd-bHfhsm_zu-KHAO9agLqRGrWMFGcsVSYrwLSfT7KzAf391O370jXKMWdEFbtC1IEfI-x1g5dQ22yGkWpMAjeOVi4JeA-k6Aw2chnbE0O4LqcUHBO7P70jXNnPV4hZ4khB9rf6jLN9JgmcE" /></figure>""", unsafe_allow_html=True)


    st.markdown(r"""
which is something we'll be using a lot in future exercises, as we deal with more complicated architectures with hierarchically nested components.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
### Exercise - creating `ConvNet`

Although you're creating a neural network rather than a single layer, this is structurally very similar to the exercises at the end of yesterday when you created `nn.Module` objects to wrap around functions. This time, you're creating an `nn.Module` object to contain the modules of the network. 

Below `__init__`, you should define all of your modules. It's conventional to number them, e.g. `self.conv1 = Conv2d(...)` and `self.linear1 = Linear(...)` (or another common convention is `fc`, for "fully connected"). Below `forward`, you should return the value of sequentially applying all these layers to the input.

```python
class ConvNet(nn.Module):
    def __init__(self):
        pass
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


if MAIN:
    model = ConvNet()
    print(model)
```

Note - rather than defining your network this way, it would be possible to just wrap everything inside an `nn.Sequential`. For simple examples like this, both ways work just fine. However, for more complicated architectures involving nested components and multiple different branches of computation (e.g. the ResNet we'll be building later today), there will be major advantages to building your network in this way.
""")

        with st.expander("Help - I'm not sure where to start."):
            st.markdown(r"""As an example, the first thing you should define in the initialisation section is:
    
```python
self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
```

Then in the forward loop, the first thing that you call on your input should be:

```python
x = self.conv1(x)
```

After this, it's just a matter of repeating these steps for all the other layers in your model.
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=3136, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=10)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.fc2(self.fc1(self.flatten(x)))
        return x
```
""")
        st.markdown(r"""
We can also use the useful library `torchinfo` to print out a much more informative description of our model. You can use this function in two ways:

1. Specify `input_size` (remember the batch dimension!), and optionally `dtypes` (this defaults to float).
2. Specify a sample input tensor, via the argument `input_data`.

Below is an example of the first one:

```python
if MAIN:
    summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
    print(summary)
```
""")

    st.markdown(r"""
## Transforms

Before we use this model to make any predictions, we first need to think about our input data. Below is a block of code to fetch and process MNIST data. We will go through it line by line.

```python
def get_mnist(subset: int = 10):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=mnist_transform)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset

if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=True)
```

The `torchvision` package consists of popular datasets, model architectures, and common image transformations for computer vision. `transforms` is a library from `torchvision` which provides access to a suite of functions for preprocessing data. The three functions used here are:

1. `transforms.Compose`, for stringing together multiple transforms into a sequence.
2. `transforms.ToTensor`, for converting a PIL image or numpy array into a tensor. In this case, our data are initially stored as PIL images (you can check this by inspecting the contents of `mnist_trainset` when you omit the `transform` argument).
3. `transforms.Normalize`, for applying linear transformations to an image. Here, we're subtracting 0.1307 and dividing by 0.3081 for each image.

Note that each transform is applied individually, to each element in the dataset. Other kinds of transforms include `RandomCrop` for cropping to a certain size at random positions, `GaussianBlur` for blurring an image, and even `Lambda` which allows the users to convert their own functions into torchvision transform objects.

---

Next, we define `mnist_trainset`. We use `datasets.MNIST` to get our data (further details of this database can be found [here](http://yann.lecun.com/exdb/mnist/)). The argument `root="./data"` indicates that we're storing our data in the `./data` directory, and `transform=transform` tells us that we should apply our previously defined `transform` to each element in our data.

Print `dir(mnist_trainset)` to see all the attributes of your dataset. Not all datasets have exactly the same attributes, but there will usually be some common ones. In this example, you should try running the following to see what they do:

* `mnist_trainset.data`
* `mnist_trainset.targets`
* `mnist_trainset.classes`
* `mnist_trainset[0]` (i.e. just indexing the dataset)

The `Sample` function allows us to take a sample of a dataset (useful when our training loops are taking a very long time!). The argument `indices` is a list of indices to sample from the dataset. For example, `Sample(mnist_trainset, indices=[0, 1, 2])` will return a dataset containing only the first three elements of `mnist_trainset`.

---

Finally, `DataLoader` provides a useful abstraction to work with a dataset. It takes in a dataset, and a few arguments including `batch_size` (how many inputs to feed through the model on which to compute the loss before each step of gradient descent) and `shuffle` (whether to randomise the order each time you iterate). The object that it returns can be iterated through as follows:

```python
for X_batch, y_batch in mnist_trainloader:
    ...
```

where `X_batch` is a 3D array of shape `(batch_size, 28, 28)` where each slice is an image, and `y_batch` is a 1D tensor of labels of length `batch_size`. This is much faster than what we'd be forced to do with `mnist_trainset`:

```python
for i in range(len(mnist_trainset) // batch_size):
    
    X_batch = mnist_trainset.data[i*batch_size: (i+1)*batch_size]
    y_batch = mnist_trainset.targets[i*batch_size: (i+1)*batch_size]

    ...
```

A note about batch size - it's common to see batch sizes which are powers of two. The motivation is for efficient GPU utilisation, since processor architectures are normally organised around powers of 2, and computational efficiency is often increased by having the items in each batch split across processors. Or at least, that's the idea. The truth is a bit more complicated, and some studies dispute whether it actually saves time. We'll dive much deeper into these kinds of topics during the week on training at scale.

---

You should play around with the objects defined above (i.e. `mnist_trainset` and `mnist_trainloader`) until you have a good feel for how they work. You should also answer the questions below before proceeding.
""")

    with st.expander(r"Question - can you explain why we include a data normalization function in torchvision.transforms?"):
        st.markdown(r"""One consequence of unnormalized data is that you might find yourself stuck in a very flat region of the domain, and gradient descent may take much longer to converge.""")
    
        st.info(r"""Normalization isn't strictly necessary for this reason, because any rescaling of an input vector can be effectively undone by the network learning different weights and biases. But in practice, it does usually help speed up convergence.""")

        st.markdown(r"""Normalization also helps avoid numerical issues.""")

    with st.expander(r"""Question - can you explain why we use these exact values to normalize with?"""):
        st.markdown(r"""These values were calculated across the MNIST dataset, so that they would have approximately mean 0 and variance 1.""")

    with st.expander(r"""Question - if the dataset was of full-color images, what would the shape of trainset.data be? How about trainset.targets?"""):
        st.markdown(r"""`trainset.data` would have shape `(dataset_size, channels, height, width)` where `channels=3` represents the RGB channels. `trainset.targets` would still just be a 1D array.""")

    with st.expander(r"""Question - what is the benefit of using shuffle=True? i.e. what might the problem be if we didn't do this?"""):
        st.markdown(r"""Shuffling is done during the training to make sure we aren't exposing our model to the same cycle (order) of data in every epoch. It is basically done to ensure the model isn't adapting its learning to any kind of spurious pattern.""")

    st.markdown(r"""
### Aside - `tqdm`

You might have seen some blue progress bars running when you first downloaded your MNIST data. These were generated using a library called `tqdm`, which is also a really useful tool when training models or running any process that takes a long period of time. 

You can run the cell below to see how these progress bars are used (note that you might need to install the `tqdm` library first).

```python
from tqdm import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.01)
```

`tqdm` wraps around a list, range or other iterable, but other than that it doesn't affect the structure of your loop. You can also run multiple nested progress bars, if you add the argument `leave=False` in the inner progress bar:

```python
for j in tqdm(range(5)):
    for i in tqdm(range(100), leave=False):
        time.sleep(0.01)
```

You can also update the description of a progress bar, e.g. to print out the loss in the middle of a training loop. This can be a very handy way to see if the model is actually improving, and how quickly. A template for how you might do this during training:

```python
progress_bar = tqdm(dataloader)
for (x, y) in dataloader:
    # calculate training loss
    progress_bar.set_description(f"Training loss = {loss}")
```

One gotcha when it comes to `tqdm` - if you use it to wrap around an object with no well-defined length (e.g. an enumerator), it won't know what the progress bar total is. For instance, you can run this cell to verify it won't work as intended:

```python
for i in tqdm(enumerate(range(100))):
    time.sleep(0.01)
```

You can fix this by putting the `enumerate` outside of the `tqdm` function, or by adding the argument `total=100` (this tells `tqdm` exactly how many objects there are to iterate through).
""")

    st.markdown(r"""
### Aside - `device`

One last thing to discuss before we move onto training our model: **GPUs**. We'll discuss this in a little more detail in the next set of exercises (Training & Optimization). For now, [this page](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk) should provide a basic overview of how to use your GPU. A few things to be aware of here:

* The `to` method is really useful here - it can move objects between different devices (i.e. CPU and GPU) *as well as* changing a tensor's datatype.
    * Note that `to` is never inplace for tensors (i.e. you have to call `x = x.to(device)`), but when working with models, calling `model = model.to(device)` or `model.to(device)` are both perfectly valid.
* Errors from having one device on cpu and another on cuda are very common. Some useful practices to avoid this:
    * Throw in assert statements, to make sure tensors are on the same advice
    * Remember that when you initialise an array (e.g. with `t.zeros` or `t.arange`), it will be on CPU by default.
    * Tensor methods like [`new_zeros`](https://pytorch.org/docs/stable/generated/torch.Tensor.new_zeros.html) or [`new_full`](https://pytorch.org/docs/stable/generated/torch.Tensor.new_full.html) are useful, because they'll create tensors which match the device and dtype of the base tensor.

It's common practice to put a line like this at the top of your file, defining a global variable which you can use in subsequent modules and functions (excluding the print statement):

```python
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
```

## Training loop

Finally, we'll now build our training loop. The one below is actually constructed for you rather than left as an exercise, but you should make sure that you understand the purpose of every line below, because soon you'll be adding to it, and making your own training loops for different architectures.

```python
@dataclass
class ConvNetTrainingArgs():
    trainset: datasets.VisionDataset
    testset: datasets.VisionDataset
    epochs: int = 3
    batch_size: int = 512
    loss_fn: Callable[..., t.Tensor] = nn.CrossEntropyLoss()
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_args: Tuple = ()
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "models/part4_convnet.pt"


def train_convnet(args: ConvNetTrainingArgs) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns loss_list, the list of cross entropy losses computed on each batch.
    '''
    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)

    model = ConvNet().to(args.device).train()
    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)
    loss_list = []
    
    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (imgs, labels) in progress_bar:
            
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            
            probs = model(imgs)
            loss = args.loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return loss_list


if MAIN:
    args = ConvNetTrainingArgs(mnist_trainset, mnist_testset, epochs=2)
    loss_list = train_convnet(args)

    px.line(
        y=loss_list, x=range(0, len(loss_list)*args.batch_size, args.batch_size),
        labels={"y": "Cross entropy loss", "x": "Num images seen"}, 
        title="MNIST training curve (cross entropy loss)", template="ggplot2"
    ).update_layout(
        showlegend=False, yaxis_range=[0, max(loss_list)*1.1], height=400, width=600
    ).show()
```

You should find the results to be much better than the results we got from our earlier models. It might take longer to get to the same loss, but it will eventually get much lower than the previous model would be able to.

---

We've seen most of these components before over the last few days. The most important lines to go over are these five:

```python
y_hat = model(x)
loss = args.loss_fn(y_hat, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

An explanation of these five lines:

* `y_hat = model(x)` calculates our output of our network.
* `loss = loss_fn(y_hat, y)` computes the loss (in this case, the cross entropy loss) beween our output and the true label.
    * The first argument of this loss function are the unnormalised scores; these are converted to probabilities by applying the [softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) function, then their cross entropy is measured with `y`.
    * Here, the second argument `y` is just a single label, but this is interpreted as a probability distribution where all values are `0` or `1` (e.g. `2` is interpreted as the distribution `(0, 0, 1, 0, ..., 0)`). This loss function can accept either labels or probability distributions in the second argument.
    * If you're interested in the intuition behind cross entropy as a loss function, see [this post on KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) (note that KL divergence and cross entropy differ by a value which is independent of our model's predictions, so cross entropy minimisation and KL divergence minimisation are equivalent).
* `loss.backward()` is how we propagate the gradients backwards through our network for, to perform gradient descent.
* `optimizer.step()` updates the gradients.
* `optimizer.zero_grad()` makes sure that the gradients of all `model.parameters()` are zero, i.e. they don't just keep accumulating from previous inputs. (Note, sometimes people put this line at the start of the loop rather than the end, this is also valid.)

### Aside - `dataclasses`

Using dataclasses for creating or training models is a very useful way to keep your code clean. Dataclasses are a special kind of class which come with built-in methods for initialising and printing (i.e. no need to define an `__init__` or `__repr__`). You can set specific arguments as follows:

```python
args = ConvNetTrainingArgs(mnist_trainset, mnist_testset, epochs=10, batch_size=256)
print(args)
```

Using dataclasses, saves code, and keeps all your arguments in one easy-to-track place. Also, if you're using code autocomplete on VSCode, just typing `args.` will let you tab in the appropriate argument, which is pretty handy!

### Other notes on this code

* The `train()` method used when defining models changes the behaviour of certain types of layers, e.g. batch norm and dropout. We don't have either of these types of layers present in our model, but we'll need to use this later today when we work with ResNets.
* We used `torch.optim.Adam` as an optimiser. We'll discuss optimisers in more detail in the next set of exercises, but for now it's enough to know that Adam is a gradient descent optimisation algorithm which empirically performs much better than simpler algorithms like SGD (stochastic gradient descent).
* The `.item()` method can be used on one-element tensors, and returns their value as a standard Python int / float.
* The `torch.save` function takes in a model and a filename, and saves that model (including the values of all its parameters). Common filename extensions for PyTorch models are `.pt` and `.pth`.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
### Exercise - add a testing loop

Edit the `train_convnet` function so that, at the end of each epoch, it returns the accuracy of your model on a test set.

Below, we've also included a function which plots the loss and test set accuracy on the same graph, just to help you verify that your function is behaving as expected.

```python
def train_convnet(args: ConvNetTrainingArgs) -> Tuple[list, list]:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''
    pass


if MAIN:
    args = ConvNetTrainingArgs(mnist_trainset, mnist_testset)
    loss_list, accuracy_list = train_convnet(args)

    px.line(
        y=loss_list, x=range(0, len(loss_list)*args.batch_size, args.batch_size),
        labels={"y": "Cross entropy loss", "x": "Num images seen"}, 
        title="MNIST training curve (cross entropy loss)", template="ggplot2"
    ).update_layout(
        showlegend=False, yaxis_range=[0, max(loss_list)*1.1], height=400, width=600
    ).show()

    px.line(
        y=accuracy_list, x=range(1, len(accuracy_list)+1),
        title="Training accuracy for CNN, on MNIST data",
        labels={"x": "Epoch", "y": "Accuracy"}, template="seaborn", 
        height=400, width=600
    ).show()
```
""")

        with st.expander(r"""Help - I get 'RuntimeError: expected scalar type Float but found Byte'."""):
            st.markdown(r"""
This is commonly because one of your operations is between tensors with the wrong datatypes (e.g. `int` and `float`). Try navigating to the error line and checking your dtypes (or using VSCode's built-in debugger).
""")

        with st.expander(r"""Help - I'm not sure how to calculate accuracy."""):
            st.markdown(r"""
Recall that your model's outputs are a tensor of shape `(batch_size, 10 = num_classifications)`. You can get your model's predictions using `y_predictions = y_hat.argmax(1)` (which means taking the argmax along the 1st dimension). Then, `(y_predictions == y)` will be a boolean array, and the accuracy for this epoch will be equal to the fraction of elements of this array that are `True`.
""")
        with st.expander("Solution (one possible implementation)"):
            st.markdown(r"""
```python
def train_convnet(args: ConvNetTrainingArgs) -> Tuple[list, list]:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    
    Returns tuple of (loss_list, accuracy_list), where accuracy_list contains the fraction of accurate classifications on the test set, at the end of each epoch.
    '''

    trainloader = DataLoader(args.trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(args.testset, batch_size=args.batch_size, shuffle=True)

    model = ConvNet().to(args.device).train()
    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)
    loss_list = []
    accuracy_list = []
    
    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (imgs, labels) in progress_bar:
            
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            
            probs = model(imgs)
            loss = args.loss_fn(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")
        
        with t.inference_mode():
            
            accuracy = 0
            total = 0
            
            for (imgs, labels) in testloader:

                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                probs = model(imgs)
                predictions = probs.argmax(-1)
                accuracy += (predictions == labels).sum().item()
                total += imgs.size(0)

            accuracy_list.append(accuracy/total)
            
        print(f"Train loss = {loss:.6f}, Accuracy = {accuracy}/{total}")
    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return loss_list, accuracy_list
```
""")
 
def section_resnet():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#some-final-modules">Some final modules</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#nn-sequential">Sequential</a></li>
        <li><a class="contents-el" href="#nn-batchnorm2d">BatchNorm2d</a></li>
        <li><ul class="contents">
            <li><a class="contents-el" href="#train-and-eval-modes">Train and Eval Modes</a></li>
        </li></ul>
        <li><a class="contents-el" href="#nn-averagepool">AveragePool</a></li>
    </li></ul>
    <li><a class="contents-el" href="#building-resnet">Building ResNet</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#residual-block">Residual Block</a></li>
        <li><a class="contents-el" href="#blockgroup">BlockGroup</a></li>
        <li><a class="contents-el" href="#resnet34">ResNet34</a></li>
    </li></ul>
    <li><a class="contents-el" href="#running-your-model">Running Your Model</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#aside-hooks">Aside - hooks</a></li>
    </li></ul>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Assembling ResNet

## Reading

* [Batch Normalization in Convolutional Neural Networks](https://www.baeldung.com/cs/batch-normalization-cnn)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

You should move on once you can answer the following questions:
""")

    with st.expander(r""""Batch Normalization allows us to be less careful about initialization." Explain this statement."""):
        st.markdown(r"""Weight initialisation methods like Xavier (which we encountered yesterday) are based on the idea of making sure the activations have approximately the same distribution across layers at initialisation. But batch normalisation ensures that this is the case as signals pass through the network.""")

    with st.expander(r"""Give three reasons why batch norm improves the performance of neural networks."""):
        st.markdown(r"""The reasons given in the first linked document above are:
    * Normalising inputs speeds up computation
* Internal covariate shift is reduced, i.e. the mean and standard deviation is kept constant across the layers.
* Regularisation effect: noise internal to each minibatch is reduced""")

    with st.expander(r"""If you have an input tensor of size (batch, channels, width, height), and you apply a batchnorm layer, how many learned parameters will there be?"""):
        st.markdown(r"""A mean and standard deviation is calculated for each channel (i.e. each calculation is done across the batch, width, and height dimensions). So the number of learned params will be `2 * channels`.""")

# with st.expander(r"""Review section 1 of the "Deep Residual Learning" paper, then answer the following question in your own words: why are ResNets a natural solution to the degradation problem?"""):
#     st.markdown(r"""**Degradation problem** = increasing the depth of a network leads to worse performance not only on the test set, but *also the training set* (indicating that this isn't just caused by overfitting). One could argue that the deep network has the capacity to learn the shallower network as a "subnetwork" within itself; if it just sets most of its layers to the identity. However, empirically it seems to have trouble doing this. Skip connections in ResNets are a natural way to fix this problem, because they essentially hardcode an identity mapping into the network, rather than making the network learn the identity mapping.""")

    with st.expander(r"""In the paper, the diagram shows additive skip connections (i.e. F(x) + x). One can also form concatenated skip connections, by "gluing together" F(x) and x into a single tensor. Give one advantage and one disadvantage of these, relative to additive connections."""):
        st.markdown(r"""One advantage of concatenation: the subsequent layers can re-use middle representations; maintaining more information which can lead to better performance. Also, this still works if the tensors aren't exactly the same shape. One disadvantage: less compact, so there may be more weights to learn in subsequent layers.

Crucially, both the addition and concatenation methods have the property of preserving information, to at least some degree of fidelity. For instance, you can [use calculus to show](https://theaisummer.com/skip-connections/#:~:text=residual%20skip%20connections.-,ResNet%3A%20skip%20connections%C2%A0via%C2%A0addition,-The%20core%20idea) that both methods will fix the vanishing gradients problem.""")

    st.markdown(r"""In this section, we'll do a more advanced version of the exercise in part 1. Rather than building a relatively simple network in which computation can be easily represented by a sequence of simple layers, we're going to build a more complex architecture which requires us to define nested blocks.

## Some final modules

We'll start by defining a few more `nn.Module` objects, which we hadn't needed before.

### Sequential

Firstly, now that we're working with large and complex architectures, we should create a version of `nn.Sequential`. Recall that we briefly came across `nn.Sequential` at the end of the first day, when building our (extremely simple) neural network. As the name suggests, when an `nn.Sequential` is fed an input, it sequentially applies each of its submodules to the input, with the output from one module feeding into the next one.

The implementation is given to you below. A few notes:

* `self.add_module` is called on each provided module.
    * This adds each one to the dictionary `self._modules` in the base class, which means they'll be included in `self.parameters()` as desired.
    * It also gives each one a unique (within this `Sequential`) name, so that they don't override each other.
* The `repr` of the base class `nn.Module` already recursively prints out the submodules, so we don't need to write anything in `extra_repr`.
    * To see how this works in practice, try defining a `Sequential` which takes a sequence of modules that you've defined above, and see what it looks like when you print it.

Make sure you understand what's going on here, before moving on.

```python
class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x
```

### BatchNorm2d

Now, we'll implement our `BatchNorml2d`, the layer described in the documents you hopefully read above.

Something which might have occurred to you as you read about batch norm - how does it work when in inference mode? It makes sense to normalize over a batch of multiple input data, but normalizing over a single datapoint doesn't make any sense! This is why we have to introduce a new PyTorch concept: **buffers**.

Unlike `nn.Parameter`, a buffer is not its own type and does not wrap a `Tensor`. A buffer is just a regular `Tensor` on which you've called [self.register_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) from inside a `nn.Module`.

The reason we have a different name for this is to describe how it is treated by various machinery within PyTorch.

* It is normally included in the output of `module.state_dict()`, meaning that `torch.save` and `torch.load` will serialize and deserialize it.
* It is moved between devices when you call `model.to(device)`.
* It is not included in `module.parameters`, so optimizers won't see or modify it. Instead, your module will modify it as appropriate within `forward`.

#### Train and Eval Modes

This is your first implementation that needs to care about the value of `self.training`, which is set to True by default, and can be set to False by `self.eval()` or to True by `self.train()`.

In training mode, you should use the mean and variance of the batch you're on, but you should also update a stored `running_mean` and `running_var` on each call to `forward` using the "momentum" argument as described in the PyTorch docs. Your `running_mean` shuld be intialized as all zeros; your `running_var` should be initialized as all ones.

In eval mode, you should use the running mean and variance that you stored before (not the mean and variance from the current batch).
""")

    with st.columns(1)[0]:
        st.markdown(r"""
### Exercise - implement `BatchNorm2d`

Implement `BatchNorm2d` according to the [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html). Call your learnable parameters `weight` and `bias` for consistency with PyTorch.

```python
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        pass

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = rearrange(self.running_var, "channels -> 1 channels 1 1")
        
        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = rearrange(self.bias, "channels -> 1 channels 1 1")
        
        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])
```
""")
    st.markdown(r"""
### AveragePool

Let's end our collection of `nn.Module`s with an easy one 🙂

The ResNet has a Linear layer with 1000 outputs at the end in order to produce classification logits for each of the 1000 classes. Any Linear needs to have a constant number of input features, but the ResNet is supposed to be compatible with arbitrary height and width, so we can't just do a pooling operation with a fixed kernel size and stride.

Luckily, the simplest possible solution works decently: take the mean over the spatial dimensions. Intuitively, each position has an equal "vote" for what objects it can "see".
""")
    with st.columns(1)[0]:
        st.markdown(r"""
### Exercise - implement `AveragePool`

This should be a pretty straightforward implementation; it doesn't have any weights or parameters of any kind, so you only need to implement the `forward` method.

```python
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        pass
```
""")
        with st.expander(r"Solution"):
            st.markdown(r"""
```python
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, dim=(2, 3))
```
""")
    st.markdown(r"""
## Building ResNet

Now we have all the building blocks we need to start assembling your own ResNet! The following diagram describes the architecture of ResNet34 - the other versions are broadly similar. Unless otherwise noted, convolutions have a kernel_size of 3x3, a stride of 1, and a padding of 1. None of the convolutions have biases. 
""")

    with st.expander(r"""Question: there would be no advantage to enabling biases on the convolutional layers. Why?"""):
        st.markdown(r"""Every convolution layer in this network is followed by a batch normalization layer. The first operation in the batch normalization layer is to subtract the mean of each output channel. But a convolutional bias just adds some scalar `b` to each output channel, increasing the mean by `b`. This means that for any `b` added, the batch normalization will subtract `b` to exactly negate the bias term.""")

    with st.expander(r"""Question: why is it necessary for the output of the left and right computational tracks in ResidualBlock to be the same shape?"""):
        st.markdown(r"""Because they're added together at the end of the tracks. If they weren't the same shape, then they couldn't be added together.""")

    with st.expander(r"""Help - I'm confused about how the nested subgraphs work."""):
        st.markdown(r"""The right-most block in the diagram, `ResidualBlock`, is nested inside `BlockGroup` multiple times. When you see `ResidualBlock` in `BlockGroup`, you should visualise a copy of `ResidualBlock` sitting in that position. 
    
Similarly, `BlockGroup` is nested multiple times (four to be precise) in the full `ResNet34` architecture.""")


    st.write(r"""<figure style="max-width:900px"><embed type="image/svg+xml" src="https://mermaid.ink/svg/pako:eNqNU8FuozAQ_RXLZ5pCUm2laBUpLmo2UpZUlGgPJgcXTxu0YCNjV6mq_vvaODQlqbSxwAzPb2b8xuN3XEgOeIpfFGt2KItzgexozZMHcmwfjw3wFNoE9OTmuOQGLxUUupQCZWS4shSN0bSbfz6p2RZdXc0seCfFK73d3yJnuIUfN6jYMSGgagPUalVyQOOeTRJKmC52iVR1j6Ww2lA39cCDlBWd7CfoN9s720V9HAQilSz-LpQ0TUu_2I4YBigK0Gg0ClDiyfNXUOwFXKTu_75iWoPo7FUpgCnqP9b9ehaFYYik0VZm6_3XRh8rAYLn4ptqHndxaT3JYil8OX2aOCW2Cm3JDau6aE7Nn2X2C8nGBWAVapg6kFMSDsln2nvv9Sb7LgBZWFnUvtv_axskuljeUB3p-sSfIu965YCTJDptCdcLR6cxHbDHp-w5572XGX_po5ScKzzZoK97lyWm64dsuU7mK1e4aB-h873GJP7MPaCf7-ikqAcDB7gGVbOS2-v67uAc6x3UkOOpNTk8M1Npd1s_LJUZLR_fRIGnWhkIsGk40xCXzJ5KjafPrGrh4x-ZkTKg" /></figure>""", unsafe_allow_html=True)
    # graph TD
    # subgraph " "
    #     subgraph ResNet34
    #         direction TB
    #         Input[Input<br>] --> InConv[7x7 Conv<br>64 channels, stride 2] --> InBN[BatchNorm] --> InReLU[ReLU] --> InPool[3x3 MaxPool<br>Stride 2] --> BlockGroups[BlockGroups<br>0, 1, ..., N] --> AveragePool --> Flatten --> Linear[Linear<br/>1000 outputs] --> Out
    #     end

    #     subgraph BlockGroup
    #         direction TB
    #         BGIn[Input] --> DRB[ResidualBlock<br>WITH optional part] --> RB0[ResidualBlocks<br>0, 1, ..., N<br>WITHOUT optional part] --> BGOut[Out]
    #     end

    #     subgraph ResidualBlock
    #         direction TB
    #         BIn[Input] --> BConv[Strided Conv] --> BBN1[BatchNorm] --> ReLU --> BConv2[Conv] --> BBN2[BatchNorm] --> Add --> ReLu2[ReLU] --> RBOut[Out]
    #                 BIn --> DBConvD[OPTIONAL<br>1x1 Strided Conv] --> DBDBatchNorm[OPTIONAL<br>BatchNorm] --> Add
    #     end
    # end

    with st.columns(1)[0]:
        st.markdown(r"""
### Residual Block

Implement `ResidualBlock` by referring to the diagram. 

The number of channels changes from `in_feats` to `out_feats` at the first convolution in each branch; the second convolution in the left branch will have `out_feats` input channels and `out_feats` output channels.

The right branch being `OPTIONAL` means that its behaviour depends on the `first_stride` argument:

* If `first_stride=1`, this branch is just the identity operator, in other words it's a simple skip connection. Using `nn.Identity` might be useful here.
* If `first_stride>1`, this branch includes a convolutional layer with stride equal to `first_stride`, and a `BatchNorm` layer. This is also used as the stride of the **Strided Conv** in the left branch.""")

        with st.expander(r"""Question - why does the first_stride argument apply to only the first conv layer in the left branch, rather than to both convs in the left branch?"""):
            st.markdown(r"""This is to make sure that the size of the left and right branches are the same. If the `first_stride` argument applied to both left convs then the input would be downsampled too much so it would be smaller than the output of the right branch.
    
It's important for the size of the output of the left and right tracks to be the same, because they're added together at the end.""")

        with st.expander(r"""Help - I'm completely stuck on parts of the architecture."""):
            st.markdown(r"""In this case, you can use the following code to import your own `resnet34`, and inspect its architecture:

```python
if MAIN:
    resnet = torchvision.models.resnet34()
    print(torchinfo.summary(resnet, input_size=(1, 3, 64, 64)))
```

This will generate output telling you the names of each module, as well as the parameter counts.

Unfortunately, this function won't work on your own model if your model breaks when an image is passed through. Since a lot of the time mistakes in the architecture will mean your model doesn't work, you won't be able to use `torchinfo.summary` on your model. Instead, you should compare the models by printing them out.
""")

        st.markdown(r"""
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        pass
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        
        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )
        
        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()
            
        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        x_left = self.left(x)
        x_right = self.right(x)
        return self.relu(x_left + x_right)
```
""")

    with st.columns(1)[0]:
        st.markdown(r"""
### BlockGroup

Implement `BlockGroup` according to the diagram. 

The number of channels changes from `in_feats` to `out_feats` in the first `ResidualBlock` (all subsequent blocks will have `out_feats` input channels and `out_feats` output channels). 

The `height` and `width` of the input will only be changed if `first_stride>1` (in which case it will be downsampled by exactly this amount). 

You can also read the docstring for a description of the input and output shapes.

```python
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        pass
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        pass
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        
        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
            ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
        ]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)
```
""")
    with st.columns(1)[0]:
        st.markdown(r"""
### ResNet34

Last step! Assemble `ResNet34` using the diagram.
""")

        with st.expander(r"""Help - I'm not sure how to construct each of the BlockGroups."""):
            st.markdown(r"""
Each BlockGroup takes arguments `n_blocks`, `in_feats`, `out_feats` and `first_stride`. In the initialisation of `ResNet34` below, we're given a list of `n_blocks`, `out_feats` and `first_stride` for each of the BlockGroups. To find `in_feats` for each block, it suffices to note two things:
    
1. The first `in_feats` should be 64, because the input is coming from the convolutional layer with 64 output channels.
2. The `out_feats` of each layer should be equal to the `in_feats` of the subsequent layer (because the BlockGroups are stacked one after the other; with no operations in between to change the shape).

You can use these two facts to construct a list `in_features_per_group`, and then create your BlockGroups by zipping through all four lists.
""")

        with st.expander(r"""Help - I'm not sure how to construct the 7x7 conv at the very start."""):
            st.markdown(r"""
All the information about this convolution is given in the diagram, except for `in_channels`. Recall that the input to this layer is an RGB image. Can you deduce from this how many input channels your layer should have?
""")

        st.markdown(r"""
```python
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        pass


if MAIN:
    my_resnet = ResNet34()
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_feats0 = 64

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group,
                )
            )
        )
        # Alternative that uses `add_module`, in a way which makes the layer names line up:
        # for idx, (n_blocks, in_feats, out_feats, first_stride) in enumerate(zip(
        #     n_blocks_per_group, all_in_feats, out_features_per_group, strides_per_group
        # )):
        #     self.add_module(f"layer{idx+1}", BlockGroup(n_blocks, in_feats, out_feats, first_stride))

        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x
```
""")

    st.markdown(r"""
Now that you've built your `ResNet34`, we'll copy weights over from PyTorch's pretrained resnet to yours. This is a good way to verify that you've designed the architecture correctly.

```python
def copy_weights(my_resnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''
    
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict)
    
    state_dict_to_load = {}
    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue
    
    my_resnet.load_state_dict(state_dict_to_load)
    
    return my_resnet


if MAIN:
    pretrained_resnet = torchvision.models.resnet34(pretrained=True)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
```

This function uses the `state_dict()` method, which returns an  `OrderedDict` (documentation [here](https://realpython.com/python-ordereddict/)) object containing all the parameter/buffer names and their values. State dicts can be extracted from models, saved to your filesystem (this is a common way to store the results of training a model), and can also be loaded back into a model using the `load_state_dict` method. (Note that you can also load weights using a regular Python `dict`, but since Python 3.7, the builtin `dict` is guaranteed to maintain items in the order they're inserted.)

If the copying fails, this means that your model's layers don't match up with the layers in the PyTorch model implementation.

To debug here, we've given you a helpful function `utils.print_param_count`, which takes one or more models (as `*args`) and prints out a stylized dataframe comparing the parameter names and shapes of each model. It will tell you when your model matches up with the PyTorch implementation. It can be used as follows:

```python
utils.print_param_count(my_resnet, pretrained_resnet)
```
""")

    st_image('resnet-compared.png', width=900)

    st.markdown(r"""
Tweaking your model until all the layers match up might be a difficult and frustrating exercise at times! However, it's a pretty good example of the kind of low-level model implementation and debugging that is important for your growth as ML engineers. We'll be doing a few more model-building exercises similar to these in later sections.

## Running Your Model

We've provided you with some images for your model to classify:

```python
if MAIN:
    IMAGE_FILENAMES = [
        "chimpanzee.jpg",
        "golden_retriever.jpg",
        "platypus.jpg",
        "frogs.jpg",
        "fireworks.jpg",
        "astronaut.jpg",
        "iguana.jpg",
        "volcano.jpg",
        "goofy.jpg",
        "dragonfly.jpg",
    ]

    IMAGE_FOLDER = "./resnet_inputs"

    images = [Image.open(f"{IMAGE_FOLDER}/{filename}") for filename in IMAGE_FILENAMES]
```

Our `images` are of type `PIL.Image.Image`, so we can just call them in a cell to display them.

```python
images[0]
```
""")

    st_image('chimpanzee.jpg', width=500)

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - prepare the data

We now need to define a `transform` object like we did for MNIST. We will use the same transforms to convert the PIL image to a tensor, and to normalize it. But we also want to resize the images to `height=224, width=224`, because not all of them start out with this size and we need them to be consistent before passing them through our model. You should use `transforms.Resize` for this. Note that you should apply this resize to a tensor, not to the PIL image.

In the normalization step, we'll use a mean of `[0.485, 0.456, 0.406]`, and a standard deviation of `[0.229, 0.224, 0.225]`. Note that each of these values has three elements, because ImageNet contains RGB rather than monochrome images, and we're normalising over each of the three RGB channels separately.

```python
transform = transforms.Compose([]) # fill this in as instructed
```
""")
        with st.expander(r"Solution"):
            st.markdown(r"""
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
""")
        st.markdown(r"""
Now, write a function to prepare the data in `images` to be fed into our model. This should involve preprocessing each image, and stacking them into a single tensor along the batch (0th) dimension.

```python
def prepare_data(images: list[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    pass


if MAIN:
    prepared_images = prepare_data(images)
```
""")

        with st.expander(r"""Help - I'm not sure how to stack the images."""):
            st.markdown(r"""Use `t.stack`. The first argument of `t.stack` should be a list of preprocessed images.""")

        with st.expander(r"Solution"):
            st.markdown(r"""
```python
def prepare_data(images: List[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    x = t.stack([transform(img) for img in images], dim=0)
    return x
```
""")

    st.markdown(r"""Finally, we have provided you with a simple function which predicts the image's category by taking argmax over the output of the model.

```python
def predict(model, images):
    logits = model(images)
    return logits.argmax(dim=1)
```

You should use this function to compare your outputs to those of PyTorch's model. Hopefully, you should get the same results! We've also provided you with a file `imagenet_labels.json` which you can use to get the actual classnames of imagenet data, and see what your model's predictions actually are.

```python
with open("imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

if MAIN:
    # Check your predictions match the pretrained model's
    my_predictions = predict(my_resnet, prepared_images)
    pretrained_predictions = predict(pretrained_resnet, prepared_images)
    assert all(my_predictions == pretrained_predictions)

    # Print out your predictions, next to the corresponding images
    for img, label in zip(images, my_predictions):
        print(f"Class {label}: {imagenet_labels[label]}")
        display(img)
        print()
```

If you've done everything correctly, your version should give the same classifications, and the percentages should match at least to a couple decimal places.

If it does, congratulations, you've now run an entire ResNet, using barely any code from `torch.nn`! The only things we used were `nn.Module` and `nn.Parameter`.

If it doesn't, you get to practice model debugging! Remember to use the `utils.print_param_count` function that was provided.
""")

    with st.expander(r"""Help! My model is predicting roughly the same percentage for every category!"""):
        st.markdown(r"""This can indicate that your model weights are randomly initialized, meaning the weight loading process didn't actually take. Or, you reinitialized your model by accident after loading the weights.""")

    st.markdown(r"""
### Aside - hooks

One problem you might have encountered is that your model outputs `NaN`s rather than actual numbers. When debugging this, it's useful to try and identify which module the error first appears in. This is a great use-case for **hooks**, which are something we'll be digging a lot more into during our mechanistic interpretability exercises later on.

A hook is basically a function which you can attach to a particular `nn.Module`, which gets executed during your model's forward or backward passes. Here, we'll only consider forward hooks. A hook function's type signature is:

```python
def hook(module: nn.Module, inputs: List[t.Tensor], output: t.Tensor) -> None:
    pass
```

The `inputs` argument is a list of the inputs to the module (often just one tensor), and the `output` argument is the output of the module. This hook gets registered to a module by calling `module.register_forward_hook(hook)`. During forward passes, the hook function will run.

Here is some code which will check for `NaN`s in the output of each module, and raise a `ValueError` if it finds any. We've also given you an example tiny network which produces a `NaN` in the output of the second layer, to demonstrate it on.

```python
class NanModule(nn.Module):
    def forward(self, x):
        return t.full_like(x, float('nan'))

model = nn.Sequential(
    nn.Identity(),
    NanModule(),
    nn.Identity()
)


def hook_check_for_nan_output(module: nn.Module, input: Tuple[t.Tensor], output: t.Tensor) -> None:
    if t.isnan(output).any():
        raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
    '''
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    '''
    module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
    '''
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    '''
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


if MAIN:
    model.apply(add_hook)
    input = t.randn(3)
    output = model(input)
    model.apply(remove_hooks)
```

When you run this code, you should find it raising an error at the `NanModule`.
""")

    st.info(r"""
Important - when you're working with PyTorch hooks, make sure you remember to remove them at the end of each exercise! This is a classic source of bugs, and one of the things that make PyTorch hooks so janky. When we study TransformerLens in subsequent exercises, we'll use a version of hooks that is essentially the same under the hood, but comes with quite a few quality of life improvements!
""")

    st.markdown(r"""
---

Congrats for finishing the exercises! In the next day, we'll dig a bit deeper into training and optimizers, and we'll end by training a ResNet from scratch on data from ImageNet.
""")
 
def section_finetune():
    st.markdown(r"""
# Finetuning ResNet (bonus)

For your final exercise of the day, we're purposefully giving you much less guidance than we have for previous days. 

The goal here is to write a training loop for your ResNet (also recording accuracy on a test set at the end of each epoch) which gets decent performance on ImageNet data. However, since ResNet is a much larger architecture that takes a really long time (and a massive amount of data) to train, we'll just be doing **finetuning**. This is where you take a pretrained network and then retrain just the final layer (or few layers) of a neural network to perform some particular task. This is based on the idea that later layers in the network will already have been trained to recognise certain important features of the image, and all you're doing is applying some relatively simple functions to turn these features into useful predictors for your particular task.

[This PyTorch tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) takes you through the key ideas of **finetuning**, and also gives you an example task: distinguising ants from bees. You should have most the information you need to apply this technique to your own resnet. A few extra things you might need/want to look into:

* The function `datasets.ImageFolder` will be necessary to read in your data. See the [documentation page](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) for more details on how this function works. You can use the `hymenoptera` data in the ARENA GitHub repo (or you can download it directly from the PyTorch tutorial page). 
* You will need to use `model.train()` before each batch of training starts, as discussed earlier (see [this StackOverflow answer](https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch) for an explanation of why), and you'll need to use `model.eval()` when going into testing mode.

If you get stuck, then you can look at more parts of the PyTorch tutorial page (and maybe copy some parts of it), although you should ideally try and write the code on your own before doing this.

Good luck!
""")
 
func_list = [section_home, section_cnn, section_resnet] #, section_finetune]

page_list = ["🏠 Home", "1️⃣ Building & Training a CNN", "2️⃣ Assembling ResNet"] #, "3️⃣ Finetuning ResNet (bonus)"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:

        radio = st.radio("Section", page_list)

        st.markdown("---")

    func_list[page_dict[radio]]()


# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == st.secrets["password"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store password
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show input for password.
#         st.text_input("Password", type="password", on_change=password_entered, key="password")
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input("Password", type="password", on_change=password_entered, key="password")
#         st.error("😕 Password incorrect")
#         return False
#     else:
#         # Password correct.
#         return True

# if is_local or check_password():
page()