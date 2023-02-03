import os
if not os.path.exists("./images"):
    os.chdir("./ch0")
from st_dependencies import *
styling()

import plotly.io as pio
import re
import json

def read_from_html(filename):
    filename = f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

def get_fig_dict():
    names = [f"fig{i}" for i in range(1, 16)] + [f"rosenbrock_{i}" for i in range(1, 5)]
    return {(name[-1] if name.startswith("fig") else name): read_from_html(name) for name in names}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
fig_dict = st.session_state["fig_dict"]

def section_home():
    st_image("stats.png", 350)
    st.markdown(r"""
# Training models

## 1️⃣ Optimizers

These exercises will take you through how different optimisation algorithms work (specifically SGD, RMSprop and Adam). You'll write your own optimisers, and use plotting functions to visualise gradient descent on loss landscapes.

## 2️⃣ Weights and Biases

In this section, we'll look at methods for choosing hyperparameters effectively. You'll learn how to use **Weights and Biases**, a useful tool for hyperparameter search, which should allow you to tune your own transformer model by the end of today's exercises.

## 3️⃣ Lambda Labs (bonus)

This third section is optional. It shows you how to set up a Lambda Labs account, and connect to a more powerful GPU. This allows you to train larger models, and may come in handy for future exrcises in this program.
""")

def section_wandb():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#cifar10"><code>CIFAR10</code></a></li>
    <li><a class="contents-el" href="#what-is-weights-and-biases">What is Weights and Biases?</a></li>
    <li><a class="contents-el" href="#logging-runs-with-wandb">Logging runs with <code>wandb</code></a></li>
    <li><a class="contents-el" href="#hyperparameter-search">Hyperparameter search</a></li>
    <li><a class="contents-el" href="#running-hyperparameter-sweeps-with-wandb">Running hyperparameter sweeps with <code>wandb</code></a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Weights and Biases

Today, we'll look at methods for choosing hyperparameters effectively. You'll learn how to use **Weights and Biases**, a useful tool for hyperparameter search, which should allow you to tune your own transformer model by the end of today's exercises.

The exercises themselves will be based on your ResNet implementations from yesterday, although the principles should carry over to other models you'll build in this course (such as transformers next week).

Note, this page only contains one exercise, and it's relatively short. You're encouraged to spend some time playing around with Weights and Biases, but you should also spend some more time finetuning your ResNet from yesterday (you might want to finetune ResNet during the morning, and look at today's material in the afternoon - you can discuss this with your partner). You should also spend some time reviewing the last three days of material, to make sure there are no large gaps in your understanding.

## CIFAR10

The benchmark we'll be training on is [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60000 32x32 colour images in 10 different classes. Don't peek at what other people online have done for CIFAR10 (it's a common benchmark), because the point is to develop your own process by which you can figure out how to improve your model. Just reading the results of someone else would prevent you from learning how to get the answers. To get an idea of what's possible: using one V100 and a modified ResNet, one entry in the DAWNBench competition was able to achieve 94% test accuracy in 24 epochs and 76 seconds. 94% is approximately [human level performance](http://karpathy.github.io/2011/04/27/manually-classifying-cifar10/).

Below is some boilerplate code for downloading and transforming `CIFAR10` data (this shouldn't take more than a minute to run the first time). There are a few differences between this and our code yesterday week - for instance, we omit the `transforms.Resize` from our `transform` object, because CIFAR10 data is already the correct size (unlike the sample images from last week).

```python
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

utils.show_cifar_images(trainset, rows=3, cols=5)
```

We have also provided a basic training & testing loop, almost identical to the one you used yesterday. This one doesn't use `wandb` at all, although it does plot the train loss and test accuracy when the function finishes running. You should run this function to verify your model is working, and that the loss is going down. Also, make sure you understand what each part of this function is doing.
""")

    with st.expander("TRAIN FUNCTION - SIMPLE"):

        st.markdown(r"""
```python
def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> tuple[list, list]:

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    
    loss_list = []
    accuracy_list = []

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = "./w0d3_resnet.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)

    utils.plot_results(loss_list, accuracy_list)
    return loss_list, accuracy_list
    
epochs = 1
loss_fn = nn.CrossEntropyLoss()
batch_size = 128
lr = 0.001
    
loss_list, accuracy_list = train(trainset, testset, epochs, loss_fn, batch_size, lr)
```
""")

    st.markdown(r"""
## What is Weights and Biases?

Weights and Biases is a cloud service that allows you to log data from experiments. Your logged data is shown in graphs during training, and you can easily compare logs across different runs. It also allows you to run **sweeps**, where you can specifiy a distribution over hyperparameters and then start a sequence of test runs which use hyperparameters sampled from this distribution.

The way you use weights and biases is pretty simple. You take a normal training loop, and add a bit of extra code to it. The only functions you'll need to use today are:

* `wandb.init`, which starts a new run to track and log to W&B
* `wandb.watch`, which hooks into your model to track parameters and gradients
* `wandb.log`, which logs metrics and media over time within your training loop
* `wandb.save`, which saves the details of your run
* `wandb.sweep` and `wandb.agent`, which are used to run hyperparameter sweeps

You should visit the [Weights and Biases homepage](https://wandb.ai/home), and create your own user. You will also have to login the first time you run `wandb` code (this can be done by running `wandb login` in whichever terminal you are using).

## Logging runs with `wandb`

The most basic way you can use `wandb` is by logging variables during your run. This removes the need for excessive printing of output. Below is an example training loop which does this.
""")

    with st.expander("TRAIN FUNCTION - WANDB LOGGING"):
        st.markdown(r"""
```python
def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> None:

    config_dict = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }
    wandb.init(project="w2d1_resnet", config=config_dict)

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

train(trainset, testset, epochs, loss_fn, batch_size, lr)
```
""")

    st.markdown(r"""
When you run this function, it should give you a url which you can click on to see a graph of parameters plotted over time. You can also switch to a tabular view, which will let you compare multiple runs.

Some notes on the parts of this function which have been changed:

#### `wandb.init`

This starts a new run to track and log to Weights and Biases. The `project` keyword allows you to group all your runs in a single directory, and the `config` keyword accepts a dictionary of hyperparameters which will be logged to wandb (you can see these in your table when you compare different runs).

`wandb.init` must always be the very first `wandb` function that gets called.

#### `wandb.watch`

This hooks into your model to track parameters and gradients. You should be able to see graphs of your parameter and gradient values (`log="all"` means that both of these are tracked; the other options being `log="gradients"` and `log="parameters"`).

#### `wandb.log`

This logs metrics and media over time within your training loop. The two arguments it takes here are `data` (a dictionary of values which should be logged; you will be able to see a graph for each of these parameters on Weights and Biases) and `step` (which will be the x-axis of the graphs). If `step` is not provided, by default `wandb` will increment step once each time it logs, so `step` will actually correspond to the batch number. In the code above, `step` corresponds to the total number of examples seen.

#### `wandb.save`

This saves the details of your run. You can view your saved models by navigating to a run page, clicking on the `Files` tab, then clicking on your model file.

You should try runing this function a couple of times with some different hyperparameters, and get an idea for how it works.

## Hyperparameter search

One way to do hyperparameter search is to choose a set of values for each hyperparameter, and then search all combinations of those specific values. This is called **grid search**. The values don't need to be evenly spaced and you can incorporate any knowledge you have about plausible values from similar problems to choose the set of values. Searching the product of sets takes exponential time, so is really only feasible if there are a small number of hyperparameters. I would recommend forgetting about grid search if you have more than 3 hyperparameters, which in deep learning is "always".

A much better idea is for each hyperparameter, decide on a sampling distribution and then on each trial just sample a random value from that distribution. This is called **random search** and back in 2012, you could get a [publication](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) for this. The diagram below shows the main reason that random search outperforms grid search. Empirically, some hyperparameters matter more than others, and random search benefits from having tried more distinct values in the important dimensions, increasing the chances of finding a "peak" between the grid points.
""")

    st_image('grid_vs_random.png', width=550)

    st.markdown(r"""
It's worth noting that both of these searches are vastly less efficient than gradient descent at finding optima - imagine if you could only train neural networks by randomly initializing them and checking the loss! Either of these search methods without a dose of human (or eventually AI) judgement is just a great way to turn electricity into a bunch of models that don't perform very well.

## Running hyperparameter sweeps with `wandb`

Now we've come to one of the most impressive features of `wandb` - being able to perform hyperparameter sweeps. Below is a final function which implements hyperparameter sweeps.
""")

    with st.expander("TRAIN FUNCTION - WANDB SWEEP"):
        st.markdown("""
```python
def train() -> None:

    wandb.init()

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)


        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

sweep_config = {
    'method': 'random',
    'name': 'w2d1_resnet_sweep_2',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [64, 128, 256]},
        'epochs': {'min': 1, 'max': 3},
        'lr': {'max': 0.1, 'min': 0.0001, 'distribution': 'log_uniform_values'}
     }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='w2d1_resnet')

wandb.agent(sweep_id=sweep_id, function=train, count=2)
```
""")

    st.markdown(r"""
There's a lot of extra stuff here, but in fact the core concepts are relatively straightforward. We'll go through the new code line by line.

Firstly, note that we've kept everything the same in our `train` function except for the code at the start. The function now doesn't take any arguments (`trainset`, `testset` and `loss_fn` are used as global variables), and the hyperparameters `epochs`, `batch_size` and `lr` are now defined from `wandb.config` rather than being passed into `wandb.init` via the `config` argument.

Most of the extra code comes at the end. First, let's look at `sweep_config`. This dictionary provides all the information `wandb` needs in order to conduct hyperparameter search. The important keys are:

* `method`, which determines how the hyperparameters are searched for.
    * `random` just means random search, as described above.
    * Other options are `grid` (also described above) and `bayes` (which is a smart way of searching for parameters that adjusts in the direction of expected metric improvement based on a Gaussian model).
* `name`, which is just the name of your sweep in Weights and Biases.
* `metric`, which is a dictionary of two keys: `name` (what we want to optimise) and `goal` (the direction we want to optimise it in). 
    * Note that `name` must be something which is logged by our model in the training loop (in this case, `'test_accuracy'`).
    * You can also be clever with the metrics that you maximise: for instance:
        * Minimising training time, by having your training loop terminate early when the loss reaches some threshold (although you'll have to be careful here, e.g. in cases where your loss never reaches the threshold).
        * Optimising some function of multiple metrics, e.g. a weighted sum.
* `parameters`, which is a dictionary with items of the form `hyperparameter_name: search_method`. This determines which hyperparameters are searched over, and how the search is conducted.
    * There are several ways to specify hyperparameter search in each `search_method`. You can read more [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).
    * The simplest search methods are `values` (choose uniformly from a list of values).
        * This can also be combined with `probabilities`, which should be a list specifying the probability of selecting each element from `values`.
    * You can also specify `min` and `max`, which causes wandb to choose uniformly, either from a discrete or continuous uniform distribution depending on whether the values for `min` and `max` are integers or floats.
    * You can also pass the argument `distribution`, which gives you more control over how the random values are selected. For instance, `log_uniform_values` returns a value `X` between `min` and `max` s.t. `log(X)` is uniformly distributed between `log(min)` and `log(max)`.
        * (Question - can you see why a log uniform distribution for `lr` makes more sense than a uniform distribution?)
""")

    with st.expander("Note on using YAML files (optional)"):
        st.markdown("""
Rather than using a dictionary, you can alternatively store the `sweep_config` data in a YAML file if you prefer. You will then be able to run a sweep via the following terminal command:

```
wandb sweep sweep_config.yaml

wandb agent <SWEEP_ID>
```

where `SWEEP_ID` is the value returned from the first terminal command. You will also need to add another line to the YAML file, specifying the program to be run. For instance, your YAML file might start like this:

```yaml
program: train.py
method: random
metric:
    name: test_accuracy
    goal: maximize
```
""")

    st.markdown("""

The final two lines above return a `sweep_id`, and then use that ID to run a sweep. Note that `wandb.agent`'s arguments include a named function (this is why it was important for our `train` function not to take any arguments), and `count` (which determines how many sweeps will be run before the process terminates). 

When you run the code above, you will be given a url called **Sweep page**, in output that will look like:

```
Sweep page: https://wandb.ai/<WANDB-USERNAME>/<PROJECT-NAME>/<SWEEP_ID>
```

This URL will bring you to a page comparing each of your sweeps. You'll be able to see overlaid graphs of each of their training loss and test accuracy, as well as a bunch of other cool things like:

* Bar charts of the [importance](https://docs.wandb.ai/ref/app/features/panels/parameter-importance) (and correlation) of each hyperparameter wrt the target metric. Note that only looking at the correlation could be misleading - something can have a correlation of 1, but still have a very small effect on the metric.
* A [parallel coordinates plot](https://docs.wandb.ai/ref/app/features/panels/parallel-coordinates), which summarises the relationship between the hyperparameters in your config and the model metric you're optimising.

---

`wandb` is an incredibly useful tool when training models, and you should find yourself using it a fair amount throughout this programme. You can always return to this page of exercises if you forget how any part of it works.
""")

def section_lambda():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#reading">Reading</a></li>
   <li><a class="contents-el" href="#introduction-lambda-labs">Introduction - Lambda Labs</a></li>
   <li><a class="contents-el" href="#instructions-for-signing-up">Instructions for signing up</a></li>
   <li><a class="contents-el" href="#vscode-remote-ssh-extension">VSCode remote-ssh extension</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#windows">Windows</a></li>
       <li><a class="contents-el" href="#linux-/-macos">Linux / MacOS</a></li>
   </ul></li>
   <li><a class="contents-el" href="#launch-your-instance">Launch your instance</a></li>
   <li><a class="contents-el" href="#set-up-your-config-file">Set up your config file</a></li>
   <li><a class="contents-el" href="#connect-to-your-instance)">Connect to your instance</a></li>
   <li><a class="contents-el" href="#exercise-use-your-gpu-to-speed-up-training-loops">Exercise - use your GPU to speed up training loops</a></li>
</ul>
""", unsafe_allow_html=True)
    st.markdown(r"""
# Lambda Labs

This page provides a guide for how to get set up on Lambda Labs (with different instructions depending on your OS). Once you finish this, you should be able to run large models.

There is also some reading material, which provides an overview of what GPUs are and how they work, as well as some of the topics we'll be returning to in later parts of this chapter.

## Reading

* [Techniques for Training Large Neural Networks](https://openai.com/blog/techniques-for-training-large-neural-networks/)
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

## Introduction - Lambda Labs

Lambda Labs is a service giving you access to higher-quality GPUs than you are likely to find in your laptop. Knowing how to run models on GPUs is essential for performing large-scale experients.

In later sections of this chapter we'll look at multi-GPU setups, but for now we'll just stick to the basics: setting up a single GPU, and SSHing into it.
""")
    st.error("""Warning - **Lambda Labs charge by the hour for GPU usage**. If you use it, make sure you remember to terminate your instances when you're done with them!""")

    st.markdown(r"""
## Instructions for signing up

Sign up for an account [here](https://lambdalabs.com/service/gpu-cloud).

Add an **SSH key**. Give it a name like `<Firstname><Lastname>` (we will refer to this as `<keyname>` from now on).

When you create it, it will automatically be downloaded. The file should have a `.pem` extension - this is a common container format for keys or certificates.

## VSCode remote-ssh extension

The [**remote ssh extension**](https://code.visualstudio.com/docs/remote/ssh) is very useful for abstracting away some of the messy command-line based details of SSH. You should install this extension now.""")

    st_image("architecture-ssh.png", 600)

    st.markdown(r"""
At this point, the instructions differ between Windows and Linux/MacOS.

### Windows

Having installed the SSH extension, Windows may have automatically created a .ssh file for you, and it will be placed in `C:\Users\<user>` by default. If it hasn't done this, then you should create one yourself (you can do this from the Windows command prompt via `md C:\Users\<user>\.ssh`).

Move your downloaded SSH key into this folder. Then, set permissions on the SSH key (i.e. the `.pem` file):
		
* Right click on file, press “Properties”, then go to the “Security” tab.
* Click “Advanced”, then “Disable inheritance” in the window that pops up.""")
    st_image("instruction1.png", 500)
    st.markdown(r"""
* Choose the first option “Convert inherited permissions…”""")
    st_image("instruction2.png", 500)
    st.markdown(r"""
* Go back to the “Security” tab, click "Edit" to change permissions, and remove every user except the owner.
    * You can check who the owner is by going back to "Security -> Advanced" and looking for the "Owner" field at the top of the window).

### Linux / MacOS

* Make your `.ssh` directory using the commands `mkdir -p ~/.ssh` then `chmod 700 ~/.ssh`.
* Set permissions on the key: `chmod 600 ~/.ssh/<keyname>.pem`

Yep, it's that much simpler than Windows 😂

## Launch your instance

Go back to the Lambda Labs page, go to "instances", and click "Launch instance".

You'll see several options, some of them might be greyed out if unavailable. Pick a cheap one (we're only interested in testing this at the moment, and at any rate even a relatively cheap one will probably be more powerful than the one you're currently using in your laptop). 

Enter your SSH key name. Choose a region (your choice here doesn't really matter for our purposes).

Once you finish this process, you should see your GPU instance is running:""")

    st_image("gpu_instance.png", 700)
    st.markdown(r"""You should also see an SSH LOGIN field, which will look something like: `ssh ubuntu@<ip-address>`.

## Set up your config file

Setting up a **config file** remove the need to use long command line arguments, e.g. `ssh -i ~/.ssh/<keyname>.pem ubuntu@instance-ip-address`.""")

    st.markdown(f"""Click on the {st_image("vscode-ssh.png", 35, return_html=True)} button in the bottom left, choose "Open SSH Configuration File...", then click <code>C:\\Users\\<user>\\.ssh\\config</code>.""", unsafe_allow_html=True)

    st.markdown(r"""
An empty config file will open. You should copy in the following instructions:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>.pem
    User <user>
```

where the IP address and user come from the **SSH LOGIN** field in the table, and the identity file is the path of your SSH key. For instance, the file I would use (corresponding to the table posted above) looks like:

```c
Host <ip-address>
    IdentityFile C:\Users\<user>\.ssh\<keyname>.pem
    User <user>
```

## Connect to your instance""")

    st.markdown(f"""Click the green button {st_image("vscode-ssh.png", 35, return_html=True)} again, and choose "Connect to Host...". Your IP address should appear as one of the hosts. Choose this option.""", unsafe_allow_html=True)
    st.markdown(r"""
A new VSCode window will open up. If you're asked if you want to install the recommended extensions for Python, click yes. If you're asked to choose an OS (Windows, Mac or Linux), choose Linux.

Click on the file explorer icon in the top-left, and open the directory `ubuntu` (or whichever directory you want to use as your working directory in this machine). 

And there you go - you're all set! 

To check your GPU is working, you can open a Python or Notebook file and run `!nvidia-smi`. You should see GPU information which matches the machine you chose from the Lambda Labs website, and is different from the result you get from running this command on your local machine. 

Another way to check out your GPU For instance is to run the PyTorch code `torch.cuda.get_device_name()`. For example, this is what I see after SSHing in:""")

    st_image("gpu_type.png", 600)
    st_image("gpu_type_2.png", 450)

    st.markdown(r"""
You can also use `torch.cuda.get_device_properties` (which takes your device as an argument).

Once you've verified this is working, you can start running code on GPUs. The easiest way to do this is just to drag and drop your files into the file explorer window on the left hand side.

You'll also need to choose a Python interpreter. Choose the conda or miniconda one if it's available, if not then choose the top-listed version. You'll probably need to `!pip install` some libraries, e.g. einops, fancy-einsum, and plotly if you're using it.

## Exercise - use your GPU to speed up training loops

You can now go back to your ResNet fine-tuning code from the previous section. How much of a speedup is there? Can you relate this to what you read in [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)?
""")


def section_optim():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#reading">Reading</a></li>
    <li><a class="contents-el" href="#gradient-descent">Gradient Descent</a></li>
    <li><a class="contents-el" href="#stochastic-gradient-descent">Stochastic Gradient Descent</a></li>
    <li><a class="contents-el" href="#batch-size">Batch Size</a></li>
    <li><a class="contents-el" href="#common-themes-in-gradient-based-optimizers">Common Themes in Gradient-Based Optimizers</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#weight-decay">Weight Decay</a></li>
        <li><a class="contents-el" href="#momentum">Momentum</a></li>
    </ul></li>
    <li><a class="contents-el" href="#visualising-optimization-with-rosenbrocks-banana">Visualising Optimization With Rosenbrock's Banana</a></li>
    <li><a class="contents-el" href="#build-your-own-optimizers">Build Your Own Optimizers</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#exercise-implement-sgd">SGD</a></li>
        <li><a class="contents-el" href="#exercise-implement-rmsprop">RMSprop</a></li>
        <li><a class="contents-el" href="#exercise-implement-adam">Adam</a></li>
    </ul></li>
    <li><a class="contents-el" href="#plotting-multiple-optimisers">Plotting multiple optimisers</a></li>
    <li><a class="contents-el" href="#bonus-parameter-groups">Bonus - parameter groups</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#when-to-use-parameter-groups">When to use parameter groups</a></li>
        <li><a class="contents-el" href="#exercises">Exercises</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Optimizers

## Imports

```python
import torch as t
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb

import part4_training_utils as utils
import part4_training_tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

```

## Reading

Some of these are strongly recommended, while others are optional. If you like, you can jump back to some of these videos while you're going through the material, if you feel like you need to.

* Andrew Ng's video series on gradient descent variants:
    * [Gradient Descent With Momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I) (9 mins)
    * [RMSProp](https://www.youtube.com/watch?v=_e-LFe_igno) (7 mins)
    * [Adam](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23) (7 mins)
* [A Visual Explanation of Gradient Descent Methods](https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c)
* [Why Momentum Really Works (distill.pub)](https://distill.pub/2017/momentum/)
""")

    st.markdown(r"""
## Gradient Descent

Yesterday, you implemented backpropagation. Today, we're going to use the gradients produced by backpropagation for optimizing a loss function using gradient descent.""")

    st.info("""
Note the conceptual shift here - we're not optimising the parameters of a neural network; we're optimising parameters `(x, y)` which represent coordinates at which we evaluate a function. We're doing this because the image of "loss landscapes" can be very helpful when thinking about the behaviour of different gradient descent algorithms.""")

    st.markdown(r"""
A loss function can be any differentiable function such that we prefer a lower value. To apply gradient descent, we start by initializing the parameters to random values (the details of this are subtle), and then repeatedly compute the gradient of the loss with respect to the model parameters. It [can be proven](https://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx) that for an infinitesimal step, moving in the direction of the gradient would increase the loss by the largest amount out of all possible directions.

We actually want to decrease the loss, so we subtract the gradient to go in the opposite direction. Taking infinitesimal steps is no good, so we pick some learning rate $\lambda$ (also called the step size) and scale our step by that amount to obtain the update rule for gradient descent:

$$\theta_t \leftarrow \theta_{t-1} - \lambda \nabla L(\theta_{t-1})$$

We know that an infinitesimal step will decrease the loss, but a finite step will only do so if the loss function is linear enough in the neighbourhood of the current parameters. If the loss function is too curved, we might actually increase our loss.

The biggest advantage of this algorithm is that for N bytes of parameters, you only need N additional bytes of memory to store the gradients, which are of the same shape as the parameters. GPU memory is very limited, so this is an extremely relevant consideration. The amount of computation needed is also minimal: one multiply and one add per parameter.

The biggest disadvantage is that we're completely ignoring the curvature of the loss function, not captured by the gradient consisting of partial derivatives. Intuitively, we can take a larger step if the loss function is flat in some direction or a smaller step if it is very curved. Generally, you could represent this by some matrix P that pre-multiplies the gradients to rescale them to account for the curvature. $P$ is called a preconditioner, and gradient descent is equivalent to approximating $P$ by an identity matrix, which is a very bad approximation.

Most competing optimizers can be interpreted as trying to do something more sensible for $P$, subject to the constraint that GPU memory is at a premium. In particular, constructing $P$ explicitly is infeasible, since it's an $N \times N$ matrix and N can be hundreds of billions. One idea is to use a diagonal $P$, which only requires N additional memory. An example of a more sophisticated scheme is [Shampoo](https://arxiv.org/pdf/1802.09568.pdf).
""")

    st.info("""The algorithm is called **Shampoo** because you put shampoo on your hair before using conditioner, and this method is a pre-conditioner.
    
If you take away just one thing from this entire curriculum, please don't let it be this.""")

    st.markdown("""
## Stochastic Gradient Descent

The terms gradient descent and SGD are used loosely in deep learning. To be technical, there are three variations:

- Batch gradient descent - the loss function is the loss over the entire dataset. This requires too much computation unless the dataset is small, so it is rarely used in deep learning.
- Stochastic gradient descent - the loss function is the loss on a randomly selected example. Any particular loss may be completely in the wrong direction of the loss on the entire dataset, but in expectation it's in the right direction. This has some nice properties but doesn't parallelize well, so it is rarely used in deep learning.
- Mini-batch gradient descent - the loss function is the loss on a batch of examples of size `batch_size`. This is the standard in deep learning.

The class `torch.SGD` can be used for any of these by varying the number of examples passed in. We will be using only mini-batch gradient descent in this course.

## Batch Size

In addition to choosing a learning rate or learning rate schedule, we need to choose the batch size or batch size schedule as well. Intuitively, using a larger batch means that the estimate of the gradient is closer to that of the true gradient over the entire dataset, but this requires more compute. Each element of the batch can be computed in parallel so with sufficient compute, one can increase the batch size without increasing wall-clock time. For small-scale experiments, a good heuristic is thus "fill up all of your GPU memory".

At a larger scale, we would expect diminishing returns of increasing the batch size, but empirically it's worse than that - a batch size that is too large generalizes more poorly in many scenarios. The intuition that a closer approximation to the true gradient is always better is therefore incorrect. See [this paper](https://arxiv.org/pdf/1706.02677.pdf) for one discussion of this.

For a batch size schedule, most commonly you'll see batch sizes increase over the course of training. The intuition is that a rough estimate of the proper direction is good enough early in training, but later in training it's important to preserve our progress and not "bounce around" too much.

You will commonly see batch sizes that are a multiple of 32. One motivation for this is that when using CUDA, threads are grouped into "warps" of 32 threads which execute the same instructions in parallel. So a batch size of 64 would allow two warps to be fully utilized, whereas a size of 65 would require waiting for a third warp to finish. As batch sizes become larger, this wastage becomes less important.

Powers of two are also common - the idea here is that work can be recursively divided up among different GPUs or within a GPU. For example, a matrix multiplication can be expressed by recursively dividing each matrix into four equal blocks and performing eight smaller matrix multiplications between the blocks.

In tomorrow's exercises, you'll have the option to expore batch sizes in more detail.

## Common Themes in Gradient-Based Optimizers

### Weight Decay

Weight decay means that on each iteration, in addition to a regular step, we also shrink each parameter very slightly towards 0 by multiplying a scaling factor close to 1, e.g. 0.9999. Empirically, this seems to help but there are no proofs that apply to deep neural networks.

In the case of linear regression, weight decay is mathematically equivalent to having a prior that each parameter is Gaussian distributed - in other words it's very unlikely that the true parameter values are very positive or very negative. This is an example of "**inductive bias**" - we make an assumption that helps us in the case where it's justified, and hurts us in the case where it's not justified.

For a `Linear` layer, it's common practice to apply weight decay only to the weight and not the bias. It's also common to not apply weight decay to the parameters of a batch normalization layer. Again, there is empirical evidence (such as [Jai et al 2018](https://arxiv.org/pdf/1807.11205.pdf)) and there are heuristic arguments to justify these choices, but no rigorous proofs. Note that PyTorch will implement weight decay on the weights *and* biases of linear layers by default - see the bonus exercises tomorrow for more on this.

### Momentum

Momentum means that the step includes a term proportional to a moving average of past gradients. [Distill.pub](https://distill.pub/2017/momentum/) has a great article on momentum, which you should definitely read if you have time. Don't worry if you don't understand all of it; skimming parts of it can be very informative. For instance, the first half discusses the **conditioning number** (a very important concept to understand in optimisation), and concludes by giving an intuitive argument for why we generally set the momentum parameter close to 1 for ill-conditioned problems (those with a very large conditioning number).

## Visualising Optimization With Rosenbrock's Banana

"Rosenbrock's Banana" is a (relatively) famous function that has a simple equation but is challenging to optimize because of the shape of the loss landscape.

We've provided you with a function to calculate Rosenbrock's Banana, and another one to plot arbitrary functions. You can see them both below:

```python
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

x_range = [-2, 2]
y_range = [-1, 3]
fig = utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
```

Your output should look like:
""")

    st.plotly_chart(fig_dict["rosenbrock_1"].update_layout(height=600), use_container_width=True)

    with st.expander("Question - where is the minimum of this function?"):
        st.markdown(r"""
The first term is minimised when `x=a` and the second term when `y=x**2`. So we deduce that the minimum is at `(a, a**2)`. When `a=1`, this gives us the minimum `(1, 1)`.

You can pass the extra argument `show_min=True` to all plotting functions, to indicate the minimum.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `opt_fn_with_sgd`

Implement the `opt_fn` function using `torch.optim.SGD`. Starting from `(-1.5, 2.5)`, run your function and add the resulting trajectory of `(x, y)` pairs to your contour plot. Did it find the minimum? Play with the learning rate and momentum a bit and see how close you can get within 100 iterations.

```python
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad
    pass
```
""")

        with st.expander("Help - I'm not sure if my `opt_banana` is implemented properly."):
            st.markdown(r"With a learning rate of `0.001` and momentum of `0.98`, my SGD was able to reach `[ 1.0234,  1.1983]` after 100 iterations.")

        with st.expander("Help - all my (x, y) points are the same."):
            st.markdown(r"""This is probably because you've stored your `xy` values in a list, so they change each time you perform a gradient descent step. 

Instead, try creating a tensor of zeros to hold them, and fill in that tensor using `xys[i] = xy.detach()` at each step.""")

        with st.expander("Help - I'm getting 'Can't call numpy() on Tensor that requires grad'."):
            st.markdown("""
This is a protective mechanism built into PyTorch. The idea is that once you convert your Tensor to NumPy, PyTorch can no longer track gradients, but you might not understand this and expect backprop to work on NumPy arrays.

All you need to do to convince PyTorch you're a responsible adult is to call detach() on the tensor first, which returns a view that does not require grad and isn't part of the computation graph.
""")

    st.markdown("""
We've also provided you with a function `plot_optimisation_sgd` to plot the steps in your optimisation algorithm. It can be run like this:

```python
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]

fig = utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, lr=0.001, momentum=0.98, show_min=True)

fig.show()
```

Hopefully, you should see output like this:
""")

    st.plotly_chart(fig_dict["rosenbrock_2"].update_layout(height=600), use_container_width=True)

    st.markdown("""
## Build Your Own Optimizers

Now let's build our own drop-in replacement for these three classes from `torch.optim`. The documentation pages for these algorithms have pseudocode you can use to implement your step method.

""")
    st.info("""
**A warning regarding in-place operations**

Be careful with expressions like `x = x + y` and `x += y`. They are NOT equivalent in Python.

- The first one allocates a new `Tensor` of the appropriate size and adds `x` and `y` to it, then rebinds `x` to point to the new variable. The original `x` is not modified.
- The second one modifies the storage referred to by `x` to contain the sum of `x` and `y` - it is an "in-place" operation.
    - Another way to write the in-place operation is `x.add_(y)` (the trailing underscore indicates an in-place operation).
    - A third way to write the in-place operation is `torch.add(x, y, out=x)`.
- This is rather subtle, so make sure you are clear on the difference. This isn't specific to PyTorch; the built-in Python `list` follows similar behavior: `x = x + y` allocates a new list, while `x += y` is equivalent to `x.extend(y)`.

The tricky thing that happens here is that both the optimizer and the `Module` in your model have a reference to the same `Parameter` instance. 
""")

    with st.expander("Question - should we use in-place operations in our optimizer?"):
        st.markdown("""
You MUST use in-place operations in your optimizer because we want the model to see the change to the Parameter's storage on the next forward pass. If your optimizer allocates a new tensor, the model won't know anything about the new tensor and will continue to use the old, unmodified version.

Note, this observation specifically refers to the parameters. When you're updating non-parameter variables that you're tracking, you should be careful not to accidentally use an in-place operation where you shouldn't!)""")

    st.markdown(r"""### More Tips

- The provided `params` might be a generator, in which case you can only iterate over it once before the generator is exhausted. **You should copy it into a `list` to be able to iterate over it repeatedly.**
- Your step function shouldn't modify the gradients. Use the `with torch.inference_mode():` context for this. Fun fact: you can instead use `@torch.inference_mode()` (note the preceding `@`) as a method decorator to do the same thing.
- If you create any new tensors, they should be on the same device as the corresponding parameter. Use `torch.zeros_like()` or similar for this.
- Be careful not to mix up `Parameter` and `Tensor` types in this step.
- The actual PyTorch implementations have an additional feature called parameter groups where you can specify different hyperparameters for each group of parameters. You can ignore this for now; we'll come back to it in the next section.

Note, the configurations used during testing will start simple (e.g. all parameters set to zero except `lr`) and gradually move to more complicated ones. This will help you track exactly where in your model the error is coming from.

You should also fill in the default PyTorch keyword arguments, where appropriate.

""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement SGD

First, you should implement stochastic gradient descent. It should be like the [PyTorch version](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD), but assume `nesterov=False`, `maximize=False`, and `dampening=0`. These simplifications mean that there are many variables in the pseudocode at that link which you can ignore.

```python
class SGD:
    params: list

    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float, weight_decay: float):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        pass


if MAIN:
    tests.test_sgd(SGD)
```

If you've having trouble, you can use the following process when implementing your optimisers:

1. Take the pseudocode from the PyTorch documentation page, and write out the "simple version", i.e. without all of the extra variables which you won't need. (It's good practice to be able to parse pseudocode and figure out what it actually means - during the course we'll be doing a lot more of "transcribing instructions / formulae from paper into code"). You'll want pen and paper for this!

2. Figure out which extra variables you'll need to track within your class.

3. Implement the `step` function using these variables.

You can click on the expander below to see what the first two steps look like for the case of SGD (try and have a go at each step before you look).
""")

        with st.expander("STEP 1"):
            st.markdown(r"""
In the SGD pseudocode, you'll first notice that we can remove the nesterov section, i.e. we always do $g_t \leftarrow \boldsymbol{b}_t$. Then, we can actually remove the variable $\boldsymbol{b_t}$ altogether (because we only needed it to track values while implementing nesterov). Lastly, we have `maximize=False` and `dampening=0`, which allows us to further simplify. So we get the simplified pseudocode:

$
\text {for } t=1 \text { to } \ldots \text { do } \\
\quad g_t \leftarrow \nabla_\theta f_t\left(\theta_{t-1}\right) \\
\quad \text {if } \lambda \neq 0 \\
\quad\quad g_t \leftarrow g_t+\lambda \theta_{t-1} \\
\quad \text {if } \mu \neq 0 \text{ and } t>1 \\
\quad\quad g_t \leftarrow \mu g_{t-1} + g_t \\
\quad \theta_t \leftarrow \theta_{t-1} - \gamma g_t
$

Note - you might find it helpful to name your variables in the `__init__` step in line with their definitions in the pseudocode, e.g. `self.mu = momentum`. This will make it easier to implement the `step` function.
""")

        with st.expander("STEP 2"):
            st.markdown(r"""
In the formula from STEP 1, $\theta_t$ represents the parameters themselves, and $g_t$ represents variables which we need to keep track of in order to implement momentum. We need to track $g_t$ in our model, e.g. using a line like:

```python
self.gs = [t.zeros_like(p) for p in self.params]
```

We also need to track the variable $t$, because the behavour is different when $t=0$. (Technically we could just as easily not do this, because the behaviour when $t=0$ is just the same as the behaviour when $g_t=0$ and $t>0$. But I've left $t$ in my solutions, to make it more obvious how the `SGD.step` function corrsponds to the pseudocode shown in STEP 1.

Now, you should be in a good position to attempt the third step: applying SGD in the `step` function, using this algorithm and these tracked variables.
""")

    st.markdown("")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement RMSprop

Once you've implemented SGD, you should do RMSprop in a similar way. Although the pseudocode is more complicated and there are more variables you'll have to track, there is no big conceptual difference between the task for RMSprop and SGD.

If you want to better understand why RMSprop works, then you can return to some of the readings at the top of this page.

```python
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        pass
    


tests.test_rmsprop(RMSprop)
```
""")
    st.markdown("")
    with st.columns(1)[0]:
        st.markdown(r"""
### Exercise - implement Adam

Finally, you'll do the same for Adam. This is a very popular optimizer in deep learning, which empirically often outperforms most others. It combines the heuristics of both momentum (via the $\beta_1$ parameter), and RMSprop's handling of noisy data by dividing by the $l_2$ norm of gradients (via the $\beta_2$ parameter).

```python
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def __repr__(self) -> str:
        pass

tests.test_adam(Adam)
```
""")
    st.markdown(r"""
## Plotting multiple optimisers

Finally, we've provided some code which should allow you to plot more than one of your optimisers at once.

First, you should fill in this function, which will be just like your `opt_fn_with_sgd` from earlier, except that it works when passed an arbitrary optimizer (from the ones that you've defined).

```python
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad
    pass
```

Once you've implemented this function, you can use `utils.plot_optimization` to create plots of multiple different optimizers at once. An example of how this should work can be found below. The argument `optimizers` should be a list of tuples `(optimizer_class, optimizer_kwargs)` which will get passed into `opt_fn`.

```python
xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (solution.SGD, dict(lr=1e-3, momentum=0.98)),
    (solution.SGD, dict(lr=5e-4, momentum=0.98)),
]

fig = utils.plot_optimization(opt_fn, fn, xy, optimizers, x_range, y_range)

fig.show()
```
""")
    st.plotly_chart(fig_dict["rosenbrock_3"].update_layout(height=600), use_container_width=True)

    st.markdown(r"""
You can try and play around with a few optimisers. Do Adam and RMSprop do well on this function? Why / why not? Can you find some other functions where they do better / worse, and plot those?

## Bonus - parameter groups
""")
    st.error(r"""
*If you're interested in these exercises then you can go through them, if not then you can move on to the next section (weights and biases).*
""")
    st.markdown(r"""
Rather than passing a single iterable of parameters into an optimizer, you have the option to pass a list of parameter groups, each one with different hyperparameters. As an example of how this might work:

```python
optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```

The first argument here is a list of dictionaries, with each dictionary defining a separate parameter group. Each should contain a `params` key, which contains an iterable of parameters belonging to this group. The dictionaries may also contain keyword arguments. If a parameter is not specified in a group, PyTorch uses the value passed as a keyword argument. So the example above is equivalent to:

```python
optim.SGD([
    {'params': model.base.parameters(), 'lr': 1e-2, 'momentum': 0.9},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'momentum': 0.9}
])
```

PyTorch optimisers will store all their params and hyperparams in the `param_groups` attribute, which is a list of dictionaries like the one above, where each one contains *every* hyperparameter rather than just the ones that were specified by the user at initialisation. Optimizers will have this `param_groups` attribute even if they only have one param group - then `param_groups` will just be a list containing a single dictionary.

### When to use parameter groups

Parameter groups can be useful in several different circumstances. A few examples:

* Finetuning a model by freezing earlier layers and only training later layers is an extreme form of parameter grouping. We can use the parameter group syntax to apply a modified form, where the earlier layers have a smaller learning rate. This allows these earlier layers to adapt to the specifics of the problem, while making sure they don't forget all the useful features they've already learned.
* Often it's good to treat weights and biases differently, e.g. effects like weight decay are often applied to weights but not biases. PyTorch doesn't differentiate between these two, so you'll have to do this manually using paramter groups.
    * This in particular, we'll be doing next week when we train BERT from scratch.

More generally, if you're trying to replicate a paper, it's important to be able to use all the same training details that the original authors did, so you can get the same results.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - rewrite SGD to use parameter groups

You should rewrite the `SGD` optimizer from the earlier exercises, to use `param_groups`. A few things to keep in mind during this exercise:

* The learning rate must either be specified as a keyword argument, or it must be specified in every group. If it isn't specified as a keyword argument or there's at least one group in which it's not specified, you should raise an error.
    * This isn't true for the other hyperparameters like momentum. They all have default values, and so they don't need to be specified.
* You should add some code to check that no parameters appear in more than one group (PyTorch raises an error if this happens).

```python
class SGD:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        pass
    
    def zero_grad(self) -> None:
        pass

tests.test_sgd_param_groups(SGD)
```
""")

func_list = [section_home, section_optim, section_wandb, section_lambda]

page_list = ["🏠 Home", "1️⃣ Optimizers", "2️⃣ Weights and Biases", "3️⃣ Lambda Labs"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:

        radio = st.radio("Section", page_list)

        st.markdown("---")

    func_list[page_dict[radio]]()

page()