import os
from st_dependencies import *
st.set_page_config(layout="wide")

st.markdown(r"""
<style>
div[data-testid="column"] {
    background-color: #f9f5ff;
    padding: 15px;
}
.st-ae h2 {
    margin-top: -15px;
}
p {
    line-height:1.48em;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code {
    color:red;
    white-space: pre-wrap !important;
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#about-this-page">About this page</a></li>
    <li><a class="contents-el" href="#how-you-should-use-this-material">How you should use this material</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#option-1-colab">Option 1: Colab</a></li>
        <li><a class="contents-el" href="#option-2-your-own-ide">Option 2: Your own IDE</a></li>
        <li><a class="contents-el" href="#chatbot-assistant">Chatbot assistant</a></li>
    </ul></li>
    <li><a class="contents-el" href="#hints">Hints</a></li>
    <li><a class="contents-el" href="#test-functions">Test functions</a></li>
    <li><a class="contents-el" href="#tips">Tips</a></li>
    <li><a class="contents-el" href="#feedback">Feedback</a></li>
</ul>
""", unsafe_allow_html=True)

def section_home():
    st_image('prereqs.png', width=600)
    # start
    st.error(r"""
Note - these exercises have now been supplanted by [chapter 0](https://arena-ch0-fundamentals.streamlit.app/) of the ARENA material, which covers everything in this page and then some. This page will no longer be updated going forwards.
""")
    st.markdown(r"""
# Chapter 1: Fundamentals

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, and generative models).

Some highlights from this chapter include:
* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism

---

## About this page

This page was made using an app called Streamlit. It's hosted from the prerequisite materials [GitHub repo](https://github.com/callummcdougall/Prerequisite-materials). It provides a very simple way to display markdown, as well as more advanced features like interactive plots and animations. This is how the instructions for each day will be presented.

On the left, you can see a sidebar (or if it's collapsed, you will be able to see if you click on the small arrow in the top-left to expand it). This sidebar should show a page called `Home` (which is the page you're currently reading), as well as one for each of the different parts of today's exercises.
""")

#     st.info(r"""
# Note - these exercises form different sections of the day, rather than corresponding to different days. At the start of each exercise, I've included an estimated completion time. This should be taken with a pinch of salt (you might prefer to go at different speeds, or be more/less comfortable with certain sections). But if you find yourself going well outside this estimate, then it's probably a sign that you should be more willing to ask for help (either by sending a message in the `#technical-questions` Slack, or asking any TAs who are present).
# """)

    st.markdown(r"""
If you want to change to dark mode, you can do this by clicking the three horizontal lines in the top-right, then navigating to Settings → Theme.

## How you should use this material

### Option 1: Colab

The simplest way to get these exercises up and running is to use Colab. This guarantees good GPU support, and means you'll spend less time messing around with environments and IDEs. Each set of exercises will have a link to the accompanying Colab notebook, which you should make a copy of and work through. The Colabs have essentially the same structure as the Streamlit pages.

[Here](https://drive.google.com/drive/folders/1YnTChxQTJnJfFhqyHA44h9Nro79AaYpn?usp=sharing) is the link to the folder containing all the Colabs, and the data you'll need. You can find the individual Colabs below (not all the exercises have been converted into Colab form yet):

* Raytracing: [**exercises**](https://colab.research.google.com/drive/1tp-vd591FarVyn7pA2V9oYDqYiWmjEjF?usp=share_link), [**solutions**](https://colab.research.google.com/drive/19QroufIT25oZ5yG7JGWL5Jp9IPcsq0d4?usp=sharing)
* as_strided, convolutions and CNNs: [**exercises**](https://colab.research.google.com/drive/1hQE1inYldFI_mmpCiLbIW8yI2C-PxBev?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1VZk9ba3j7HJP9ChntblOoEAwxZukCgHn?usp=sharing)
* Build Your Own Backprop Framework: [**exercises**](https://colab.research.google.com/drive/1n-OG0x7kZfZaMCNO-S4L86-W6bE_jiVz?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1K3f_ebaaHDufnGbn_zzzTisejXTM_b01?usp=sharing)
* ResNets & Model Training: Links to Colab: [**exercises**](https://colab.research.google.com/drive/1N1Cu13q4dk2Z0qYgdy7Cnb6ESAlOu5ge?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1obMRz1Y9iXrJbQBXaYCBS61S-mxOIhWO?usp=sharing)
* Optimization & Hyperparameters: [**exercises**](https://colab.research.google.com/drive/1Wi_SVL8eDYiNcmcmUeF4GfkNfQKT6x3O?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1JfIRCJZ_Fi_WJGneuOKKqF_qsxJfdbfZ?usp=sharing)

You can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells. You can also look at the **solutions** notebooks if you're just interested in the output (since they have all the correct code filled in, and all the output on display within the notebook).

### Option 2: Your own IDE

An alternative way to use this material is to run it on an IDE of your own choice (we strongly recommend VSCode). The vast majority of the exercises will not require a particularly good GPU, and where there are exceptions we will give some advice for how to get the most out of the exercises regardless.

Full instructions for running the exercises in this way:

* Clone the [GitHub repo](https://github.com/callummcdougall/TransformerLens-intro) into your local directory.
* Open in your choice of IDE (we recommend VSCode).
* Make & activate a virtual environment
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
* Install requirements.
    * First, install PyTorch using the following command: `conda install pytorch=1.11.0 torchdata torchvision -c pytorch -y`.
    * Then install the rest of the requirements by navigating to the directory and running `pip install -r requirements.txt`.
* While in the directory, run `streamlit run Home.py` in your terminal (this should work since Streamlit is one of the libraries in `requirements.txt`).
    * This should open up a local copy of the page you're reading right now, and you're good to go!

To complete one of the exercise pages, you should:

* Navigate to `exercises` in the repo
* Create a file called `part1_answers.py` (or `part1_answers.ipynb` if you prefer using notebooks)
* Go through the Streamlit page, and copy over / fill in then run the appropriate code as you go through the exercises.

### Chatbot assistant

In the sidebar of this page, below the contents page, you will (at first) see an error message saying "Please set the OpenAI key...". This is space for a chatbot assistant, which can help answer your questions about the material. Take the following steps to set it up:

* Go to the [OpenAI API](https://openai.com/blog/openai-api) and sign up for an account.
* Create a secret key from [this page](https://platform.openai.com/account/api-keys). Copy this key.
* Create a file `.streamlit/secrets.toml` in this repo, and have the first line read `api_secret = "<your key>"`.
* Refresh the page, and you should now be able to use the chatbot.

This interface was built using the `openai` library, and it exists to help answer questions you might have about the material. All prompts from this chatbot are prepended with most\* of the material on the page and section you're currently reading. For instance, try passing in the question ***What are 2 ways to use this material?*** to the chatbot, and it should describe the two options given above (i.e. colab, or your own IDE). This feature is very experimental, so please [let me know](mailto:cal.s.mcdougall@gmail.com) if you have any feedback!

\**Because of the context window, the entire page isn't always included in the prompt (e.g. generally code blocks aren't included). When in doubt, you can copy sections of the page into the prompt and run it! If you get an error message saying that the prompt is too long, then you can use the **clear chat** button and start again.*

Here are some suggestions for the kinds of questions you can ask the chatbot (in the appropriate sections of the course):

* *(copying in a function to the start of your prompt)* What does this function do?
* What is an intuitive explanation of induction heads?
* What is the difference between top-k and top-p sampling?

## Hints

There will be occasional hints throughout the document, for when you're having trouble with a certain task but you don't want to read the solutions. Click on the expander to reveal the solution in these cases. Below is an example of what they'll look like:
""")

    with st.expander("Help - I'm stuck on a particular problem."):
        st.markdown("Here is the answer!")

    st.markdown(r"""Always try to solve the problem without using hints first, if you can.

## Test functions

Most of the blocks of code will also come with test functions. These are imported from python files with names such as `exercises/part1_raytracing_tests.py`. You should make sure these files are in your working directory while you're writing solutions. One way to do this is to clone the [main GitHub repo](https://github.com/callummcdougall/arena-v1) into your working directory, and run it there. When we decide exactly how to give participants access to GPUs, we might use a different workflow, but this should suffice for now. Make sure that you're getting the most updated version of utils at the start of every day (because changes might have been made), and keep an eye out in the `#errata` channel for mistakes which might require you to change parts of the test functions.

## Tips

* To get the most out of these exercises, make sure you understand why all of the assertions should be true, and feel free to add more assertions.
* If you're having trouble writing a batched computation, try doing the unbatched version first.
* If you find these exercises challenging, it would be beneficial to go through them a second time so they feel more natural.

## Feedback

If you have any feedback on this course (e.g. bugs, confusing explanations, parts that you feel could be structured better), please let me know using [this Google Form](https://forms.gle/2ZhdHa87wWsrATjh9).
""")
    # end

# ## Support

# If you ever need help, you can send a message on the ARENA Slack channel `#technical-questions`. You can also reach out to a TA (e.g. Callum) if you'd like a quick videocall to talk through a concept or a problem that you've been having, although there might not always be someone available.

# You can also read the solutions by downloading them from the [GitHub](https://github.com/callummcdougall/arena-v1). However, ***this should be a last resort***. Really try and complete the exercises as a pair before resorting to the solutions. Even if this involves asking a TA for help, this is preferable to reading the solutions. If you do have to read the solutions, then make sure you understand why they work rather than just copying and pasting. 

# At the end of each day, it can be beneficial to look at the solutions. However, these don't always represent the optimal way of completing the exercises; they are just how the author chose to solve them. If you think you have a better solution, we'd be really grateful if you could send it in, so that it can be used to improve the set of exercises for future ARENA iterations.

# Happy coding!

# if is_local or check_password():

if "current_section" not in st.session_state:
    st.session_state["current_section"] = ["", ""]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = ["", ""]

def page():
    section_home()
    current_page = r"home"
    prepend = parse_text_from_page(current_page, r"section_home")
    st.session_state["current_page"] = [current_page, st.session_state["current_page"][0]]
    new_page = st.session_state["current_page"][0] != st.session_state["current_page"][1]
    chatbot_setup(prepend=prepend, new_page=new_page, debug=False)

page()
