import os
# if not os.path.exists("./images"):
#     os.chdir("./ch0")
from st_dependencies import *
styling()

import plotly.io as pio
import re
import json

def read_from_html(filename):
    filename = f"../images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return pio.from_json(json.dumps(plotly_json))

def get_fig_dict():
    return {str(i): read_from_html(f"fig{i}") for i in range(1, 16)}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
fig_dict = st.session_state["fig_dict"]

# %%

# make_rays_1d.png
# raytracing.png

def section_home():
    pass

def section_1():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
   <li><a class="contents-el" href="#1d-image-rendering">1D Image Rendering</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#tip-the-out-keyword-argument">Tip - the <code>out</code> keyword argument</a></li>
   </ul></li>
   <li><a class="contents-el" href="#ray-object-intersection">Ray-Object Intersection</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#aside-typechecking">Aside - typechecking</a></li>
   </ul></li>
   <li><a class="contents-el" href="#batched-ray-segment-intersection">Batched Ray-Segment Intersection</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#tip-ellipsis">Tip - Ellipsis</a></li>
       <li><a class="contents-el" href="#tip-elementwise-logical-operations-on-tensors">Tip - Elementwise Logical Operations on Tensors</a></li>
       <li><a class="contents-el" href="#tip-operator-precedence">Tip - Operator Precedence</a></li>
       <li><a class="contents-el" href="#tip-logical-reductions">Tip - Logical Reductions</a></li>
       <li><a class="contents-el" href="#tip-broadcasting">Tip - Broadcasting</a></li>
       <li><a class="contents-el" href="#summary-of-all-these-tips">Summary of all these tips</a></li>
   </ul></li>
   <li><a class="contents-el" href="#2d-rays">2D Rays</a></li>
   <li><a class="contents-el" href="#triangle-coordinates">Triangle Coordinates</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#triangle-ray-intersection">Triangle-Ray Intersection</a></li>
   </ul></li>
   <li><a class="contents-el" href="#single-triangle-rendering">Single-Triangle Rendering</a></li>
   <li><ul class="contents">
       <li><a class="contents-el" href="#views-and-copies">Views and Copies</a></li>
       <li><a class="contents-el" href="#storage-objects">Storage Objects</a></li>
       <li><a class="contents-el" href="#tensor-base"><code>Tensor._base</code></a></li>
   </ul></li>
   <li><a class="contents-el" href="#mesh-loading">Mesh Loading</a></li>
   <li><a class="contents-el" href="#mesh-rendering">Mesh Rendering</a></li>
   <li><a class="contents-el" href="#bonus-content">Bonus Content</a></li>
""", unsafe_allow_html=True)
    st_image("raytracing.png", 350)
    st.markdown(r"""
# Ray Tracing

Today we'll be practicing batched matrix operations in PyTorch by writing a basic graphics renderer. We'll start with an extremely simplified case and work up to rendering your very own 3D Pikachu! Note that if you're viewing this file on GitHub, some of the equations may not render properly. Viewing it locally in VS Code should fix this.

## 1D Image Rendering

In our initial setup, the **camera** will be a single point at the origin, and the **screen** will be the plane at x=1.

**Objects** in the world consist of triangles, where triangles are represented as 3 points in 3D space (so 9 floating point values per triangle). You can build any shape out of sufficiently many triangles and your Pikachu will be made from 412 triangles.

The camera will emit one or more **rays**, where a ray is represented by an **origin** point and a **direction** point. Conceptually, the ray is emitted from the origin and continues in the given direction until it intersects an object.

We have no concept of lighting or color yet, so for now we'll say that a pixel on our screen should show a bright color if a ray from the origin through it intersects an object, otherwise our screen should be dark.
""")
    st_image("ray_tracing.png", 400)
    st.markdown(r"""
To start, we'll let the z dimension in our `(x, y, z)` space be zero and work in the remaining two dimensions. 

Implement the following `make_rays_1d` function so it generates some rays coming out of the origin, which we'll take to be `(0, 0, 0)`.

Calling `render_lines_with_pyplot` on your rays should look like this (note the orientation of the axes):
""")
    st_image("make_rays_1d.png", 400)
    st.markdown(r"""
```python
import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as wg
from torchtyping import TensorType as TT
from IPython.display import display

import part1_raytracing_tests as tests

MAIN = __name__ == "__main__"


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    pass


def render_lines_with_plotly(lines: t.Tensor, bold_lines: t.Tensor = t.Tensor()):
    '''
    Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).

    bold_lines: same shape as lines. If supplied, these lines will be rendered in black on top of the other lines.
    '''
    fig = go.Figure(layout=dict(showlegend=False, title="3D rays"))
    for line in lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines")
    for line in bold_lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines", line_width=5, line_color="black")
    fig.show()


if MAIN:
    rays1d = make_rays_1d(9, 10.0)
    fig = render_lines_with_plotly(rays1d)
```
""")
    with st.expander("Soluion"):
        st.markdown(r"""
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays
""")

    st.markdown(r"""

### Tip - the `out` keyword argument

Many PyTorch functions take an optional keyword argument `out`. If provided, instead of allocating a new tensor and returning that, the output is written directly to the `out` tensor.

If you used `torch.arange` or `torch.linspace` above, try using the `out` argument. Note that a basic indexing expression like `rays[:, 1, 1]` returns a view that shares storage with `rays`, so writing to the view will modify `rays`. You'll learn more about views later today.

## Ray-Object Intersection

Suppose we have a line segment defined by points $L_1$ and $L_2$. Then for a given ray, we can test if the ray intersects the line segment like so:

- Supposing both the ray and line segment were infinitely long, solve for their intersection point.
- If the point exists, check whether that point is inside the line segment and the ray. 

Our camera ray is defined by the origin $O$ and direction $D$ and our object line is defined by points $L_1$ and $L_2$.

We can write the equations for all points on the camera ray as $R(u)=O +u D$ for $u \in [0, \infty)$ and on the object line as $O(v)=L_1+v(L_2 - L_1)$ for $v \in [0, 1]$.

The following interactive widget lets you play with this parameterization of the problem:

```python
if MAIN:
    v = wg.FloatSlider(min=-2, max=2, step=0.01, value=0.5, description="v")
    seed = wg.IntSlider(min=0, max=10, step=1, value=0, description="random seed")

    fig = go.FigureWidget(go.Scatter(x=[], y=[]))
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12)
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
    fig.update_layout(showlegend=False, xaxis_range=[-1.5, 2.5], yaxis_range=[-1.5, 2.5])

    def response(change):
        t.manual_seed(seed.value)
        L_1, L_2 = t.rand(2, 2)
        P = lambda v: L_1 + v * (L_2 - L_1)
        x, y = zip(P(-2), P(2))
        with fig.batch_update(): 
            fig.data[0].update({"x": x, "y": y}) 
            fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
            fig.data[2].update({"x": [P(v.value)[0]], "y": [P(v.value)[1]]}) 
        
    v.observe(response)
    seed.observe(response)
    response("")

    box = wg.VBox([v, seed, fig])
    display(box)
```

Setting the line equations from above equal gives the solution:

$$
\begin{aligned}O + u D &= L_1 + v(L_2 - L_1) \\ u D - v(L_2 - L_1) &= L_1 - O  \\ \begin{pmatrix} D_x & (L_1 - L_2)_x \\ D_y & (L_1 - L_2)_y \\ \end{pmatrix} \begin{pmatrix} u \\ v \\ \end{pmatrix} &=  \begin{pmatrix} (L_1 - O)_x \\ (L_1 - O)_y \\ \end{pmatrix} \end{aligned}
$$

Once we've found values of $u$ and $v$ which satisfy this equation, if any (the lines could be parallel) we just need to check that $u \geq 0$ and $v \in [0, 1]$.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - which segments intersect with the rays?
 
For each of the following segments, which camera rays from earlier intersect? You can do this by inspection or using `render_lines_with_pyplot`.

```python
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])
"TODO: YOUR CODE HERE"
```
""")
        with st.expander("Solution - Intersecting Rays"):
            st.markdown(r"""
```python
if MAIN:
    render_lines_with_plotly(rays1d, segments)
```

- Segment 0 intersects the first two rays.
- Segment 1 doesn't intersect any rays.
- Segment 2 intersects the last two rays. Computing `rays * 2` projects the rays out to `x=1.5`. Remember that while the plot shows rays as line segments, rays conceptually extend indefinitely.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `intersect_ray_1d`

Using [`torch.lingalg.solve`](https://pytorch.org/docs/stable/generated/torch.linalg.solve.html) and [`torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html), implement the `intersect_ray_1d` function to solve the above matrix equation.
""")
        with st.expander("Aside - difference between stack and concatenate"):
            st.markdown(r"""
`torch.stack` will combine tensors along a new dimension.

```python
>>> t.stack([t.ones(2, 2), t.zeros(2, 2)], dim=0)
tensor([[[1., 1.],
         [1., 1.]],

        [[0., 0.],
         [0., 0.]]])
```

`torch.concat` (alias `torch.cat`) will combine tensors along an existing dimension.

```python
>>> t.cat([t.ones(2, 2), t.zeros(2, 2)], dim=0)
tensor([[1., 1.], 
        [1., 1.],
        [0., 0.],
        [0., 0.]])
```

Here, you should use `torch.stack` to construct e.g. the matrix on the left hand side, because you want to combine the vectors $D$ and $L_1 - L_2$ to make a matrix.
""")
        st.markdown(r"""
Is it possible for the solve method to fail? Give a sample input where this would happen.
""")
        with st.expander("Answer - Failing Solve"):
            st.markdown(r"""
If the ray and segment are exactly parallel, then the solve will fail because there is no solution to the system of equations. For this function, handle this by catching the exception and returning False.
""")
        st.markdown(r"""
```python
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''

if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
```
""")

        with st.expander("Help! My code is failing with a 'must be batches of square matrices' exception."):
            st.markdown(r"""
Our formula only uses the x and y coordinates - remember to discard the z coordinate for now. 

It's good practice to write asserts on the shape of things so that your asserts will fail with a helpful error message. In this case, you could assert that the `mat` argument is of shape (2, 2) and the `vec` argument is of shape (2,). Also, see the aside below on typechecking.
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    # Get the x and y coordinates (ignore z)
    ray = ray[..., :2]
    segment = segment[..., :2]

    # Ray is [[Ox, Oy], [Dx, Dy]]
    O, D = ray
    # Segment is [[L1x, L1y], [L2x, L2y]
    L_1, L_2 = segment

    # Create matrix and vector, and solve equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    # Solve equation (return False if no solution)
    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False
    
    # If there is a solution, check the soln is in the correct range for there to be an intersection
    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)
```
""")
    st.markdown(r"""
### Aside - typechecking

Typechecking is a useful habit to get into. It's not strictly necessary, but it can be a great help when you're debugging.

One good way to typecheck in PyTorch is with the `torchtyping`. The most important object in this library is the `TensorType` object, which can be used to specify things like the shape and dtype of a tensor.

In its simplest form, this just behaves like a fancier version of a docstring or comment (signalling to you, as well as any readers, what the size of objects should be). But you can also use the `typeguard.typechecked` to strictly enforce the type signatures of your inputs and outputs. For instance, if you replaced the `intersect_ray_1d` function with the following:

```python
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def intersect_ray_1d(ray: TT[2, 3], segment: TT[2, 3]) -> bool:
    ...
```

then you would get an error when running this function, if the 0th dimension of `logit_attr` didn't match the length of `tokens`. Alternatively, you could use `TT[2, "ndim"]` for both arguments; this signals to your IDE that these values are meant to be the same, so you'll get an error whenever they're different.

*(Note, it's necessary to call `patch_typeguard()` once before you use the `typechecked` decorator, but you only need to call it once in your file, rather than before each function.)*

You can do other things with `TorchTyping`, such as:
* Specify values for dimensions, e.g. `TT["batch", 512, "embed"]`
* Specify values and names, e.g. `TT["batch", 512, "embed": 768]`
* Specify dtypes, e.g. `TT["batch", 512, t.float32]` checks the tensor has shape `(?, 512)` and dtype `torch.float32`.

You can read more [here](https://github.com/patrick-kidger/torchtyping).

## Batched Ray-Segment Intersection

Next, implement a batched version that takes multiple rays, multiple line segments, and returns a boolean for each ray indicating whether **any** segment intersects with that ray.

Note - in the batched version, we don't want the solver to throw an exception just because some of the equations don't have a solution - these should just return False. 

### Tip - Ellipsis

You can use an ellipsis `...` in an indexing expression to avoid repeated `:' and to write indexing expressions that work on varying numbers of input dimensions. 

For example, `x[..., 0]` is equivalent to `x[:, :, 0]` if `x` is 3D, and equivalent to `x[:, :, :, 0]` if `x` is 4D.

### Tip - Elementwise Logical Operations on Tensors

For regular booleans, the keywords `and`, `or`, and `not` are used to do logical operations and the operators `&`, `|`, and `~` do and, or and not on each bit of the input numbers. For example `0b10001 | 0b11000` is `0b11001` or 25 in base 10.

Tragically, Python doesn't allow classes to overload keywords, so if `x` and `y` are of type `torch.Tensor`, then `x and y` does **not** do the natural thing that you probably expect, which is compute `x[i] and y[i]` elementwise. It actually tries to coerce `x` to a regular boolean, which throws an exception.

As a workaround, PyTorch (and NumPy) have chosen to overload the bitwise operators but have them actually mean logical operations, since you usually don't care to do bitwise operations on tensors. So the correct expression would be `x & y` to compute `x[i] and y[i]` elementwise.

### Tip - Operator Precedence

Another thing that tragically doesn't do what you would expect is an expression like `v >= 0 & v <= 1`. The operator precedence of `&` is so high that this statement parses as `(v >= (0 & v)) <= 1`.

The correct expression uses parentheses to force the proper parsing: `(v >= 0) & (v <= 1)`. 

### Tip - Logical Reductions

In plain Python, if you have a list of lists and want to know if any element in a row is `True`, you could use a list comprehension like `[any(row) for row in rows]`. The efficient way to do this in PyTorch is with `torch.any()` or equivalently the `.any()` method of a tensor, which accept the dimension to reduce over. Similarly, `torch.all()` or `.all()` method. Both of these methods accept a `dim` argument, which is the dimension to reduce over.

You can accomplish the same thing with `einops.reduce` but that's more cumbersome.

### Tip - Broadcasting

Broadcasting is what happens when you perform an operation on two tensors, and one is a smaller size, but is copied along the dimensions of the larger one in order to apply to it. Example:

```python
A = t.ones(2, 3)
B = t.arange(3)
print(A + B)
```

Broadcasting sematics are a bit messy, and we'll go into it in more detail later in the course. If you want to get a full picture of it then click on the dropdown below, but for now here's the important thing to know - ***broadcasting of tensors `A` and `B` (where `B` has fewer dimensions) will work if the shape of `B` is a suffix of the shape of `A`***. The code block above is an example of this.
""")
    with st.expander("More on broadcasting"):
        st.markdown(r"""
If you try to broadcast tensors `A` and `B`, then the following happens:

* The tensor with fewer dimensions is padded with dimensions of size one on the left.
* Once the tensors have the same number of dimensions, they are checked for compatibility.
    * Two dimensions are compatible if they are equal, or if one of them is one (in the latter case, we repeat the size-1 tensor along that dimension until it's the same size as the larger one).
    * If they are not compatible, then broadcasting is not allowed.

Here are a few examples:

* If `A` has shape `(2, 3)` and `B` has shape `(3,)`, then:
    * `B` is padded with a dimension of size one on the left, giving `(1, 3)`.
    * `B` is copied along the first dimension, giving `(2, 3)`.
    * Broadcasting works!
* If `A` has shape `(2, 3)` and `B` has shape `(2,)`, then:
    * `B` is padded with a dimension of size one on the left, giving `(1, 2)`.
    * Compare the new shapes of `A` and `B`; broadcasting fails because the last dimensions don't match.
""")
    st.info(r"""
### Summary of all these tips

* Use `...` to avoid repeated `:` in indexing expressions.
* Use `&`, `|`, and `~` for elementwise logical operations on tensors.
* Use parentheses to force the correct operator precedence.
* Use `torch.any()` or `.any()` to do logical reductions (you can do this over a single dimension, with the `dim` argument).
* If you're trying to broadcast tensors `A` and `B` (where `B` has fewer dimensions), this will work if the shape of `B` is a **suffix** of the shape of `A`.
""")
    st.markdown("")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `intersect_rays_1d`

```python
def intersect_rays_1d(
    rays: TT["nrays", 2, 3], 
    segments: TT["nsegments", 2, 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    pass


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
```
""")

        with st.expander("Help - I'm not sure how to implment this function without a for loop."):
            st.markdown(r"""
Initially, `rays.shape == (NR, 2, 3)` and `segments.shape == (NS, 2, 3)`. Try performing `einops.repeat` on them, so both their shapes are `(NR, NS, 2, 3)`. Then you can formulate and solve the batched system of matrix equations.
""")
        with st.expander("Help - I'm not sure how to deal with the cases of zero determinant."):
            st.markdown(r"""
You can use `t.linalg.det` to compute the determinant of a matrix, or batch of matrices *(gotcha: the determinant won't be exactly zero, but you can check that it's very close to zero, e.g. `det.abs() < 1e-6`)*. This will give you a boolean mask for which matrices are singular.

You can set all singular matrices to the identity (this avoids errors), and then at the very end you can use your boolean mask again to set the intersection to `False` for the singular matrices.
""")
        with st.expander("Help - I'm still stuck on the zero determinant cases."):
            st.markdown(r"""
After formulating the matrix equation, you should have a batch of matrices of shape `(NR, NS, 2, 2)`, i.e. `mat[i, j, :, :]` is a matrix which looks like:

$$
\begin{pmatrix} D_x & (L_1 - L_2)_x \\ D_y & (L_1 - L_2)_y \\ \end{pmatrix}
$$

Calling `t.linalg.det(mat)` will return an array of shape `(NR, NS)` containing the determinants of each matrix. You can use this to construct a mask for the singular matrices *(gotcha: the determinant won't be exactly zero, but you can check that it's very close to zero, e.g. `det.abs() < 1e-6`)*.

Indexing `mat` by this mask will return an array of shape `(x, 2, 2)`, where the zeroth axis indexes the singular matrices. As we discussed in the broadcasting section earlier, this means we can use broadcasting to set all these singular matrices to the identity:

```python
mat[is_singular] = t.eye(2)
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def intersect_rays_1d(rays: TT["nrays", 2, 3], segments: TT["nsegments", 2, 3]) -> TT["nrays", bool]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    NR = rays.size(0)
    NS = segments.size(0)

    # Get just the x and y coordinates
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Repeat rays and segments so that we can compuate the intersection of every (ray, segment) pair
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    # Each element of `rays` is [[Ox, Oy], [Dx, Dy]]
    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    # Each element of `segments` is [[L1x, L1y], [L2x, L2y]]
    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]
    assert L_1.shape == (NR, NS, 2)

    # Define matrix on left hand side of equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    assert is_singular.shape == (NR, NS)
    mat[is_singular] = t.eye(2)

    # Define vector on the right hand side of equation
    vec = L_1 - O

    # Solve equation, get results
    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)
```
""")
    st.markdown(r"""
## 2D Rays

Now we're going to make use of the z dimension and have rays emitted from the origin in both y and z dimensions.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `make_rays_2d`

Implement `make_rays_2d` analogously to `make_rays_1d`. The result should look like a pyramid with the tip at the origin.

```python
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    pass


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
```
""")
        with st.expander("Help - I'm not sure how to implement this function."):
            st.markdown(r"""
Don't write it as a function right away. The most efficient way is to write and test each line individually in the REPL to verify it does what you expect before proceeding.

You can either build up the output tensor using `torch.stack`, or you can initialize the output tensor to its final size and then assign to slices like `rays[:, 1, 1] = ...`. It's good practice to be able to do it both ways.

Each y coordinate needs a ray with each corresponding z coordinate - in other words this is an outer product. The most elegant way to do this is with two calls to `einops.repeat`. You can also accomplish this with `unsqueeze`, `expand`, and `reshape` combined.
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays
```
""")
    st.markdown(r"""

## Triangle Coordinates

The area inside a triangle can be defined by three (non-collinear) points $A$, $B$ and $C$, and can be written algebraically as a **convex combination** of those three points:

$$
\begin{align*}
P(w, u, v) &= wA + uB + vC \quad\quad \\
    \\
s.t. \quad 0 &\leq w,u,v \\
1 &= w + u + v
\end{align*}
$$

Or equivalently:

$$
\begin{align*}
\quad\quad\quad\quad P(u, v) &= (1 - u - v)A + uB + vC \\
&= A + u(B - A) + v(C - A) \\
\\
s.t. \quad 0 &\leq u,v \\
u + v &\leq 1
\end{align*}
$$

These $u, v$ are called "barycentric coordinates".

If we remove the bounds on $u$ and $v$, we get an equation for the plane containing the triangle. Play with the widget to understand the behavior of $u, v$.

```python
if MAIN:
    u = wg.FloatSlider(min=-0.5, max=1.5, step=0.01, value=0, description="u")
    v = wg.FloatSlider(min=-0.5, max=1.5, step=0.01, value=0, description="v")

    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = go.FigureWidget(
        data=[
            go.Scatter(x=x, y=y, mode="markers+text", text=["A", "B", "C"], textposition="middle left", textfont_size=18, marker_size=12),
            go.Scatter(x=[*x, x[0]], y=[*y, y[0]], mode="lines"),
            go.Scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
        ],
        layout=dict(
            title="Barycentric coordinates illustration", showlegend=False,
            xaxis_range=[-3, 8], yaxis_range=[-2, 5.5],
        )
    )

    def response(change):
        P = A + u.value * (B - A) + v.value * (C - A)
        fig.data[2].update({"x": [P[0]], "y": [P[1]]})
        
    u.observe(response)
    v.observe(response)
    response("")

    box = wg.VBox([u, v, fig])
    display(box)
```

### Triangle-Ray Intersection

Given a ray with origin $O$ and direction $D$, our intersection algorithm will consist of two steps:

- Finding the intersection between the line and the plane containing the triangle, by solving the equation $P(u, v) = P(s)$;
- Checking if $u$ and $v$ are within the bounds of the triangle.

Expanding the equation $P(u, v) = P(s)$, we have:

$$
\begin{align*}
A + u(B - A) + v(C - A) &= O + sD \\ 
\Rightarrow
\begin{pmatrix}
    -D & (B - A) & (C - A) \\
\end{pmatrix}
\begin{pmatrix} 
    s \\ 
    u \\ 
    v  
\end{pmatrix}
&= \begin{pmatrix} O - A \end{pmatrix} \\
\Rightarrow \begin{pmatrix} 
    -D_x & (B - A)_x & (C - A)_x \\
    -D_y & (B - A)_y & (C - A)_y \\ 
    -D_z & (B - A)_z & (C - A)_z \\
\end{pmatrix}
\begin{pmatrix}
    s \\ 
    u \\ 
    v  
\end{pmatrix} &= \begin{pmatrix}
    (O - A)_x \\ 
    (O - A)_y \\ 
    (O - A)_z \\ 
\end{pmatrix}
\end{align*}
$$

$$
$$

We can therefore find the coordinates `s`, `u`, `v` of the intersection point by solving the linear system above.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `triangle_line_intersects`

Using `torch.linalg.solve` and `torch.stack`, implement `triangle_line_intersects(A, B, C, O, D)`.

A few tips:

* If you have a 0-dimensional tensor with shape `()` containing a single value, use the `item()` method to convert it to a plain Python value.
* If your function isn't working, try making a simple ray and triangle with nice round numbers where you can work out manually if it should intersect or not, then debug from there.

```python
def triangle_line_intersects(A: t.Tensor, B: t.Tensor, C: t.Tensor, O: t.Tensor, D: t.Tensor) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    '''
    pass


if MAIN:
    tests.test_triangle_line_intersects(triangle_line_intersects)
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def triangle_line_intersects(A: t.Tensor, B: t.Tensor, C: t.Tensor, O: t.Tensor, D: t.Tensor) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    '''
    s, u, v = t.linalg.solve(
        t.stack([-D, B - A, C - A], dim=1), 
        O - A
    )
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()
```
""")
    st.markdown(r"""
## Single-Triangle Rendering

Implement `raytrace_triangle` using only one call to `torch.linalg.solve`. 

Reshape the output and visualize with `plt.imshow`. It's normal for the edges to look pixelated and jagged - using a small number of pixels is a good way to debug quickly. 

If you think it's working, increase the number of pixels and verify that it looks less pixelated at higher resolution.

### Views and Copies

It's critical to know when you are making a copy of a `Tensor`, versus making a view of it that shares the data with the original tensor. It's preferable to use a view whenever possible to avoid copying memory unnecessarily. On the other hand, modifying a view modifies the original tensor which can be unintended and surprising. Consult [the documentation](https://pytorch.org/docs/stable/tensor_view.html) if you're unsure if a function returns a view. A short reference of common functions:

- `torch.expand`: always returns a view
- `torch.view`: always returns a view
- `torch.detach`: always returns a view
- `torch.repeat`: always copies
- `torch.clone`: always copies
- `torch.flip`: always copies (different than numpy.flip which returns a view)
- `torch.tensor`: always copies, but PyTorch recommends using `.clone().detach()` instead.
- `torch.Tensor.contiguous`: returns self if possible, otherwise a copy
- `torch.transpose`: returns a view if possible, otherwise (sparse tensor) a copy
- `torch.reshape`: returns a view if possible, otherwise a copy
- `torch.flatten`: returns a view if possible, otherwise a copy (different than numpy.flatten which returns a copy)
- `einops.repeat`: returns a view if possible, otherwise a copy
- `einops.rearrange`: returns a view if possible, otherwise a copy
- Basic indexing returns a view, while advanced indexing returns a copy.

### Storage Objects

Calling `storage()` on a `Tensor` returns a Python object wrapping the underlying C++ array. This array is 1D regardless of the dimensionality of the `Tensor`. This allows you to look inside the `Tensor` abstraction and see how the actual data is laid out in RAM.

Note that a new Python wrapper object is generated each time you call `storage()`, and both `x.storage() == x.storage()` and `x.storage() is x.storage()` evaluates to False.

If you want to check if two `Tensor`s share an underlying C++ array, you can compare their `storage().data_ptr()` fields. This can be useful for debugging.

### `Tensor._base`

If `x` is a view, you can access the original `Tensor` with `x._base`. This is an undocumented internal feature that's useful to know. Consider the following code:

```python
x = t.zeros(1024*1024*1024)
y = x[0]
del x
```

Here, `y` was created through basic indexing, so `y` is a view and `y._base` refers to `x`. This means `del x` won't actually deallocate the 4GB of memory, and that memory will remain in use which can be quite surprising. `y = x[0].clone()` would be an alternative here that does allow reclaiming the memory.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `raytrace_triangle`

```python
def raytrace_triangle(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangle: TT["npoints": 3, "ndims": 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    pass


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z)
    px.imshow(img, origin="lower", labels={"x": "X", "y": "Y"}).update_layout(coloraxis_showscale=False).show()
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def raytrace_triangle(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangle: TT["npoints": 3, "ndims": 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    assert isinstance(rays, t.Tensor)
    assert isinstance(triangle, t.Tensor)

    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "p d -> p nrays d", nrays=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O = rays[:, 0]
    D = rays[:, 1]
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.T

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
```
""")
    st.markdown(r"""
## Mesh Loading

Use the given code to load the triangles for your Pikachu. By convention, files written with `torch.save` end in the `.pt` extension, but these are actually just zip files.

```python
with open("pikachu.pt", "rb") as f:
    triangles = t.load(f)
```

## Mesh Rendering

For our purposes, a mesh is just a group of triangles, so to render it we'll intersect all rays and all triangles at once. We previously just returned a boolean for whether a given ray intersects the triangle, but now it's possible that more than one triangle intersects a given ray. 

For each ray (pixel) we will return a float representing the minimum distance to a triangle if applicable, otherwise the special value `float('inf')` representing infinity. We won't return which triangle was intersected for now.
""")

    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `raytrace_mesh`

Implement `raytrace_mesh` and as before, reshape and visualize the output. Your Pikachu is centered on (0, 0, 0), so you'll want to slide the ray origin back to at least `x=-2` to see it properly.

```python
def raytrace_mesh(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangles: TT["ntriangles", "npoints": 3, "ndims": 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    pass


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma")
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
```
""")
        with st.expander("Solution"):
            st.markdown(r"""
```python
def raytrace_mesh(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangles: TT["ntriangles", "npoints": 3, "ndims": 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT p d -> p NR NT d", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR p d -> p NR NT d", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns (note, s is the distance along ray)
    sol = t.linalg.solve(mat, vec)
    s, u, v = einops.rearrange(sol, "NR NT d -> d NR NT")

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float("inf") # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values
```
""")
    st.markdown(r"""
## Bonus Content

Congratulations, you've finished the main content for today!

Some fun extensions to try:

- Vectorize further to make a video. 
    - Each frame will have its own rays coming from a slightly different position.
    - Pan the camera around for some dramatic footage. 
    - One way to do it is using the `mediapy` library to render the video.
- Try rendering on the GPU and see if you can make it faster.
- Allow each triangle to have a corresponding RGB color value and render a colored image.
- Use multiple rays per pixel and combine them somehow to have smoother edges.
""")


section_1()