# minitorch
This repository is part of my journey to learn the underlying concepts of deep learning systems through implementing a minimal version of PyTorch library.

If you're interested in learning more, I highly recommend checking out the excellent [MiniTorch lectures](https://minitorch.github.io) and [Youtube playlist](https://www.youtube.com/playlist?list=PLO45-80-XKkQyROXXpn4PfjF1J2tH46w8) by [Prof. Rush](https://rush-nlp.com), and the [self-study guide](https://github.com/mukobi/Minitorch-Self-Study-Guide-SAIA/tree/main) by [Gabriel Mukobi](https://gabrielmukobi.com) that answers some common questions.

## Setup

My venv is Python 3.8.

Install dependencies
```bash
pip install -r  requirements.txt
pip install -r  requirements.extra.txt
pip install -Ue .
conda install llvmlite
pip install altair==4.0
```

Make sure that everything is installed by running python and then checking:
```python
import minitorch
```

Make sure pre-commit enforces style and typing guidelines defined in the config file (.pre-commit-config.yaml)
```bash
pre-commit run --all-files
```

Automatically run pre-commit on git commit
```bash
pre-commit install
```

Tour of Repo:
```
.
|-- minitorch/
|   `-- core of torch libarry
|-- project/
|   `-- code for building ML using minitorch
`-- tests/
    `-- test cases for minitorch
```


To access the autograder:

* Module 0: https://classroom.github.com/a/qDYKZff9
* Module 1: https://classroom.github.com/a/6TiImUiy
* Module 2: https://classroom.github.com/a/0ZHJeTA0
* Module 3: https://classroom.github.com/a/U5CMJec1
* Module 4: https://classroom.github.com/a/04QA6HZK
* Quizzes: https://classroom.github.com/a/bGcGc12k

## Module 0 - Fundamentals

- [x] Operators
- [x] Testing and Debugging
- [x] Functional Python
- [x] Modules
- [x] Visualization

![task0.5](./figs/task0.5.png)

## Module 1 - Autodiff

- [ ] Numerical Derivatives
- [ ] Scalars
- [ ] Chain Rule
- [ ] Backpropagation
- [ ] Training

## Module 2 - Tensors

- [ ] Tensor Data - Indexing
- [ ] Tensor Broadcasting
- [ ] Tensor Operations
- [ ] Gradients and Autograd
- [ ] Training

## Module 3 - Efficiency

- [ ] Parallelization
- [ ] Matrix Multiplication
- [ ] CUDA Operations
- [ ] CUDA Matrix
- [ ] Training

## Module 4 - Networks

- [ ] 1D Convolution
- [ ] 2D Convolution
- [ ] Pooling
- [ ] Softmax and Dropout
- [ ] Extra Credit
- [ ] Training an Image Classifier
