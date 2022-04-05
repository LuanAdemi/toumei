![Header](assets/header.png)

# Toumei (tōmei)

![PyTorch](https://img.shields.io/badge/pytorch-1.10.0%2B-success)
![License](https://img.shields.io/github/license/LuanAdemi/toumei.svg)
![Issues](https://img.shields.io/github/issues/LuanAdemi/toumei.svg)

Toumei is a feature visualization<sup>1</sup> library for pytorch providing a pipeline like work-flow for defining and optimizing custom objective functions. It implements every main idea presented in the distill articles and provides a user friendly interface for analyzing models.

There are multiple parameterization approaches currently implemented in toumei:
- pixel based image parameterization
- fast fourier transformation based parameterization
- ~~neural net based parameterization~~ (WIP)
- ~~GAN based parameterization~~<sup>2</sup> (WIP)

# Quick start
```python
# the model we want to inspect
alexNet = models.alexnet(pretrained=True)

# list the modules of the model
probe.print_modules(alexNet)

# build a pipeline
objective = Pipeline(
    # the image generator object
    param.Transform(param.PixelImage(1, 3, 512, 512)),

    # the objective function
    obj.Channel("features.8:12")
)

# attach the objective to the model
objective.attach(alexNet)

# optimize the objective
objective.optimize()

```

# See also
- Another great feature visualization library called <a href="https://github.com/greentfrapp/lucent">lucent</a>
- The original library (research only) <a href="https://github.com/tensorflow/lucid">lucid</a>

# References
[1] https://distill.pub/2017/feature-visualization/

[2] https://arxiv.org/abs/1605.09304
