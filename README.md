![Header](assets/header.png)

# Toumei (tōmei) (WIP)
Toumei is a feature visualization<sup>1</sup> library for pytorch providing a pipeline like work-flow for defining and optimizing custom objective functions. It implements every main idea presented in the distill articles and provides a user friendly interface for analyzing models.

There are multiple parameterization approaches currently implemented in toumei:
- pixel based image parameterization
- fast fourier transformation based parameterization
- neural net based parameterization
- GAN based parameterization<sup>2</sup>

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
fv.optimize()

```

# References
[1] https://distill.pub/2017/feature-visualization/

[2] https://arxiv.org/abs/1605.09304
