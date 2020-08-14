# Hierarchical GP

Implementation of a hierarchical Gaussian process as in [Hensman et. al.](https://link.springer.com/article/10.1186/1471-2105-14-252) using GPFlow2. Fundamentally this this model is just a new kernel that can then be used in any of the `GPModule` inherited objects in GPFlow. The kernel can just be import from `kernel.py` and a demo notebook is also provided. It is worth running this in a virtual environment using the dependencies listed in `requirements.txt` as TensorFlow changes quite significantly from version-to-version.
