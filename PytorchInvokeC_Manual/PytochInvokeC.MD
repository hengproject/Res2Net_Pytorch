# source
The raw content can be found in this page:
https://pytorch.org/tutorials/advanced/cpp_extension.html

# Steps to use C++ Extension in Pytorch

## Writing a C++ Extension

> C++ extensions come in two flavors: They can be built “ahead of time” with setuptools, or “just in time” via torch.utils.cpp_extension.load(). We’ll begin with the first approach and discuss the latter later.

### Building with `setuptools`

For the “ahead of time” flavor, we build our C++ extension by writing a setup.py script that uses setuptools to compile our C++ code. For the LLTM, it looks as simple as this:

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
cmdclass={'build_ext': cpp_extension.BuildExtension})
```

In this code, `CppExtension` is a convenience wrapper around `setuptools.Extension` that passes the correct include paths and sets the language of the extension to C++. The equivalent vanilla setuptools code would simply be:

```python
Extension(
   name='lltm_cpp',
   sources=['lltm.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
```

### Writing C++ OP
#### forward pass

#### backward pass

#### binding Python

> via https://pybind11.readthedocs.io/en/stable/ documention

