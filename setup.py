from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
import torch

import os
from typing import List, Set


ext_modules = []
cpu_extension = CppExtension(
   name='cpu_ops',
   #extra_compile_args = ['-g', '-O0'],
   sources=['quanto/library/cpu/cpu_mm_mkl.cpp'],
    )
ext_modules.append(cpu_extension)


setup(
    name="quanto",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huggingface/quanto",
    author="David Corvoysier",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
