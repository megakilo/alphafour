"""Build script for compiling the Cython Pure Python Mode self-play engine.

Usage:
    uv run python setup_cython.py build_ext --inplace
"""

import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extra_compile_args: list[str] = []
extra_link_args: list[str] = []

if sys.platform == "win32":
    extra_compile_args.extend(["/O2", "/fp:fast"])
else:
    extra_compile_args.extend(
        [
            "-O3",
            "-march=native",
            "-fno-math-errno",
            "-fno-trapping-math",
        ]
    )
    if sys.platform.startswith("linux") and os.environ.get("ALPHAFOUR_NO_OPENMP") != "1":
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

extensions = [
    Extension(
        name="src.self_play",
        sources=["src/self_play.py"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="alphafour-self-play",
    packages=["src"],
    package_dir={"src": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "binding": True,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    ),
)
