import os
from distutils.core import setup, Extension


# your complier args
cpp_args = ['/std:c++latest', '/O2']   

# your pybind11 path
includes = [
    # r"D:\software\anaconda3\src\envs\seg\Lib\site-packages\pybind11\include"
    r"E:\Anaconda3\anaconda3\envs\segmentation\Lib\site-packages\pybind11\include",
    r"E:\Anaconda3\anaconda3\envs\segmentation\Lib\site-packages\torch\include"
]

sources_root = "cppextend"
sources = [f"{sources_root}/{file}" for file in filter(lambda x: x.endswith(".cpp"), os.listdir(sources_root))]
module_name = "cpp"

ext_modules = [
    Extension(
        module_name,
        sources=sources,
        include_dirs=includes,
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name=module_name,
    version='0.0.1',
    ext_modules=ext_modules,
)
# python setup.py build_ext -i 
# stubgen -m cpp