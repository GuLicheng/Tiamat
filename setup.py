# https://blog.csdn.net/weixin_44966641/article/details/123303080
from setuptools import setup, find_packages

setup(
    name="leviathan",
    version="0.0.1",
    keywords=["pip", "leviathan"],
    description="A Python library for functional programming.",
    long_description="A Python library for functional programming.",
    license="MIT License",
    url="",
    author="LichengGu",
    author_email="576356467@qq.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "statsmodels",
        "scikit-learn",
        "tensorflow",
        "torch",
        "transformers",
    ]
)