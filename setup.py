from setuptools import setup


setup(
    name='dni-pytorch',
    version='0.1.0',
    author='Piotr Kozakowski',
    author_email='kozak000@gmail.com',
    url='https://github.com/koz4k/dni-pytorch',
    description=(
        'Decoupled Neural Interfaces using Synthetic Gradients for PyTorch'
    ),
    py_modules=['dni'],
    install_requires=[
        'torch>=0.2.0'
    ]
)
