#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup
import setuptools

setup(
    name='speech_hackathon',
    version='1.0',
    install_requires=[
        'python-Levenshtein',
        'scipy',
        'wavio'
    ]
)
