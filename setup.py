from setuptools import setup, find_packages

setup(
    name='alba_imaging',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'nilearn',
        'numpy',
        'openpyxl',
        'pandas',
        'pingouin',
        'plotly',
        'progressbar',
        'scipy',
        'seaborn',
        'scikit-learn',
        'statsmodels',
    ],
    author='Rian Bogley',
    author_email='rianbogley@gmail.com',
    description='',
)