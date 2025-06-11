#!/usr/bin/env python3
from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the content of your LICENCE file
# with open('LICENCE') as f:
#     licence = f.read()

setup(
    name='PrivTrajGen',
    version='0.9.0',
    packages=find_packages(),
    url='https://github.com/erik-buchholz/CostOfTrajectoryPrivacy',
    license="CSIRO Open Source Software Licence Agreement",
    author='Erik Buchholz',
    author_email='e.buchholz@unsw.edu.au',
    description='Cost of Differential Privacy for Deep Learning-Based Trajectory Prediction',
    python_requires='>=3.12',
    install_requires=requirements  # Set the required packages from requirements.txt
)
