from setuptools import setup

setup(
    name="DS Box",
    version=1.3,
    author='Vincent Levorato',
    author_email='vincent.levorato@gmail.com',
    packages=['dsbox', 'dsbox.ml', 'dsbox.operators'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
