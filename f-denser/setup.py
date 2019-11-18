import setuptools

with open("../readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Fast-DENSER", # Replace with your own username
    version="1.1.0",
    author="Filipe Assuncao and Nuno Lourenco",
    author_email="fga@dei.uc.pt",
    description="Fast Deep Evolutionary Network Structured Representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/fillassuncao/f-denser",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=2.7',
)
