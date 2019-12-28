import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="christopher-image-classifier", # Replace with your own username
    version="0.0.1",
    author="Ko Gi Hun",
    author_email="rhrlgns96@gmail.com",
    description="Simple Image Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nordic96/Christopher_Image_Classifier",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)