import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "dlib", "opencv-python", "numpy", "imutils", "tqdm"
]

test_requirements = [
]

setuptools.setup(
    name="FaceReco",
    version="0.1.0",
    author="1297rohit",
    author_email="1297rohit@gmail.com",
    description="Face Recognition package in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1297rohit/FaceReco",
    packages=setuptools.find_packages(),
    package_data={
        'FaceReco': ['*']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='face_recognition',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
