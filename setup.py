import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name='random_bode_generator',    # This is the name of your PyPI-package.
    version='1.1.3',
    url='https://github.com/ryanGT/random_bode_generator',
    author='Ryan Krauss',
    author_email='ryanwkrauss@gmail.com',
    description="package for generating random Bode plots intelligently",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
