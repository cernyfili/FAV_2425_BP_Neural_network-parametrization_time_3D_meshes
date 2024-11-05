# setup.py

from setuptools import setup, find_packages

setup(
    name="my_project",                         # Package name
    version="0.1.0",                           # Version of your project
    description="A brief description of my project",
    long_description=open("README.md").read(), # Long description from README file
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/username/my_project",  # URL to the project's homepage
    license="MIT",                             # License type (e.g., MIT, Apache)

    # Packages and Modules
    packages=find_packages("src"),             # Use src folder as the base for packages
    package_dir={"": "src"},                   # Tell setuptools that the packages are in the "src" directory

    # Dependencies
    install_requires=[
    ],#TODO

    # Optional dependencies for extra features
    extras_require={
        "dev": [],  # Development dependencies
        "docs": [],               # Documentation dependencies
    },

    # Metadata for PyPI and search keywords
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",                    # Minimum Python version required

    # Include data files
    include_package_data=True,                  # If data files are specified in MANIFEST.in
    package_data={
        "": [],    # Include data and config files
    },
    entry_points={
        "console_scripts": [
            "my_project=my_project.main:main",  # Creates a command-line script
        ],
    },
)
