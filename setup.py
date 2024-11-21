### WARNING: This setup.py should only be used when installing the minimal environment using Python 3.6 (old versions of pip)

from setuptools import setup, find_packages
import os

# Define the base directory of the package
base_dir = os.path.dirname(os.path.abspath(__file__))

# Read the __version__ from the package's __init__.py file
def get_version():
    version_file = os.path.join(base_dir, 'SPACEtomo', '__init__.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                # Extract the version number from the line
                return line.split('=')[-1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")

setup(
    name="SPACEtomo",
    version=get_version(), # Use the __version__ from __init__.py
    description="Smart Parallel Automated Cryo Electron tomography (SPACEtomo) is a package that - together with SerialEM - completely automates the cryoET data acquisition workflow.",
    long_description=open('README.md').read(), # Read the long description from README.md
    long_description_content_type="text/markdown",
    python_requires=">=3.6,!=3.7.*,!=3.8.*,!=3.10.*", # SerialEM does not support 3.10
    license=open('LICENSE').read(),
    keywords=["SPACEtomo", "cryoET", "tomography", "electron-tomography", "SerialEM"],
    maintainer="Fabian Eisenstein",
    maintainer_email="fabian.eisenstein@ethz.ch",
    packages=find_packages(), # Automatically find packages
    install_requires=[
        "matplotlib>=3.5; python_version>='3.9'",
        "mrcfile>=1.5.0; python_version>='3.9'",
        "numpy>=1.26.4; python_version>='3.9'",
        "Pillow>=10.4.0; python_version>='3.9'",
        "scipy>=1.12.0; python_version>='3.9'",
        "scikit-image>=0.22.0; python_version>='3.9'",

        "matplotlib>=3.3; python_version<'3.7'",
        "mrcfile>=1.5; python_version<'3.7'",
        "numpy>=1.19; python_version<'3.7'",
        "Pillow>=8.0; python_version<'3.7'",
        "scipy>=1.5; python_version<'3.7'",
        "scikit-image>=0.17; python_version<'3.7'",
        "imageio==2.15.0; python_version<'3.7'", # max version 2.16.0 needs numpy>=1.20
    ],
    extras_require={
        "gui": [
            "dearpygui>=1.10.0; python_version>='3.9'",
            ],
        "predict": [
            "torch>=2.2.0; python_version>='3.9'", 
            "torchvision>=0.17.0; python_version>='3.9'", 
            "nnunetv2>=2.2.1; python_version>='3.9'", 
            "ultralytics>=8.1.12; python_version>='3.9'",
            "ultralytics>=8.1.12,<8.3.10; python_version=='3.9'",
            ],
    },
    entry_points={
        "console_scripts": [
            "SPACEtomo=SPACEtomo.CLI:main",
            "SPACEmonitor=SPACEtomo.run_monitor:main",
            "SPACEmodel=SPACEtomo.setup_model:main",
            "SPACEconfig=SPACEtomo.setup_config:main",
        ]
    },
    url="https://github.com/eisfabian/SPACEtomo",
    project_urls={
        "Homepage": "https://github.com/eisfabian/SPACEtomo",
        "Documentation": "https://github.com/eisfabian/SPACEtomo/blob/main/README.md",
        "Repository": "https://github.com/eisfabian/SPACEtomo",
        "Issues": "https://github.com/eisfabian/SPACEtomo/issues",
    },
)
