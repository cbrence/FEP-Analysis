from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="fep_analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="FEP Outcome Prediction with Cost-Sensitive Metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fep-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fep-dashboard=fep_analysis.webapp.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fep_analysis": ["config/*.yaml", "webapp/static/*"],
    },
)
