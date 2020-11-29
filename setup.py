import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="power-bdd",
    version="0.0.1",
    install_requires=['sortedcontainers >= 2.0'],
    author="Ingo Wilms",
    author_email="ingo.wilms@gmail.com",
    description="Calculates power indices (Banzhaf/Penrose, Shapley/Shubik) via binary decision diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oceanwhiskey/power_bdd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    py_modules=['power_bdd.node', 'power_bdd.bdd'],
    python_requires='>=3.6',
    license='LICENCE.txt',
)
