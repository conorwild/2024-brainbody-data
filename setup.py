from setuptools import setup, find_packages

setup(
    name="brainbodydata",
    version="0.0.1",
    author="Conor J. Wild",
    author_email="conorwild@gmail.com",
    description="Dataset from the Brain & Body Study",
    packages=find_packages(),
    install_requires=[
        "cbspython @ git+https://bitbucket.org/cambridgebrainsciences/cbspython.git@main",  # noqa: E501
        "datalad",
        "datalad-dataverse",
        "datalad-osf",
        "nreporter @ git+https://github.com/conorwild/nreporter.git@v1.0.0",
        "pandas>=1.3.3",
        "pandera",
    ],
)
