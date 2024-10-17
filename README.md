# # Characterizing the Cognitive and Mental Health Benefits of Exercise and Video Game Playing: The Brain and Body Study.

Conor J. Wild<sup>1,2*</sup>, Sydni G. Paleczny<sup>1*</sup>, Alex Xue<sup>2</sup>, Roger Highfield<sup>4</sup>, Adrian M. Owen<sup>1,2,3</sup>

1. Western Institute for Neuroscience, Western University, Ontario, Canada
2. Department of Physiology and Pharmacology, Western University, Ontario, Canada
3. Department of Psychology, Western University, Ontario, Canada
4. Science Museum Group, United Kingdom

\* Co-first authors

Corresponding authors: Conor J. Wild (cwild@uwo.ca), Sydni G. Paleczny (spaleczn@uwo.ca)

# Abstract
Two of the most actively studied modifiable lifestyle factors, exercise and video gaming, are regularly touted as easy and effective ways to enhance brain function and/or protect it from age-related decline. However, some critical lingering questions and methodological inconsistencies leave it unclear what aspects of brain health are affected by exercise and video gaming, if any at all. In a global online study of over 1000 people, we collected data about participants' physical activity levels, time spent playing video games, mental health, and cognitive performance using tests of short-term memory, verbal abilities, and reasoning skills from the Creyos battery. The amount of regular physical activity was not significantly related to any measure of cognitive performance; however, more physical activity was associated with better mental health as indexed using the PHQ-2 and GAD-2 screeners for depression and anxiety. Conversely, we found that more time spent playing video games was associated with better cognitive performance but was unrelated to mental health. We conclude that exercise and video gaming have differential effects on the brain, which may help individuals tailor their lifestyle choices to promote mental and cognitive health, respectively, across the lifespan.

# Description
This repository a Python package that wraps and processes the raw data for Wild & Paleczny, et al. 2024. The raw data files can be found here:

<code>Wild, Conor, 2024, "testing sandbox", https://doi.org/10.5683/SP3/WUYAGU, Borealis, DRAFT VERSION</code>

The code in this repository pulls in the data from Borealis (dataverse) and applies various preprocessing to various data source files. Installing this package into your Python environment provides an easy-to-use API for access the preprocessed data, without having to manually use datlad to fetch files, or apply any data wrangling.

To install this dataset, simply do this:
```
datalad clone osf://vq2wx/ brainbodydata
cd brainbodydata && pip3 install -e .
```

# Notes

Don't forget to enable datalad extensions:
```
git config --global --add datalad.extensions.load next
git config --global --add datalad.extensions.load dataverse
```