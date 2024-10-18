# 2024-BRAINBODY-DATA

### Characterizing the Cognitive and Mental Health Benefits of Exercise and Video Game Playing: The Brain and Body Study.

Conor J. Wild<sup>1,2*</sup>, Sydni G. Paleczny<sup>1*</sup>, Alex Xue<sup>2</sup>, Roger Highfield<sup>4</sup>, Adrian M. Owen<sup>1,2,3</sup>

1. Western Institute for Neuroscience, Western University, Ontario, Canada
2. Department of Physiology and Pharmacology, Western University, Ontario, Canada
3. Department of Psychology, Western University, Ontario, Canada
4. Science Museum Group, United Kingdom

\* Co-first authors

Corresponding authors: Conor J. Wild (cwild@uwo.ca), Sydni G. Paleczny (spaleczn@uwo.ca)

# Description

This repository contains a Datalad dataset that implements a Python package to wrap and processes the raw data for Wild & Paleczny, et al. 2024. The raw data files are stored separately, and can be found here:

<code>Wild, Conor; Xue, Alex; Paleczny, Sydni; Highfield, Roger; Owen, Adrian, 2024, "The Brain & Body Study - complete raw dataset", https://doi.org/10.5683/SP3/WUYAGU, Borealis, DRAFT VERSION, UNF:6:2w6TmV63h3y9/GNUox7O1A== [fileUNF]</code>

The purpose of *this* package is to provide an easy-to-use API for working with these data in a Python notebook. Using Datalad, this API will automatically clone the data raw rom Borealis (dataverse). Once installed, the classes exposed by this API will automatically load, preprocess, and score the various data source files, producing clean [Pandas](https://pandas.pydata.org/) dataframes that you can use for your analses. Data preprocessing uses [Pandera](https://pandera.readthedocs.io/en/stable/) to validate dataframes and convert data types (e.g., converting strings into categorical variables). You can use this package to enable your own (Python) analyses of these data! If you use R, SPSS, or something else to analyze data then you will have to start with the raw data files (see Borealis link).

# Requirements

You need to have [Datalad](https://www.datalad.org/), [datalad-dataverse](https://docs.datalad.org/projects/dataverse/en/latest/index.html), and [datalad-osf](http://docs.datalad.org/projects/osf/en/latest/) installed in order to clone and use this dataset. Assuming that you have activate a Python virtual environment or a Conda environment:

```
~$ pip3 install datalad datalad-osf datalad-dataverse
~$ git config --global --add datalad.extensions.load next
~$ git config --global --add datalad.extensions.load dataverse

~$ datalad clone osf://vq2wx/ brainbodydata
~$ cd brainbodydata
~/brainbodydata/$ pip3 install -e .
```

# Accessing the Data

Basically, there are four main classes provided by this package:

`BBCogsData` - the Creyos cognitive test score data in intermediate, and pre-processed forms.
`BBMasterList` - a de-identified list of user identifiers.
`BBQuestionnaire` - the demographic and health questionnaire, including GAD-2 and PHQ-2
`BBRawCogsData` - the raw Creyos data.
`PAAQ` - the Physical Activity Adult Questionnaire data.
`VGQ` - the Video Game Questionnaire data.

Instantiating an object from one of these classes will clone the raw data file(s) (if required), load the data into a Pandas dataframe, and apply preprocessing. The clean dataframe can be access with the `.data` propery:

For example:

```
paaq = PAAQ(dl_rendered="disabled")  # I don't want to see the Datalad output
paaq.data.head()
```

# Notes

Also, see http://docs.datalad.org/projects/osf/en/latest/tutorial.html