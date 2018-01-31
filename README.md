[![DOI](https://zenodo.org/badge/113903219.svg)](https://zenodo.org/badge/latestdoi/113903219)

# [Scene classificator](https://github.com/godbhaal/scene-classificator)

This repository contains an implementation of an scene classificator in [Python](https://www.python.org/) using Machine Learning techniques.

The project is in the framework of [Master of Computer Vision Barcelona's](http://pagines.uab.cat/mcv) [Module 3](http://pagines.uab.cat/mcv/content/m3-machine-learning-computer-vision).

# Instructions

Place your train and test dataset using this file structure:

- data/train/\*/\*.jpg
- data/test/\*/\*.jpg

If using [Pipenv](https://docs.pipenv.org) add a file called _.env_ with the environment var
_PYTHONPATH_ set to the root of this project.

```bash
PYTHONPATH=[absolute_path_to_project_root]
```

if not is possible you have to set it in your environment using,

```
export PYTHONPATH=[absolute_path_to_project_root]`
```

# Report

The generated report and resources can be found in the folder 
[results](results) of this repository.

# About dataset

The dataset is splitted in several ways, some are redundant:

- train + validation_and_test (1881 + 807)
- train + validation + test (1881+320+487)
- train_small (400)
- train_toy (80)
