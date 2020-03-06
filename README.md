Models accompanying the publicaton doi: xxx
## Applying Machine Learning Methods to better understand, model and estimate mass concentrations of traffic-related pollutants at a typical street canyon
### Iva Šimić, Mario Lovrić, Ranka Godec, Mark Kröll, Ivan Bešlić


## Project structure
    .
    ├── data                 # Data for modelling
    ├── results              # Results 
    ├── scripts              # Automated tests and run as .py
    ├── src                  # Source, models, tools, utilities
    ├── LICENSE
    └── README.md
    
______________________________________________
The code is set up as follows:

> `src` has all the modules necessary for modelling

> `src/config.py` has the configurations used by `src/models.py`
> `src/preprocessing.py` is used by `src/models.py` to preprocess the data partially
> `src/feat_utils.py` has some supporting functions and data
> `scripts/run.py` is the script for running the code
______________________________________________
The original data can be downloaded from:
https://zenodo.org/record/3694131

A preprocessed and imputed data files is present in the "data" folder
* `data/preprocessed.csv`          |

______________________________________________
## Running it

For running the script, a conda environment is recommended (or other Python distributions).

Conda installation:
`https://docs.conda.io/projects/conda/en/latest/user-guide/install/`

Once Conda is installed the environment for running this script can be created as follows:
`conda create -n envpol python=3.6 scikit-learn=0.22 eli5 numpy pandas`

In the script folder, the experiment 
`python run.py`


<<<<<<< HEAD
=======
* TB
>>>>>>> ae4f42d49f63d4bfd2922619b81e279eacfb7194
