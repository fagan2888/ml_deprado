# Lumberjack BTC Trader

A Machine Learning System for Trading Bitcoin ("Lumberjack" because it utilizes Random Forests)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See Installing and Deploying for notes on how to deploy the project on a live system.

### Prerequisites

There are some prerequisite modules that need to be installed in your python environment. They are:

```text
matplotlib==3.0.2
missingno==0.4.1
numpy==1.14.3
pandas==0.23.4
seaborn==0.9.0
scikit-learn==0.19.1
scipy==1.1.0
TA-Lib==0.4.17
tqdm==4.27.0
```

They can be installed using the included requirements.txt file.

```python
pip install -r requirements.txt
```

### Installing and Deploying

You will need:

* a Jupyter notebook environment - see [Installing Jupyter](http://jupyter.org/install.html)
* access to this folder and its subfolders from your Jupyter environment

In your Jupyter environment, navigate to the folder **notebooks** and open the notebook named **1.0-bk-machine-learning-new.ipynb**.
Alternatively, you can simply open the html version, **1.0-bk-machine-learning-new.html** and view the output.

A raw data file consisting of 1 min BTC (Bitcoin) data since 2014 is needed in **data/raw**, the file is named **coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv** and can be obtained from [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data#coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv)

### File Structure

```text
.
|____notebooks
| |____1.0-bk-machine-learning-new.ipynb
| |____1.0-bk-machine-learning-new.html
| |____.ipynb_checkpoints
| | |____1.0-bk-machine-learning-new-checkpoint.ipynb
|____requirements.txt
|____README.md
|____documents
| |____solutions_design_report_team_1b_part3.pdf
| |____fund_factsheet_team_1b_part3.pdf
| |____solutions_document_team_1b_part3.pdf
|____modules
| |____technical_analysis.py
| |____deprado.py
| |______init__.py
| |____bars_tools.py
| |______pycache__
| | |____technical_analysis.cpython-36.pyc
| | |____bars_tools.cpython-36.pyc
| | |____create_bars.cpython-36.pyc
| | |____deprado.cpython-36.pyc
| | |____plot_roc_curve.cpython-36.pyc
| | |______init__.cpython-36.pyc
| |____plot_roc_curve.py
|____data
| |____processed
| |____raw
```

NB: The notebook stores some csv data files in the **data/processed** folder: the cleaned up and processed 1 min bitcoin data file, the generated dollar bar ohlc file, and finally the results.csv file which contains our returns information.

The data files have been deleted as they result in the zip file being too large for WQU's 20MB upload limit.

The PDF files for this project are stored in the **documents** folder. They have descriptive names:

    * solutions_document_team_1b_part3.pdf
    * solutions_design_report_team_1b_part3.pdf
    * fund_factsheet_team_1b_part3.pdf

**Please note the last page (Page 40) in the Solutions Design Report is intentionally left blank.**

NOTE: Documents are excluded.

## Built With

* [Python](https://www.python.org) - The Programming Language
* [Jupyter](http://jupyter.org) - The Jupyter Notebook Environment
* [Pandas](https://pandas.pydata.org) - Data Analysis
* [Numpy](http://www.numpy.org) - Scientific Computing
* [Scipy](https://www.scipy.org) - Mathematics, Statistics, Engineering Library
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine Learning Library


## Versioning

The current version is 1.0

## Authors

* **JBK**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Marcos Lopez de Prado for his book
* Jacques Joubert, BlackArbsCEO
* Various Kagglers
* Numerous Google searches
