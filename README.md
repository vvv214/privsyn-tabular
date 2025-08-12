# PrivSyn

This repository is an implementation of paper [PrivSyn: Differentially Private Data Synthesis](https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf). 

## Introducion
The pipeline of the PrivSyn consists of three functional modules: data preprocessing, PrivSyn main process, and synthesis evaluation. The file structure can be summarized as follows.
* `data/`: used for save raw data.
* `preprocess_common/`: code for data preprocessing.
* `privsyn/`: code for the main procedure of PrivSyn.
* `evaluator/`: code for evaluation.
* `eval_models/`: this file stores the settings of evaluation models.
* `util/`: code for some helper functions.
* `exp/`: the results of experiments will be collected and save in this file. 


## Quick Start
### Hyper-paprameters
The code for running experiments is in `main.py`. The detailed description of some common hyper-parameters are give as follows.
* `method`: which synthesis method you will run.
* `dataset`: name of dataset.
* `device`: the device used for running algorithms. 
* `epsilon`: DP parameter, which must be delivered when running code. 
* `--delta`: DP parameter, which is set to $1e-5$ by default.
* `--num_preprocess`: preprocessing method for numerical attributes, which is set to uniform binning by default. 
* `--rare_threshold`: threshold of preprocessing method for categorical attributes, which is set to $0.2\%$ by default.
* `--sample_device`: device used for sample data, by default is set to the same as running device.
There are some other hyper-paramters specifically for PrivSyn main procedure in file `privsyn/privsyn.py`. We recommend using default values for these hyper-parameters.

### Run PrivSyn
Firstly, make sure the datasets are put in the correct fold (in the following examples, the fold is `data/bank`, and the necessary dataset has already been provided). In this repository, the evaluation model is already tuned so users do not need any operation. Otherwise, you should tune the evaluation model (using the following code) before any further operation. For instance, you can finetune a mlp model for evaluation like
```
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```

After preparation, we can try the following code to make an overall evaluation. Usually, we by default set `num_preprocess` to be "uniform_kbins" except for DP-MERF and TabDDPM, and set `rare_threshold` to 0.002 for overall evaluation. Therefore, if you do not want to change these settings, you do not need to include these hyper-parameters in your command line.
```
python main.py privsyn bank cuda:0 1.0
```

## PrivSyn Modules
Except for an overall implementation of PrivSyn, this repository also offers the modularized API of PrivSyn, which are `InDif selection` and `GUM`. 

### InDif selection
This is a method for marginal selection by measuring InDif. We implement it as a static method in `PrivSyn` class, called `two_way_marginal_selection` (see `privsyn/privsyn.py`). This method will return a list of 2-way marginal tuple, as the final selection. The hypermeters of this method can be summarized as 
* `df`: a dataframe of dataset
* `domain`: a dictionary of attributes domain
* `rho_indif`: privacy budget for measuring InDif
* `rho_measure`: privacy budget for measuring selected marginals (actually this budget will not be used in this phase, but works as an optimization term during selection)

You can use this static method to select marginals for other synthesis modules
```
selected_marginals = PrivSyn.two_way_marginal_selection(df, domain, rho_indif, rho_measure)
```

### GUM
We construct a class of GUM synthesis method called `GUM_Mechanisms` (see `privsyn/lib_synthesize/GUM.py`). Here is a instruction of using this closed-form synthesis module.
* `Initialization`. The initialization of GUM requires an input of hyperparameters dictionary (same as PrivSyn), dataset (Dataset class), a dictionary of 1-way marginals (used for data initialization, can be an empty dictionary), a dictionary of 2-way marginals. The dictionary of marginals should be in the form of:

    ```
    {'(attr1, attr2)': Marginal1, '(attr3, attr4)': Marginal2, ...}
    ```

    Here `Marginal1` and `Marginal2` should be in Marginal class (see `privsyn/lib_marginal/marg.py`), and measured by method `count_records`. You can initialize a GUM class like 

    ```
    model = GUM_Mechanism(args, df, dict1, dict2)
    ```

* `Main procedure`. The main procedure of GUM is finished by method `run`, which only requires the sampling number. This process includes three main steps: graph seperation, marginal consistency, and records updation. 
    ```
    model.run(n_sample = 10000)
    synthesized_df = model.synthesized_df
    ```

* `Adaptive mechanism`. We also support measurement for synthesized dataset, which can be used for adaptive marginal selection. 
    ```
    syn_vector = model.project(('attr1', 'attr2')).datavector()
    real_vector = dataset.project(('attr1', 'attr2')).datavector()
    gap = np.linalg.norm(syn_vector - real_vector, ord=1)
    ```
     