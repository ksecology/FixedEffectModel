FixedEffectModel
=======================
This is a Python Package that provides solutions for linear model with high dimensional fixed effects, including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects.

Installation
-------------
Install this package directly from PyPI
```bash
$ pip install FixedEffectModel
```

Main Functions
-----------
|Function name| Description|Usage
|-------------|------------|----|
|ols_high_d_category|get main result|ols_high_d_category(data_df, consist_input=None, out_input=None, category_input=None, cluster_input=[],formula=None, robust=False, c_method='cgm', psdef=True, epsilon=1e-8, max_iter=1e6, process=5)|
|ols_high_d_category_multi_results|get results of multiple models based on same dataset|ols_high_d_category_multi_results(data_df, models, table_header)|
|getfe|get fixed effects|getfe(result, epsilon=1e-8)|
|alpha_std|get standard error of fixed effects|alpha_std(result, formula, sample_num=100)|


Example
----------
```python
import FixedEffectModel.api as FEM
import pandas as pd

df = pd.read_csv('yourdata.csv')

#define model:'dependent variable ~ continuous variable|fixed_effect|clusters'
formula = 'y~x+x2|id+firm|id+firm'
result1 = FEM.ols_high_d_category(df, formula = formula,robust=False,c_method = 'cgm',epsilon = 1e-8,psdef= True,max_iter = 1e6)

#show result
result1.summary()

#get fixed effects
getfe(result1 , epsilon=1e-8)

#define the expression of standard error of difference between two fixed effect estimations you want to know
expression = 'id_1-id_2'
#get standard error
alpha_std(result1, formula = expression , sample_num=100)

```



Requirements
------------
- Python 3.6+
- Pandas and its dependencies (Numpy, etc.)
- Scipy and its dependencies
- statsmodels and its dependencies

