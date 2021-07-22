FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
<img width="150" src="https://user-images.githubusercontent.com/61887305/126601384-35ab02e9-447c-4977-8d29-aa89397727bd.png" alt="Logo" />



[![Downloads](https://pepy.tech/badge/fixedeffectmodel)](https://pepy.tech/project/fixedeffectmodel)
[![Downloads](https://pepy.tech/badge/fixedeffectmodel/month)](https://pepy.tech/project/fixedeffectmodel)
[![Downloads](https://pepy.tech/badge/fixedeffectmodel/week)](https://pepy.tech/project/fixedeffectmodel)



**FixedEffectModel** is a Python Package designed and 
built by **Kuaishou DA ecology group**. It is used to estimate the class of 
linear models which handles panel data. Panel data refers to the type of data
when time series and cross-sectional data are combined. 

# Main Features
*   Linear model 
*   Linear model with high dimensional fixed effects
*   Difference-in-difference model with parallel checking plot
*   Instrumental variable model 
*   Robust/white standard error 
*   Multi-way cluster standard error
*   Instrumental variable model tests, including weak iv test (cragg-dolnald statistics+stock and yogo critical values), over-identification test (sargan/Basmann test), endogeneity test (durbin test)


# Installation

Install this package directly from PyPI
```bash
$ pip install FixedEffectModel
```

# Getting started
This very simple case-study is designed to get you up-and-running quickly with fixedeffectmodel. 
We will show the steps needed. 

### Loading modules and functions

After installing statsmodels and its dependencies, we load a few modules and functions:
```python
import pandas
from FixedEffectModel.api import *
from utils.panel_dgp import gen_data
```
utils.panel_dgp is the function we use to simulate data.

### Data

We use a simulated dataset with 100 cross-sectional units and 10 time units.

```python
N = 100
T = 10
beta = [-3,-1.5,1,2,3,4,5] 
ate = 1 
exp_date = 2

df = gen_data(N, T, beta, ate, exp_date)
```
Ihe the above simulated dataset, "beta" are true coefficients, 
"ate" is the true treatment effect, 
"exp_date" is the start date of experiment.

### Model fit and summary

The estimation is achieved by:

```python
consist_input = ['x_1','x_2']
out_input = ['y']
category_input = ['id','time']
cluster_input = ['id','time']
endo_input = ['x_3','x_4']
iv_input = ['x_5','x_6']
result1 = ols_high_d_category(df,
                              consist_input,
                              out_input,
                              category_input,
                              cluster_input,
                              endo_input,
                              iv_input,
                              formula=None,
                              robust=False,
                              c_method = 'cgm',
                              epsilon = 1e-8,
                              max_iter = 1e6)

#show result
result1.summary()

#get fixed effects
getfe(result1)
```
In the function above, we run a fixed effect iv model with clustered standard error.
To obtain fixed effects, you can run:
```python
getfe(result1)
```

### Diagnostics and specification tests
We provide specification test for iv models:
```python
ivtest(result1)
```
# Main Functions
Currently there are five main function you can call:

|Function name| Description|Usage
|-------------|------------|----|
|ols_high_d_category|get main result|ols_high_d_category(data_df,consist_input=None,out_input=None,category_input=[],cluster_input=[],fake_x_input=[],iv_col_input=[],treatment_input=None,formula=None,robust=False,c_method='cgm',psdef=True,epsilon=1e-08,max_iter=1e6,process=5,noint=False,**kwargs,)|
|ols_high_d_category_multi_results|get results of multiple models based on same dataset|ols_high_d_category_multi_results(data_df, models, table_header)|
|getfe|get fixed effects|getfe(result, epsilon=1e-08, normalize=False, category_input=[])|
|alpha_std|get standard error of fixed effects|alpha_std(result, formula, sample_num=100)|
|ivtest|if specified an iv model in ols_high_d_category, provide iv test result|ivtest(result)

### ols_high_d_category
The main estimation function, provide results for a single model.

|Input parameters| Type| Description
|--------|------------------|----|
|data_df|pandas dataframe| Dataframe with relevant data.|
|consist_input|list, default []|List object of independent variables|
|out_input|list|List object of dependent variables|
|category_input|list, default []|List object of category variables, i.e, fixed effect|
|cluster_input|list, default []|List object of cluster variables, i.e, the cluster level of your standard error|
|fake_x_input|list, default []|List object of endogenous independent variables.|
|iv_col_input|list, default []|List object of instrumental variables.|
|treatment_input|dict, default {}|Dict object of information to estimate difference-in-difference model. The input format is <em>treatment_input ={'treatment_col':'fake_treatment','exp_date':'2020-01-07','effect':'group'}</em>|
|formula|str, default None|Alternative format option which allows you to input variables above.|
|robust|bool, default False| Whether or not to calculate df-adjusted white standard error (HC1)|
|c_method|str, default 'cgm'| Method to calculate multi-cluster standard error. Possible choices are 'cgm' and 'cgm2'.| 
|psdef|bool, default True|if True, replace negative eigenvalue of variance matrix with 0 (only in multi-way clusters variance)|
|epsilon|double, default 1e-8|tolerance of the demean process|
|max_iter|int, default 1e6|max iteration of the demean process|
|noint|bool, default True|Whether or not generate intercept|

Return an object of results:

|Attribute| Type
|--------|------------------|
|params| Estimated coefficients| 
|df| Degree of freedom.|
|bse| standard error|
|variance_matrix| coefficients' variance-covariance matrix|

### ols_high_d_category_multi_results

This function is used to get multi results of multi models on one dataframe. During analyzing data with large data
size and complicated, we usually have several model assumptions. By using this function, we can easily get the
results comparison of the different models.


|Input parameters| Type| Description
|--------|------------------|----|
|data_df|pandas dataframe| Dataframe with relevant data|
|models|list, default []| List of models|
|table_header|str, default None| Title of summary table|

Return a summary table of results of the different models.

### getfe
This function is used to get fixed effect.

|Input parameters| Type| Description
|--------|------------------|----|
|result|object| output object of <em>ols_high_d_category<em/> function |
|epsilon|double, default 1e-8| tolerance for projection|
|normalize|bool, default False| Whether or not to normalize fixed effects.|
|category_input|list, default []| List of category variables to calculate fixed effect.|

Return a summary table of estimates of fixed effects and its standard errors.

### ivtest
This function is used to obtain iv test result.

|Input parameters| Type| Description
|--------|------------------|----|
|result|object| output object of <em>ols_high_d_category<em/> function |

Return a test result table of iv tests.

# Example

```python
# need to install from kuaishou product base
from FixedEffectModel.api import *
from utils.panel_dgp import gen_data

N = 100
T = 10
beta = [-3,-1.5,1,2,3,4,5] 
alpha = 0.9
ate = 1 
exp_date = 2

#generate sample data
df = gen_data(N, T, beta, ate, exp_date)

#define model
#you can define the model through defining formula like 'dependent variable ~ continuous variable|fixed_effect|clusters|(endogenous variables ~ instrument variables)'
formula_without_iv = 'y~x_1+x_2|id+time|id+time'
formula_without_cluster = 'y~x_1+x_2|id+time|0|(x_3|x_4~x_5+x_6)'
formula = 'y~x_1+x_2|id+time|id+time|(x_3|x_4~x_5+x_6)'
result1 = ols_high_d_category(df, 
                              formula = formula,
                              robust=False,
                              c_method = 'cgm',
                              epsilon = 1e-8,
                              psdef= True,
                              max_iter = 1e6)

#or you can define the model through defining each part
consist_input = ['x_1','x_2']
out_input = ['y']
category_input = ['id','time']
cluster_input = ['id','time']
endo_input = ['x_3','x_4']
iv_input = ['x_5','x_6']
result1 = ols_high_d_category(df,
                              consist_input,
                              out_input,
                              category_input,
                              cluster_input,
                              endo_input,
                              iv_input,
                              formula=None,
                              robust=False,
                              c_method = 'cgm',
                              epsilon = 1e-8,
                              max_iter = 1e6)

#show result
result1.summary()

#get fixed effects
getfe(result1)



```
You can also do DID with treatment_input option:
```python
# need to install from kuaishou product base
from FixedEffectModel.api import *
from utils.panel_dgp import gen_data

N = 100
T = 10
beta = [-3,-1.5,1,2,3,4,5] 
alpha = 0.9
ate = 1 
exp_date = 2

#generate sample data
df = gen_data(N, T, beta, ate, exp_date)

#did wrt group effect
formula = 'y~0|id+time|0|0'
result = ols_high_d_category(data_df,
                             formula=formula,
                             treatment_input ={'treatment_col':'treatment',
                                               'exp_date':5,
                                               'effect':'group'})
result.summary()

#did wrt individual effect
formula = 'y~0|id+time|0|0'
result = ols_high_d_category(data_df,
                             formula=formula,
                             treatment_input ={'treatment_col':'treatment',
                                               'exp_date':5,
                                               'effect':'individual'})
result.summary()

```

# Requirements
- Python 3.6+
- Pandas and its dependencies (Numpy, etc.)
- Scipy and its dependencies
- statsmodels and its dependencies
- networkx

# Citation
If you use FixedEffectModel in your research, please cite us as follows:

Kuaishou DA Ecology. **FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.**<https://github.com/ksecology/FixedEffectModel>,2020.Version 0.x

BibTex:
```
@misc{FixedEffectModel,
  author={Kuaishou DA Ecology},
  title={{FixedEffectModel: {A Python Package for Linear Model with High Dimensional Fixed Effects}},
  howpublished={https://github.com/ksecology/FixedEffectModel},
  note={Version 0.x},
  year={2020}
}
```
# Feedback
This package welcomes feedback. If you have any additional questions or comments, please contact <da_ecology@kuaishou.com>.


# Reference
[1] Simen Gaure(2019).  lfe: Linear Group Fixed Effects. R package. version:v2.8-5.1 URL:https://www.rdocumentation.org/packages/lfe/versions/2.8-5.1

[2] A Colin Cameron and Douglas L Miller. A practitioner’s guide to cluster-robust inference. Journal of human resources, 50(2):317–372, 2015.

[3] Simen Gaure. Ols with multiple high dimensional category variables. Computational Statistics & Data Analysis, 66:8–18, 2013.

[4] Douglas L Miller, A Colin Cameron, and Jonah Gelbach. Robust inference with multi-way clustering. Technical report, Working Paper, 2009.

[5] Jeffrey M Wooldridge. Econometric analysis of cross section and panel data. MIT press, 2010.
