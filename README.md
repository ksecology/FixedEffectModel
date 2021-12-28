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

For instrumental variable model, we now only provide two stage least square estimator and produce second stage regression result.
In our next release we will include GMM method and robust standard error based on GMM. 

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
import numpy as np
import pandas as pd


from fixedeffect.iv import iv2sls, ivgmm, ivtest
from fixedeffect.fe import fixedeffect, did, getfe
from fixedeffect.utils.panel_dgp import gen_data
```
gen_data is the function we use to simulate data. The function above generated
a balanced panel data set with number of cross-sectional id equals 100 and time 
id equals 10. 

### Data

We use a simulated dataset with 100 cross-sectional units and 10 time units.
```python
N = 100
T = 10
beta = [-3,1,2,3,4]
ate = 1
exp_date = 5
df = gen_data(N, T, beta, ate, exp_date)
```
Ihe the above simulated dataset, "beta" are true coefficients, 
"ate" is the true treatment effect, 
"exp_date" is the start date of experiment.

### Model fit and summary
#### Instrumental variables estimation
We include two function: "iv2sls" and "iv2gmm" for instrumental variable regression.
##### iv2sls
This function return two-stage least square estimation results. 
The estimation is achieved by:
```python
formula = 'y ~ x_1|id+time|0|(x_2~x_3+x_4)'
model_iv2sls = iv2sls(data_df = df,
                      formula = formula)
result = model_iv2sls.fit()
result.summary()
```
or
```python
exog_x = ['x_1']
endog_x = ['x_2']
iv = ['x_3','x_4']
y = ['y']

model_iv2sls = iv2sls(data_df = df,
                      dependent = y,
                      exog_x = exog_x,
                      endog_x = endog_x,
                      category = ['id','time'],
                      iv = iv)

result = model_iv2sls.fit()
result.summary()
```
You can obtain estimation result using either grammar above. 
We provide specification test for iv models:
```python
ivtest(result1)
```
Three tests are included: weak iv test (Cragg-Dolnald statistics + Stock and Yogo critical values), 
over-identification test (Sargan/Basmann test), and endogeneity test (Durbin test).

##### ivgmm
This function returns one-step gmm estimation result. 
The estimation is achieved by:
```python
formula = 'y ~ x_1|id|0|(x_2~x_3+x_4)'

model_ivgmm = ivgmm(data_df = df,
                    formula = formula)
result = model_ivgmm.fit()
result.summary()
```
or
```python
exog_x = ['x_1']
endog_x = ['x_2']
iv = ['x_3','x_4']
y = ['y']

model_ivgmm = ivgmm(data_df = df,
                      dependent = y,
                      exog_x = exog_x,
                      endog_x = endog_x,
                      category = ['id','time'],
                      iv = iv)

result = model_ivgmm.fit()
result.summary()
```
#### Fixed Effect Model
This function returns fixed effect model estimation result. 
The estimation is achieved by:
```python
formula = 'y ~ x_1|id+time|id+time|0'

model_fe = fixedeffect(data_df = df,
                       formula = formula,
                       no_print=True)
result = model_fe.fit()
result.summary()
```
Sample code above estimate a two-way fixed effect model with cluster standard
error clustering at the individual and time level.
You can also achieve the same estimation results by:
```python
exog_x = ['x_1']
y = ['y']
category = ['id','time']
cluster = ['id','time']


model_fe = fixedeffect(data_df = df,
                      dependent = y,
                      exog_x = exog_x,
                      category = category,
                      cluster = cluster)

result = model_fe.fit()
result.summary()
```
#### Difference in Difference
DID is simply a specific fixed effect model. We provide a function of DID to help 
simplify the estimation process. The regular DID estimation is achieved using 
following command:
```python
formula = 'y ~ 0|0|0|0'

model_did = did(data_df = df,
                formula = formula,
                treatment = ['treatment'],
                csid = ['id'],
                tsid = ['time'],
                exp_date = 2)
result = model_did.fit()
result.summary()
```
"exp_date" is the first date that the experiment begins, "treatment" is the
column name of the treatment variable. This command estimate the equation below:

<img src="https://latex.codecogs.com/svg.image?y_{it}&space;=&space;Treat_i&space;Post_t&space;\beta_1&space;&plus;&space;&space;Treat_i\beta_2&space;&plus;&space;Post_t&space;\beta_3&space;&plus;&space;\varepsilon_{it}" title="y_{it} = Treat_i Post_t \beta_1 + Treat_i\beta_2 + Post_t \beta_3 + \varepsilon_{it}" />

We also provide DID with individual effect:
```python
formula = 'y ~ 0|0|0|0'

model_did = did(data_df = df,
                formula = formula,
                treatment = ['treatment'],
                group_effect='individual',
                csid = ['id'],
                tsid = ['time'],
                exp_date = 2)
result = model_did.fit()
result.summary()
```
This command above estimate the equation below:
<img src="https://latex.codecogs.com/svg.image?y_{it}&space;=&space;\beta_0&space;&plus;&space;\beta_2&space;Treat_i*post_t&space;&plus;&space;user_i&plus;&space;date_t&space;&plus;&space;\varepsilon_{it}" title="y_{it} = \beta_0 + \beta_2 Treat_i*post_t + user_i+ date_t + \varepsilon_{it}" />


# Main Functions
Currently there are five main function you can call:

|Function name| Description|Usage
|-----------|--------------|----|
|fixedeffect|define class for fixed effect estimation|fixedeffect (data_df = None, dependent = None, exog_x = None, category = None, cluster = None, formula = None, robust = False, noint = False, c_method = 'cgm', psdef = True)|
|iv2sls|define class for 2sls estimation|iv2sls (data_df = None, dependent = None, exog_x = None, endog_x = None, iv = None, category = None, cluster = None, formula = None, robust = False, noint = False)|
|ivgmm|define class for gmm estimation|ivgmm (data_df = None, dependent = None, exog_x = None, endog_x = None, iv = None, category = None, cluster = None, formula = None, robust = False, noint = False)|
|did|define class for did estimation|did (data_df = None, dependent = None, exog_x = None, treatment = None, csid = None, tsid = None, exp_date = None, group_effect = 'treatment', cluster = None, formula = None, robust = False, noint = False, c_method = 'cgm', psdef = True)|
|model.fit |fit pre-defined models|result = model.fit()|
|result.summary|result.object|result.summary()
|fit_multi_model|fit multiple models|models = [model,model_did,model_iv2sls], fit_multi_model (models)|
|getfe|get fixed effects|getfe(result)|
|ivtest|get iv post estimation tests results |ivtest (result)|


### fixedeffect
Provide results for a fixed effect model:

*model = fixedeffect (data_df = None, dependent = None, exog_x = None, category = None, cluster = None, formula = None, robust = False, noint = False, c_method = 'cgm', psdef = True)*


|Input parameters| Type| Description
|--------|------------------|----|
|data_df|pandas dataframe| Dataframe with relevant data.|
|dependent|list|List object of dependent variables|
|exog_x|list|List object of independent variables|
|category|list, default []|List object of category variables, i.e, fixed effect|
|cluster|list, default []|List object of cluster variables, i.e, the cluster level of your standard error|
|formula|string, default None|Formula used to parse grammar.|
|robust|bool, default False| Whether or not to calculate df-adjusted white standard error (HC1)|
|noint|bool, default True|Whether or not generate intercept|
|c_method|str, default 'cgm'| Method to calculate multi-cluster standard error. Possible choices are 'cgm' and 'cgm2'.| 
|psdef|bool, default True|if True, replace negative eigenvalue of variance matrix with 0 (only in multi-way clusters variance)|

Return an object of results:

|Attribute| Type
|--------|------------------|
|params| Estimated coefficients| 
|df| Degree of freedom.|
|bse| standard error|
|variance_matrix| coefficients' variance-covariance matrix|

### iv2sls/ivgmm
*model = iv2sls (data_df = None, dependent = None, exog_x = None, endog_x = None, iv = None, category = None, cluster = None, formula = None, robust = False, noint = False)*

*model = ivgmm (data_df = None, dependent = None, exog_x = None, endog_x = None, iv = None, category = None, cluster = None, formula = None, robust = False, noint = False)*

|Input parameters| Type| Description
|--------|------------------|----|
|data_df|pandas dataframe| Dataframe with relevant data.|
|dependent|list|List object of dependent variables|
|exog_x|list|List object of exogenous variables|
|endof_x|list|List object of endogenous variables|
|iv|list|List object of instrumental variables|
|category|list, default []|List object of category variables, i.e, fixed effect|
|formula|string, default None|Formula used to parse grammar.|
|robust|bool, default False| Whether or not to calculate df-adjusted white standard error (HC1)|
|noint|bool, default True|Whether or not generate intercept|

Return the same object of results as fixedeffect does.

### DID
*model = did (data_df = None, dependent = None, exog_x = None, treatment = None, csid = None, tsid = None, exp_date = None, group_effect = 'treatment', cluster = None, formula = None, robust = False, noint = False, c_method = 'cgm', psdef = True)*

|Input parameters| Type| Description
|--------|------------------|----|
|data_df|pandas dataframe| Dataframe with relevant data.|
|dependent|list|List object of dependent variables|
|exog_x|list|List object of independent variables|
|treatment|list|List object of treatment variables|
|csid|list|List object of cross sectional id variables|
|tsid|list|List object of time variables|
|exp_date|string|Experiment start date|
|group_effect|string, default 'treatment'|Either equals 'treatment' or 'individual'|
|cluster|list, default []|List object of cluster variables, i.e, the cluster level of your standard error|
|formula|string, default None|Formula used to parse grammar.|
|robust|bool, default False| Whether or not to calculate df-adjusted white standard error (HC1)|
|noint|bool, default True|Whether or not generate intercept|
|c_method|str, default 'cgm'| Method to calculate multi-cluster standard error. Possible choices are 'cgm' and 'cgm2'.| 
|psdef|bool, default True|if True, replace negative eigenvalue of variance matrix with 0 (only in multi-way clusters variance)|

Return the same object of results as fixedeffect does.


### fit_multi_model
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
|result|object| output object of <em>fixedeffect<em/> function |
|epsilon|double, default 1e-8| tolerance for projection|
|normalize|bool, default False| Whether or not to normalize fixed effects.|
|category_input|list, default []| List of category variables to calculate fixed effect.|

Return a summary table of estimates of fixed effects and its standard errors.

### ivtest
This function is used to obtain iv test result.

|Input parameters| Type| Description
|--------|------------------|----|
|result|object| output object of <em>ivgmm/iv2sls<em/> function |

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
