FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
<img width="223" alt="image" src="https://user-images.githubusercontent.com/61887305/154887555-abd074bc-e69d-4793-b850-94a0ce437b01.png">

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
*gen_data* is the function we use to simulate data. 

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
This function return two-stage least square estimation results. Define *y* as
the dependent variable, *x_1* as exogenous variable, *x_2* as endogenous variable,
*x_3* and *x_4* are instrumental variables. *id* and *time* are cross sectional 
id and time id.
An IV two-way fixed effect model estimated by two-stage least square is achieved by using:
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
The two grammars above yield identical results. 
We provide specification test for iv models:
```python
ivtest(result1)
```
Three tests are included: weak iv test (Cragg-Dolnald statistics + Stock and Yogo critical values), 
over-identification test (Sargan/Basmann test), and endogeneity test (Durbin test).

##### ivgmm
This function returns one-step gmm estimation result. With same variables definition,
 estimation is achieved by:
```python
formula = 'y ~ x_1|id+time|0|(x_2~x_3+x_4)'

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
Define *y* as
the dependent variable, *x_1* as independent variable, *id* and *time* are cross sectional 
ID and time ID.
Following code yield estimation of a two-way fixed effect model with two-way cluster
standard error:
```python
formula = 'y ~ x_1|id+time|id+time|0'

model_fe = fixedeffect(data_df = df,
                       formula = formula,
                       no_print=True)
result = model_fe.fit()
result.summary()
```
or
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
DID is simply a specific type of fixed effect model. We provide a function of DID to help 
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
"*exp_date*" is the first date that the experiment begins, "*treatment*" is the
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

We also provide two-step GMM estimator if you set thet option "gmm2=True".
Define a matrix <img src="https://latex.codecogs.com/svg.image?P_Z&space;=&space;Z(Z'Z)^{-1}Z'" title="P_Z = Z(Z'Z)^{-1}Z'" />

- "ivgmm", the one-step GMM estimator generate 
<img src="https://latex.codecogs.com/svg.image?\begin{equation}\beta_{2SLS}&space;=&space;(X'&space;P_Z&space;X)^{-1}(X'&space;P_Z&space;y)\end{equation}" title="\begin{equation}\beta_{2SLS} = (X' P_Z X)^{-1}(X' P_Z y)\end{equation}" /> with variance-covariance matrices equal
  - Unadjusted. Define <img src="https://latex.codecogs.com/svg.image?\sigma_u^2=u'u/df" title="\sigma_u^2=u'u/df" />, the variance-covariance matrix is <img src="https://latex.codecogs.com/svg.image?Var(\beta)=\sigma_u^2[X'Z(Z'Z)^{-1}Z'X]^{-1}" title="Var(\beta)=\sigma_u^2[X'Z(Z'Z)^{-1}Z'X]^{-1}" />
  - Heteroskedasticity robust. Define <img src="https://latex.codecogs.com/svg.image?\hat{\Omega}=&space;diag(\hat{u}_1^2,\dotsm,&space;\hat{u}_N^2)" title="\hat{\Omega}= diag(\hat{u}_1^2,\dotsm, \hat{u}_N^2)" /> and <img src="https://latex.codecogs.com/svg.image?W_2&space;=&space;Z'\hat{\Omega}&space;Z" title="W_2 = Z'\hat{\Omega} Z" />
  , the variance-covariance matrix is <img src="https://latex.codecogs.com/svg.image?Var_r(\beta)=[X'P_z&space;X]^{-1}[X'Z(Z'Z)^{-1}&space;W_2&space;(Z'Z)^{-1}&space;Z'X][X'&space;P_z&space;X]^{-1}" title="Var_r(\beta)=[X'P_z X]^{-1}[X'Z(Z'Z)^{-1} W_2 (Z'Z)^{-1} Z'X][X' P_z X]^{-1}" />
  - Cluster. Deine <img src="https://latex.codecogs.com/svg.image?W_2&space;=&space;\sum_g&space;Z_g'u_g&space;u_g'&space;Z_g" title="W_2 = \sum_g Z_g'u_g u_g' Z_g" />
  , the variance-covariance matrix is  <img src="https://latex.codecogs.com/svg.image?Var_c(\beta)=[X'P_z&space;X]^{-1}[X'Z(Z'Z)^{-1}&space;W_2&space;(Z'Z)^{-1}&space;Z'X][X'&space;P_z&space;X]^{-1}" title="Var_c(\beta)=[X'P_z X]^{-1}[X'Z(Z'Z)^{-1} W_2 (Z'Z)^{-1} Z'X][X' P_z X]^{-1}" />
- "ivgmm" with "gmm2=True", the two-step GMM estimator generate <img src="https://latex.codecogs.com/svg.image?\begin{equation}\beta_{GMM}&space;=&space;[X'&space;Z&space;W_2^{-1}Z'&space;X]^{-1}[X'&space;Z&space;W_2^{-1}Z'&space;Xy]\end{equation}" title="\begin{equation}\beta_{GMM} = [X' Z W_2^{-1}Z' X]^{-1}[X' Z W_2^{-1}Z' Xy]\end{equation}" />
  - Unadjusted. <img src="https://latex.codecogs.com/svg.image?Var(\beta_{GMM})&space;=&space;(X'ZW_2^{-1}Z'X)^{-1}" title="Var(\beta_{GMM}) = (X'ZW_2^{-1}Z'X)^{-1}" />
  - Heteroskedasticity robust. Define <img src="https://latex.codecogs.com/svg.image?W_3=Z'\Omega_2&space;Z" title="W_3=Z'\Omega_2 Z" />
  and <img src="https://latex.codecogs.com/svg.image?\Omega_2" title="\Omega_2" /> as the 
  diagonal matrix generated using the residual from the two-step GMM.
  , the variance-covariance matrix is <img src="https://latex.codecogs.com/svg.image?\begin{equation}Var_r(\beta_{GMM})=[X'Z(W_2)^{-1}Z'X]^{-1}&space;[X'Z(W_2)^{-1}W_3(W_2)^{-1}Z'X]&space;[X'Z(W_2)^{-1}Z'X]^{-1}\end{equation}" title="\begin{equation}Var_r(\beta_{GMM})=[X'Z(W_2)^{-1}Z'X]^{-1} [X'Z(W_2)^{-1}W_3(W_2)^{-1}Z'X] [X'Z(W_2)^{-1}Z'X]^{-1}\end{equation}" />
  - Cluster. Define 

  <img src="https://latex.codecogs.com/svg.image?\begin{aligned}&W_2&space;=&space;\sum_g&space;Z_g'u_g&space;u_g'&space;Z_g\\&W_3&space;=&space;\sum_g&space;Z_g'u_{2,g}&space;u_{2,g}'&space;Z_g\\\end{aligned}" title="\begin{aligned}&W_2 = \sum_g Z_g'u_g u_g' Z_g\\&W_3 = \sum_g Z_g'u_{2,g} u_{2,g}' Z_g\\\end{aligned}" />

  , the variance-covariance matrix is <img src="https://latex.codecogs.com/svg.image?\begin{equation}Var_c(\beta_{GMM})=[X'Z(W_2)^{-1}Z'X]^{-1}&space;[X'Z(W_2)^{-1}W_3(W_2)^{-1}Z'X]&space;[X'Z(W_2)^{-1}Z'X]^{-1}\end{equation}" title="\begin{equation}Var_c(\beta_{GMM})=[X'Z(W_2)^{-1}Z'X]^{-1} [X'Z(W_2)^{-1}W_3(W_2)^{-1}Z'X] [X'Z(W_2)^{-1}Z'X]^{-1}\end{equation}" />.


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
import numpy as np
import pandas as pd
from fixedeffect.iv import iv2sls, ivgmm,ivtest
from fixedeffect.fe import fixedeffect, did,getfe
from fixedeffect.utils.panel_dgp import gen_data 
from fixedeffect.iv import ivtest

N = 100
T = 10
beta = [-3,1,2,3,4]
ate = 1
exp_date = 5

#generate sample data
df = gen_data(N, T, beta, ate, exp_date)

#------------------------------#
#define instrumental variable model
# iv2sls 
formula = 'y ~ x_1|id+time|0|(x_2~x_3+x_4)'
model_iv2sls = iv2sls(data_df = df,
                      formula = formula)
result = model_iv2sls.fit()
result.summary()

# ivgmm 
formula = 'y ~ x_1|id|0|(x_2~x_3+x_4)'

model_ivgmm = ivgmm(data_df = df,
                    formula = formula)
result = model_ivgmm.fit()
result.summary()

# obtain iv test results
ivtest(result)

#------------------------------#

#define fixed effect model
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

#obtain fixed effect 
getfe(result)

#------------------------------#
#define DID model
formula = 'y ~ 0|0|0|0'

model_did = did(data_df = df,
                formula = formula,
                treatment = ['treatment'],
                csid = ['id'],
                tsid = ['time'],
                exp_date=2)
result = model_did.fit()
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
