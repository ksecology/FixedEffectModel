FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
**FixedEffectModel** is a Python Package designed and built by **Kuaishou DA ecology group**. It provides solutions for linear model with high dimensional fixed effects,including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects. It also supports model with instrument variables (will upgrade in late Nov.2020).

# Installation

Install this package directly from PyPI
```bash
$ pip install FixedEffectModel
```

# Main Functions

|Function name| Description|Usage
|-------------|------------|----|
|ols_high_d_category|get main result|ols_high_d_category(data_df,consist_input=None,out_input=None,category_input=[],cluster_input=[],fake_x_input=[],iv_col_input=[],treatment_input=None,formula=None,robust=False,c_method='cgm',psdef=True,epsilon=1e-08,max_iter=1e6,process=5,noint=False,**kwargs,)|
|ols_high_d_category_multi_results|get results of multiple models based on same dataset|ols_high_d_category_multi_results(data_df, models, table_header)|
|getfe|get fixed effects|getfe(result, epsilon=1e-08, normalize=False, category_input=[])|
|alpha_std|get standard error of fixed effects|alpha_std(result, formula, sample_num=100)|
|ivtest|if specified an iv model in ols_high_d_category, provide iv test result|ivtest(result)

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
