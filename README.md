FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
**FixedEffectModel** is a Python Package designed and built by **Kuaishou DA ecology group**. It provides solutions for linear model with high dimensional fixed effects,including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects.It also supports model with instrument variables.

# Installation

Install this package directly from PyPI
```bash
$ pip install FixedEffectModel
```

# Main Functions

|Function name| Description|Usage
|-------------|------------|----|
|ols_high_d_category|get main result|ols_high_d_category(data_df, consist_input=None, out_input=None, category_input=None, cluster_input=[],fake_x_input=[], iv_col_input=[], formula=None, robust=False, c_method='cgm', psdef=True, epsilon=1e-8, max_iter=1e6, process=5)|
|ols_high_d_category_multi_results|get results of multiple models based on same dataset|ols_high_d_category_multi_results(data_df, models, table_header)|
|getfe|get fixed effects|getfe(result, epsilon=1e-8)|
|alpha_std|get standard error of fixed effects|alpha_std(result, formula, sample_num=100)|


# Example

```python
import FixedEffectModel.api as FEM
import pandas as pd

df = pd.read_csv('yourdata.csv')

#define model
#you can define the model through defining formula like 'dependent variable ~ continuous variable|fixed_effect|clusters|(endogenous variables ~ instrument variables)'
formula_without_iv = 'y~x+x2|id+firm|id+firm'
formula_without_cluster = 'y~x+x2|id+firm|0|(Q|W~x3+x4+x5)'
formula = 'y~x+x2|id+firm|id+firm|(Q|W~x3+x4+x5)'
result1 = FEM.ols_high_d_category(df, formula = formula,robust=False,c_method = 'cgm',epsilon = 1e-8,psdef= True,max_iter = 1e6)

#or you can define the model through defining each part
consist_input = ['x','x2']
output_input = ['y']
category_input = ['id','firm']
cluster_input = ['id','firm']
endo_input = ['Q','W']
iv_input = ['x3','x4','x5']
result1 = FEM.ols_high_d_category(df,consist_input,out_input,category_input,cluster_input,endo_input,iv_input,formula=None,robust=False,c_method = 'cgm',epsilon = 1e-8,max_iter = 1e6)

#show result
result1.summary()

#get fixed effects
getfe(result1 , epsilon=1e-8)

#define the expression of standard error of difference between two fixed effect estimations you want to know
expression = 'id_1-id_2'
#get standard error
alpha_std(result1, formula = expression , sample_num=100)

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


#Reference
[1] Simen Gaure(2010).  lfe: Linear Group Fixed Effects. R package. version:v2.8-5.1 URL:https://www.rdocumentation.org/packages/lfe/versions/2.8-5.1

[2] A Colin Cameron and Douglas L Miller. A practitioner’s guide to cluster-robust inference. Journal of human resources, 50(2):317–372, 2015.

[3] Simen Gaure. Ols with multiple high dimensional category variables. Computational Statistics & Data Analysis, 66:8–18, 2013.

[4] Douglas L Miller, A Colin Cameron, and Jonah Gelbach. Robust inference with multi-way clustering. Technical report, Working Paper, 2009.

[5] Jeffrey M Wooldridge. Econometric analysis of cross section and panel data. MIT press, 2010.
