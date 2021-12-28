from statsmodels.compat import lrange
from statsmodels.iolib import SimpleTable
import numpy as np


def fit_multi_model(models,
                    table_header=None):
    """
    This function is used to get multi results of multi models on one dataframe. During analyzing data with large data
    size and complicated, we usually have several model assumptions. By using this function, we can easily get the
    results comparison of the different models.

    :param models: List of models
    :param table_header: Title of summary table
    :return: summary table of results of the different models
    """
    results = []


    for model_i in models:
        results.append(model_i.fit())


    consist_name_list = [result.params.index.to_list() for result in results]
    consist_name_total = []
    consist_name_total.extend(consist_name_list[0])
    for i in consist_name_list[1:]:
        for j in i:
            if j not in consist_name_total:
                consist_name_total.append(j)
    index_name = []
    for name in consist_name_total:
        index_name.append(name)
        index_name.append('std err')
        index_name.append('pvalue')

    exog_len = lrange(len(results))
    lzip = []
    y_zip = []
    b_zip = np.zeros(5)
    table_content = []
    for name in consist_name_total:
        coeff_list = []
        pvalue_list = []
        std_list = []
        for i in range(len(results)):
            if name in consist_name_list[i]:
                coeff = "%#7.4g" % (results[i].params[name])
                std = "%#8.4f" % (results[i].bse[consist_name_list[i].index(name)])
                pvalue = "%#8.2g" % (results[i].pvalues[name])
                coeff_list.append(coeff)
                pvalue_list.append(pvalue)
                std_list.append(std)
            else:
                coeff = 'Nan'
                pvalue = 'Nan'
                std = 'Nan'
                coeff_list.append(coeff)
                pvalue_list.append(pvalue)
                std_list.append(std)
        table_content.append(tuple(coeff_list))
        table_content.append(tuple(std_list))
        table_content.append(tuple(pvalue_list))

    wtffff = dict(
        fmt='txt',
        # basic table formatting
        table_dec_above='=',
        table_dec_below='-',
        title_align='l',
        # basic row formatting
        row_pre='',
        row_post='',
        header_dec_below='-',
        row_dec_below=None,
        colwidths=None,
        colsep=' ',
        data_aligns="l",
        # data formats
        # data_fmt="%s",
        data_fmts=["%s"],
        # labeled alignments
        # stubs_align='l',
        stub_align='l',
        header_align='r',
        # labeled formats
        header_fmt='%s',
        stub_fmt='%s',
        header='%s',
        stub='%s',
        empty_cell='',
        empty='',
        missing='--',
    )
    a = SimpleTable(table_content,
                    table_header,
                    index_name,
                    title='multi',
                    txt_fmt=wtffff)
    print(a)
