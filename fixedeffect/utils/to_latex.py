def to_latex(f_result, file=None):
    if file is None:
        file = 'output.tex'
    with open(file, 'w') as fh:
        fh.write(f_result.general_table.as_latex_tabular())
        fh.write('\n')
        fh.write(f_result.parameter_table.as_latex_tabular())
