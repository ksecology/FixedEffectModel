#This is to fix statsmodel version 0.12.1's tableformatting.py

gen_fmt = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 7,
    "colsep": '   ',
    "row_pre": '  ',
    "row_post": '  ',
    "table_dec_above": '=',
    "table_dec_below": None,
    "header_dec_below": None,
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": "r",
    "stubs_align": "l",
    "fmt": 'txt'
}

fmt_2 = {
    "data_fmts": ["%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 10,
    "colsep": ' ',
    "row_pre": '  ',
    "row_post": '   ',
    "table_dec_above": '=',
    "table_dec_below": '=',
    "header_dec_below": '-',
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": 'r',
    "stubs_align": 'l',
    "fmt": 'txt'
}
