#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
__author__ = 'APD'
__date__ = '2015-10-26'


def ctl_plot(data, conditions, index_order=False, mode='a', plot_mode='line'):
    import pandas as pd
    import numpy as np
    from matplotlib import rc
    rc('font', family='NanumMyeongjo')

    if mode == 'a' or mode == 'attend':
        tmp_cond = conditions.copy()
        tmp_cond.append(str(['수강인원']))
        plot_table = pd.pivot_table(data=data, columns=eval(tmp_cond[0]), index=eval(tmp_cond[1]),
                                    values=eval(tmp_cond[2]), aggfunc=np.nansum, dropna=False)

    elif mode == 'g' or mode == 'GPA':  # for GPA data
        cond_numer = conditions.copy()
        cond_denomi = conditions.copy()
        cond_numer.append(str(['학점총합']))
        cond_denomi.append(str(['학점인원']))
        numer_table = pd.pivot_table(data=data, columns=eval(cond_numer[0]), index=eval(cond_numer[1]),
                                     values=eval(cond_numer[2]), aggfunc=np.nansum, dropna=False)
        denomi_table = pd.pivot_table(data=data, columns=eval(cond_denomi[0]), index=eval(cond_denomi[1]),
                                      values=eval(cond_denomi[2]), aggfunc=np.nansum, dropna=False)
        numer_table.columns = numer_table.columns.levels[1:]
        denomi_table.columns = denomi_table.columns.levels[1:]
        plot_table = numer_table / denomi_table
        plot_table.columns = pd.MultiIndex.from_product([['성적'], plot_table.columns])

    elif mode == 'mg' or mode == 'mGPA':  # for GPA data
        cond_numer = conditions.copy()
        cond_denomi = conditions.copy()
        cond_numer.append(str(['전공평점']))
        cond_denomi.append(str(['전공학점']))
        numer_table = pd.pivot_table(data=data, columns=eval(cond_numer[0]), index=eval(cond_numer[1]),
                                     values=eval(cond_numer[2]), aggfunc=np.nansum, dropna=False)
        denomi_table = pd.pivot_table(data=data, columns=eval(cond_denomi[0]), index=eval(cond_denomi[1]),
                                      values=eval(cond_denomi[2]), aggfunc=np.nansum, dropna=False)
        numer_table.columns = numer_table.columns.levels[1:]
        denomi_table.columns = denomi_table.columns.levels[1:]
        plot_table = numer_table / denomi_table
        plot_table.columns = pd.MultiIndex.from_product([['전공성적'], plot_table.columns])

    elif mode == 'ra' or mode == 'ratio':  # for A 비율
        cond_numer = conditions.copy()
        cond_denomi = conditions.copy()
        cond_numer.append(str(['A학점인원']))
        cond_denomi.append(str(['S제외수강인원']))
        numer_table = pd.pivot_table(data=data, columns=eval(cond_numer[0]), index=eval(cond_numer[1]),
                                     values=eval(cond_numer[2]), aggfunc=np.nansum, dropna=False)
        denomi_table = pd.pivot_table(data=data, columns=eval(cond_denomi[0]), index=eval(cond_denomi[1]),
                                      values=eval(cond_denomi[2]), aggfunc=np.nansum, dropna=False)
        numer_table.columns = numer_table.columns.levels[1:]
        denomi_table.columns = denomi_table.columns.levels[1:]
        plot_table = 100 * numer_table / denomi_table


    elif mode == 'c' or mode == 'cancel':
        tmp_cond = conditions.copy()
        tmp_cond.append(str(['수강취소인원']))
        plot_table = pd.pivot_table(data=data, columns=eval(tmp_cond[0]), index=eval(tmp_cond[1]),
                                    values=eval(tmp_cond[2]), aggfunc=np.nansum, dropna=False)

    elif mode == 're' or mode == 'repeat':
        tmp_cond = conditions.copy()
        tmp_cond.append(str(['재수강인원']))
        plot_table = pd.pivot_table(data=data, columns=eval(tmp_cond[0]), index=eval(tmp_cond[1]),
                                    values=eval(tmp_cond[2]), aggfunc=np.nansum, dropna=False)

    if mode == 's' or mode == 'subject':
        tmp_cond = conditions.copy()
        tmp_cond.append(str(['교과목수']))
        plot_table = pd.pivot_table(data=data, columns=eval(tmp_cond[0]), index=eval(tmp_cond[1]),
                                    values=eval(tmp_cond[2]), aggfunc=np.nansum, dropna=False)

    if mode == 'l' or mode == 'lecture':
        tmp_cond = conditions.copy()
        tmp_cond.append(str(['강좌수']))
        plot_table = pd.pivot_table(data=data, columns=eval(tmp_cond[0]), index=eval(tmp_cond[1]),
                                    values=eval(tmp_cond[2]), aggfunc=np.nansum, dropna=False)
    plot_table.plot(kind=plot_mode)
    return plot_table


# select what the process will be done, i.e., attending, GPA, A ratio, cancel, repeat and so on...
def ctl_pivots(data, columns, index, mode='a', index_order=False, column_order=None, percent=False, sub_cond=None, rightmost=None):
    """
    :type data: object
    :rtype : pandas.core.frame.DataFrame
    :param data: database
    :param columns: pivot table columns
    :param index: pivot table index
    :param mode: output value (attending, GPA, A ratio and so on...)
    :param index_order: sorting index
    :param percent: add percentage values
    :return: pivot table
    """
    import pandas as pd
    import numpy as np
    # from os import linesep

    if mode in ['a', 'attend']:
        attend_values = ['수강인원']
        res_table = making_pivot(data=data, index=index, columns=columns, values=attend_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('0\n(nan%)', '0\n(0%)')

    elif mode == 'sa' or mode == 'sattend':
        attend_values = ['S수강인원']
        res_table = making_pivot(data=data, index=index, columns=columns, values=attend_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('0\n(nan%)', '0\n(0%)')

    elif mode == 'ua' or mode == 'uattend':
        attend_values = ['U수강인원']
        res_table = making_pivot(data=data, index=index, columns=columns, values=attend_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('0\n(nan%)', '0\n(0%)')

    elif mode == 'sug' or mode == 'suGPA':
        numer_values = ['S수강인원']
        denomi_values = ['U수강인원']
        numer_table = making_pivot(data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_columns = numer_table.columns
        # percentage process parts
        if percent:
            numer_table.columns = numer_table.columns.levels[1:]
            denomi_table.columns = denomi_table.columns.levels[1:]
            pd.options.display.float_format = None
            tmp_percent = (numer_table / (numer_table + denomi_table) * 100).apply(np.round, decimals=2).astype(str)
            tmp_percent = "\n(" + tmp_percent + "%)"
            res_table = numer_table.applymap("{:.10g}".format).astype(str).replace('nan', '') + tmp_percent
        elif not percent:
            res_table = numer_table
        res_table.columns = res_columns
        res_table = res_table.replace('\n(nan%)', '\n(0%)')

    elif mode == 'g' or mode == 'GPA':
        numer_values = ['학점총합']
        denomi_values = ['학점인원']
        numer_table = making_pivot(data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    mode=mode, sub_cond=sub_cond, mostright=rightmost)
        if isinstance(numer_table.index, pd.core.index.MultiIndex) or isinstance(denomi_table.index, pd.core.index.MultiIndex):
            numer_table.columns = pd.MultiIndex.from_product([['성적'], numer_table.columns.levels[1]])
            denomi_table.columns = pd.MultiIndex.from_product([['성적'], denomi_table.columns.levels[1]])
        elif not isinstance(numer_table.index, pd.core.index.MultiIndex) and not isinstance(denomi_table.index, pd.core.index.MultiIndex):
            tmp_numerindex = numer_table.index.delete(-1)
            numer_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            denomi_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            numer_table.columns = pd.MultiIndex.from_product([['성적'], numer_table.columns.levels[1][:-1].insert(len(numer_table.columns.levels[1][:-1]),'평균')])
            denomi_table.columns = pd.MultiIndex.from_product([['성적'], denomi_table.columns.levels[1][:-1].insert(len(denomi_table.columns.levels[1][:-1]),'평균')])
        res_table = numer_table/denomi_table
        # percentage process parts
        if percent:
            res_table = percentile_add(res_table, unit=1, ratio='gpa')
        if column_order:
            column_order_mod = column_order.copy()
            column_order_mod.append('평균')
            res_table.columns = pd.MultiIndex.from_product([['성적'], column_order_mod])
        res_table = res_table.fillna(0)
        res_table = res_table.replace('', '')
        res_table = res_table.replace('\n(nan)', '')

    elif mode == 'mg' or mode == 'mGPA':
        numer_values = ['전공평점']
        denomi_values = ['전공학점']
        numer_table = making_pivot(data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    mode=mode, sub_cond=sub_cond, mostright=rightmost)
        if isinstance(numer_table.index, pd.core.index.MultiIndex) | isinstance(denomi_table.index, pd.core.index.MultiIndex):
            numer_table.columns = pd.MultiIndex.from_product([['전공성적'], numer_table.columns.levels[1]])
            denomi_table.columns = pd.MultiIndex.from_product([['전공성적'], denomi_table.columns.levels[1]])
        elif not isinstance(numer_table.index, pd.core.index.MultiIndex) and not isinstance(denomi_table.index, pd.core.index.MultiIndex):
            tmp_numerindex = numer_table.index.delete(-1)
            numer_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            denomi_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            numer_table.columns = pd.MultiIndex.from_product([['전공성적'], numer_table.columns.levels[1][:-1].insert(len(numer_table.columns.levels[1][:-1]),'평균')])
            denomi_table.columns = pd.MultiIndex.from_product([['전공성적'], denomi_table.columns.levels[1][:-1].insert(len(denomi_table.columns.levels[1][:-1]),'평균')])
        res_table = numer_table/denomi_table
        # percentage process parts
        if percent:
            res_table = percentile_add(res_table, unit=1, ratio='gpa')
        if column_order:
            column_order_mod = column_order.copy()
            column_order_mod.append('평균')
            res_table.columns = pd.MultiIndex.from_product([['성적'], column_order_mod])
        res_table = res_table.replace('\n(nan)', '')

    elif mode == 'ra' or mode == 'ratio':  # for A 비율
        numer_values = ['A학점인원']
        denomi_values = ['S제외수강인원']
        numer_table = making_pivot(data=data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   percent=False, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data=data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    percent=False, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        if isinstance(numer_table.index, pd.core.index.MultiIndex) | isinstance(denomi_table.index, pd.core.index.MultiIndex):
            numer_table.columns = pd.MultiIndex.from_product([['A비율'], numer_table.columns.levels[1]])
            denomi_table.columns = pd.MultiIndex.from_product([['A비율'], denomi_table.columns.levels[1]])
        elif not isinstance(numer_table.index, pd.core.index.MultiIndex) and not isinstance(denomi_table.index, pd.core.index.MultiIndex):
            tmp_numerindex = numer_table.index.delete(-1)
            numer_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            denomi_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            numer_table.columns = pd.MultiIndex.from_product([['A비율'], numer_table.columns.levels[1][:-1].insert(len(numer_table.columns.levels[1][:-1]),'평균')])
            denomi_table.columns = pd.MultiIndex.from_product([['A비율'], denomi_table.columns.levels[1][:-1].insert(len(denomi_table.columns.levels[1][:-1]),'평균')])
        tmp_res_table = 100*numer_table/denomi_table
        res_table = tmp_res_table.apply(np.round, decimals=2)
        if percent:
            res_table = percentile_add(res_table, unit=1, ratio=True)
        if column_order:
            column_order_mod = column_order.copy()
            column_order_mod.append('평균')
            res_table.columns = pd.MultiIndex.from_product([['A비율'], column_order_mod])
        res_table = res_table.replace('%\n(nan)', '0')

    elif mode == 'c' or mode == 'cancel':
        numer_values = ['수강취소인원']
        denomi_values = ['수강인원']
        numer_table = making_pivot(data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_columns = numer_table.columns
        # percentage process parts
        if percent:
            numer_table.columns = numer_table.columns.levels[1:]
            denomi_table.columns = denomi_table.columns.levels[1:]
            pd.options.display.float_format = None
            tmp_percent = (numer_table / (numer_table + denomi_table) * 100).apply(np.round, decimals=2).astype(str)
            tmp_percent = "\n(" + tmp_percent + "%)"
            res_table = numer_table.applymap("{:.10g}".format).astype(str).replace('nan', '') + tmp_percent
            res_table = res_table.replace('\n(nan%)', '0\n(0%)')
        elif not percent:
            res_table = numer_table
        res_table.columns = res_columns
        res_table = res_table.replace('\n(%)', '')

    elif mode == 're' or mode == 'repeat':
        numer_values = ['재수강인원']
        denomi_values = ['수강인원']
        numer_table = making_pivot(data, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    sub_cond=sub_cond, mostright=rightmost)
        res_columns = numer_table.columns
        # percentage process parts
        if percent:
            numer_table.columns = numer_table.columns.levels[1:]
            denomi_table.columns = denomi_table.columns.levels[1:]
            pd.options.display.float_format = None
            tmp_percent = (numer_table / denomi_table * 100).apply(np.round, decimals=2).astype(str)
            tmp_percent = "\n(" + tmp_percent + "%)"
            res_table = numer_table.applymap("{:.10g}".format).astype(str).replace('nan', '') + tmp_percent
            res_table = res_table.replace('0\n(nan%)', '0\n(0%)')
        elif not percent:
            res_table = numer_table
        res_table.columns = res_columns
        res_table = res_table.replace('\n(nan%)', '0\n(0%)')

    elif mode == 's' or mode == 'subject':
        subject_values = ['교과목수']
        res_table = making_pivot(data=data, index=index, columns=columns, values=subject_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('\n(nan%)', '\n(0%)')
    elif mode == 'l' or mode == 'lecture':
        lecture_values = ['강좌수']
        res_table = making_pivot(data=data, index=index, columns=columns, values=lecture_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('\n(nan%)', '\n(0%)')
    elif mode == 'e' or mode == 'evaluation':
        data2 = data.where(data.강의평가 != 0).dropna(how="all")
        numer_values = ['강의평가']
        denomi_values = ['수강인원']
        numer_table = making_pivot(data2, index=index, columns=columns, values=numer_values,
                                   index_order=index_order, column_order=column_order,
                                   mode=mode, sub_cond=sub_cond, mostright=rightmost)
        denomi_table = making_pivot(data2, index=index, columns=columns, values=denomi_values,
                                    index_order=index_order, column_order=column_order,
                                    mode=mode, sub_cond=sub_cond, mostright=rightmost)
        if isinstance(numer_table.index, pd.core.index.MultiIndex) | isinstance(denomi_table.index, pd.core.index.MultiIndex):
            numer_table.columns = pd.MultiIndex.from_product([['강의평가'], numer_table.columns.levels[1]])
            denomi_table.columns = pd.MultiIndex.from_product([['강의평가'], denomi_table.columns.levels[1]])
        elif not isinstance(numer_table.index, pd.core.index.MultiIndex) and not isinstance(denomi_table.index, pd.core.index.MultiIndex):
            tmp_numerindex = numer_table.index.delete(-1)
            numer_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            denomi_table.index = tmp_numerindex.insert(len(tmp_numerindex),'전체평균')
            numer_table.columns = pd.MultiIndex.from_product([['강의평가'], numer_table.columns.levels[1][:-1].insert(len(numer_table.columns.levels[1][:-1]),'평균')])
            denomi_table.columns = pd.MultiIndex.from_product([['강의평가'], denomi_table.columns.levels[1][:-1].insert(len(denomi_table.columns.levels[1][:-1]),'평균')])
        res_table = numer_table/denomi_table
        # percentage process parts
        if percent:
            res_table = percentile_add(res_table, unit=1, ratio='gpa')
        if column_order:
            column_order_mod = column_order.copy()
            column_order_mod.append('평균')
            res_table.columns = pd.MultiIndex.from_product([['강의평가'], column_order_mod])
        res_table = res_table.fillna(0)
        res_table = res_table.replace('', '0')
        res_table = res_table.replace('\n()', '0')
    elif mode in ['ga']:
        attend_values = ['학점인원']
        res_table = making_pivot(data=data, index=index, columns=columns, values=attend_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('0\n(nan%)', '0\n(0%)')
    elif mode in ['gt']:
        attend_values = ['학점총합']
        res_table = making_pivot(data=data, index=index, columns=columns, values=attend_values,
                                 index_order=index_order, column_order=column_order,
                                 percent=percent, mode=mode, sub_cond=sub_cond, mostright=rightmost)
        res_table = res_table.replace('0\n(nan%)', '0\n(0%)')
    if not percent:
        if mode == 'g' or mode == 'mg' or mode == 'e':
            res_table = res_table.applymap('{:.2f}'.format).replace('nan', '').astype(str)
        elif mode == 'ra':
            res_table = res_table.applymap('{:.2f}'.format).astype(str) + "%"
            res_table = res_table.replace('nan%', '')
        elif mode in ['a', 'attend', 'c', 'cancel', 're', 'repeat']:
            res_table = res_table.applymap('{:.10g}'.format).replace('nan', '0').astype(str)
        else:
            res_table = res_table.applymap('{:.10g}'.format).replace('nan', '0').astype(str)
    res_table.index.names = index
    res_table = res_table.replace('nan', '0')
    return res_table


# making pivot tables and adding percentage table
def making_pivot(data, columns, index, values, column_order=False, index_order=False, percent=False, mode=None, local_sum=True, sub_cond=None, mostright=None):
    import pandas as pd
    import numpy as np
    if sub_cond:
        sub_data, data_mod = make_sub_data(data, sub_cond)
        sub_table = pd.pivot_table(sub_data, values=values, index=index, columns=columns, aggfunc=np.nansum,
                                   fill_value=False, margins=False, dropna=False)
    else:
        data_mod = data.copy()
    res_table = pd.pivot_table(data_mod,
                               values=values, index=index, columns=columns, aggfunc=np.nansum,
                               fill_value=False, margins=False, dropna=False)
    # try:
    #     print(res_table.loc["겨울학기", (res_table.columns.levels[0][0], 2015)])
    #     res_table.loc["겨울학기", (res_table.columns.levels[0][0], 2015)] = np.nan
    # except:
    #     print(res_table.loc["겨울학기"])
    # res_table = res_table.replace(0, np.nan)
    if index_order:
        if any(isinstance(i, list) for i in index_order):
            res_table = res_table.reindex(index=pd.MultiIndex.from_product(index_order))

        else: # elif not any(isinstance(i, list) for i in index_order):
            res_table = res_table.reindex(index=index_order)
    if column_order:
        res_table = res_table.reindex(columns=pd.MultiIndex.from_product([res_table.columns.levels[0], column_order]))
    res_table = res_table.fillna(0) # fill nan with zero (0) value (if you want calculating without nan value,
                                    # you should adjust this function)
    if local_sum:
        res_table = local_calc(res_table, mode, mostright)
    if sub_cond:
        assert isinstance(res_table, object)
        if column_order:
            sub_table = sub_table.reindex(columns=pd.MultiIndex.from_product([sub_table.columns.levels[0], column_order]))
        if local_sum:
            sub_table = local_calc(sub_table, mode)
        res_table.iloc[-1, :] = res_table.iloc[-1, :] - sub_table.iloc[-1, :]
    if percent:
        res_table = percentile_add(res_table)
    return res_table


def local_calc(data, mode=None, mostright=None):
    import pandas as pd
    import numpy as np
    # 소계 (local sum)
    if not any(isinstance(i, tuple) for i in data.index):
        tmp_sums = pd.DataFrame(np.nansum(data, 0)).transpose()
        tmp_sums.columns = data.columns
        tmp_sums.index = ['전체합계']
        data = pd.concat([data, tmp_sums], axis=0)
    elif any(isinstance(i, tuple) for i in data.index):
        total_sums = pd.DataFrame(np.nansum(data, axis=0)).transpose()
        if 'e' == mode or 'g' == mode or 'ra' == mode or 'mg' == mode:
            total_sums.index = [['전체평균'], ['평균']]
        else:
            total_sums.index = [['전체합계'], ['합계']]
        total_sums.columns = data.columns
        tmp_res_df = pd.DataFrame()
        for i in range(len(data.index.levels[0])):
            tmp_target = data.iloc[i * len(data.index.levels[1]): (i + 1) * len(data.index.levels[1])]
            tmp_sums = pd.DataFrame(np.nansum(tmp_target, 0)).transpose()
            tmp_sums.columns = data.columns
            if 'e' == mode or 'g' == mode or 'ra' == mode or 'mg' == mode:
                tmp_sums.index = [[tmp_target.index[0][0]], ['소계평균']]
            else:
                tmp_sums.index = [[tmp_target.index[0][0]], ['소계합계']]
            tmp_df = pd.concat([tmp_target, tmp_sums], axis=0)
            tmp_res_df = pd.concat([tmp_res_df, tmp_df], axis=0)
        data = tmp_res_df
        data = pd.concat([data, total_sums], axis=0)
    # 평균 (local mean)
    if 'g' == mode or 'ra' == mode or 'mg' == mode or mostright:
        tmp_means = pd.DataFrame(np.nansum(data, 1))
    else:
        tmp_means = pd.DataFrame(np.round(np.nanmean(data, 1), 2))
    tmp_means.index = data.index
    if mostright:
        tmp_means.columns = [[data.columns[0][0]], ['합계']]
    else:
        tmp_means.columns = [[data.columns[0][0]], ['평균']]
    data = pd.concat([data, tmp_means], axis=1)
    return data


def percentile_add(pivot_data, unit=100, ratio=None):
    import pandas as pd
    import numpy as np
    if isinstance(pivot_data.index, pd.core.index.MultiIndex):
        if pivot_data.index.nlevels == 2:  # Two MultiIndex
            dim1_length = len(pivot_data.index.levels[0])
            dim2_length = len(pivot_data.index.levels[1])
            tmp_pivot = pivot_data.copy(deep=False)
            sum_index = 0
            for i in range((dim1_length - 1) * (dim2_length - 1)):
                if i % (dim2_length - 1) == dim2_length - 2:
                    if unit == 1:
                        tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.2f}'.format).replace('nan', '').astype(str)
                        if ratio and ratio != 'gpa':
                            tmp_pivot.iloc[i] += "%"
                    sum_index += 1
                    continue
                if unit == 100:
                    tmp_row = "\n(" + (100 * pivot_data.iloc[i] / pivot_data.iloc[
                        (dim2_length - 2) + (sum_index * (dim2_length - 1))]).apply(np.round, decimals=2).astype(str) + "%)"
                elif unit == 1:
                    tmp_row = "\n(" + (pivot_data.iloc[i] / pivot_data.iloc[(dim2_length - 2) + (sum_index * (dim2_length - 1))]).apply(np.round, decimals=2).astype(str) + ")"
                if ratio == 'gpa':
                    tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.2f}'.format).replace('nan', '').astype(str) + tmp_row
                elif ratio:
                    tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.2f}'.format).replace('nan', '').astype(str) + "%" + tmp_row
                elif not ratio:
                    tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.10g}'.format).replace('nan', '').astype(str) + tmp_row
            if ratio == 'gpa':
                tmp_pivot.iloc[-1] = pivot_data.iloc[-1].map('{:.2f}'.format).replace('nan', '').astype(str)
            elif ratio:
                tmp_pivot.iloc[-1] = pivot_data.iloc[-1].map('{:.2f}'.format).replace('nan', '').astype(str) + "%"

    elif not isinstance(pivot_data.index, pd.core.index.MultiIndex):  # for not MultiIndex, i.e., unidimensional
        tmp_pivot = pivot_data.copy(deep=False)
        dim_length = len(pivot_data.index)
        for i in range(dim_length - 1):
            if unit == 100:
                tmp_row = "\n(" + (100*pivot_data.iloc[i]/pivot_data.iloc[dim_length - 1]).apply(np.round,decimals=2).astype(str) + "%)"
            elif unit == 1:
                tmp_row = "\n(" + (pivot_data.iloc[i]/pivot_data.iloc[dim_length - 1]).apply(np.round, decimals=2).astype(str) + ")"

            if ratio == 'gpa':
                tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.2f}'.format).replace('nan', '').astype(str) + tmp_row
            elif ratio:
                tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.2f}'.format).replace('nan', '').astype(str) + "%" + tmp_row
            elif not ratio:
                tmp_pivot.iloc[i] = pivot_data.iloc[i].map('{:.10g}'.format).replace('nan', '').astype(str) + tmp_row
        if ratio == 'gpa':
            tmp_pivot.iloc[-1] = pivot_data.iloc[-1].map('{:.2f}'.format).replace('nan', '').astype(str)
        elif ratio:
            tmp_pivot.iloc[-1] = pivot_data.iloc[-1].map('{:.2f}'.format).replace('nan', '').astype(str) + "%"
    return tmp_pivot


# where condition settings function
def make_sub_data(data, sub_cond, data_name='data'):
    """
    # example:
     # multiple condition
     sub_cond = [
     [['소영역_처리,==,외국어 Ⅰ(영어)','소영역_처리,==,외국어 Ⅱ(영어)'], ['소영역_처리','외국어'], '|'],
     [['소영역_처리,==,체육','소영역_처리,==,체육 및 기타'], ['소영역_처리','체육변형'], '|']
     ]
    """
    from pandas import DataFrame, concat
    res_data = data.copy()
    res_sub_data = DataFrame()
    for i in sub_cond:
        res_where = make_conditional(i[0], data=data_name, logical=i[2])
        try:
            tmp_sub = concat([tmp_sub, data.where(eval(res_where)).dropna(how='all')], axis=0)
        except:
            tmp_sub = data.where(eval(res_where)).dropna(how='all')
        tmp_replace = i[1]
        tmp_sub[tmp_replace[0]] = tmp_replace[1]
        res_sub_data = concat([res_sub_data, tmp_sub], axis=0)
        del tmp_sub
    res_data = concat([res_data, res_sub_data], axis=0)  # concatenated raw data and sub data
    return (res_sub_data, res_data)


def make_concat(where_cond, data):
    from pandas import DataFrame, concat
    res_concat = DataFrame()
    for i in where_cond:
        tmp_df = data.where(eval(make_conditional(i[0], logical='|'))).dropna(how='all')
        tmp_df[i[1]] = i[2]
        res_concat = concat([res_concat, tmp_df], axis=0)
    return res_concat

def make_constraint(cond, data):
    make_conditional(cond, '|')
    data.where(data[cond])
    return

def make_conditional(condition, data='data', logical='&'):
    # example: condition = ['소영역_처리,==,외국어 Ⅰ(영어)','소영역_처리,==,외국어 Ⅱ(영어)']
    import re
    # default_form = "({0}['{1}']{2}'{3}')"
    if not isinstance(condition, list):
        condition = [condition]
    index = 0
    cond = ''
    for i in condition:
        tmp_where = i.split(',')
        myre = re.compile(r'^\d{4}$')
        if myre.match(tmp_where[2]):
            default_form = "({0}['{1}']{2}{3})"
        else:
            default_form = "({0}['{1}']{2}'{3}')"
        tmp_cond = default_form.format(data, tmp_where[0], tmp_where[1], tmp_where[2])
        if len(condition) > 1:
            if index == 0:
                cond = tmp_cond
            elif index != 0:
                cond = cond + logical + tmp_cond
        elif len(condition) == 1:
            cond = tmp_cond
        index += 1
    return cond

def make_bracket(string):
    string = "(" + string + ")"
    return string

def snu_calendar(date_type=None, default=None):
    import re
    from bs4 import BeautifulSoup as bs
    from requests import get
    from datetime import date
    current_year = date.today().year
    base_url = "http://www.snu.ac.kr/academic-calendar"
    soup = bs(get(base_url).content, "html.parser")
    year = re.findall(r'\d{4}', soup.select('caption')[0].get_text())[0]
    date = []
    old_i = ''
    if date_type == 'l' or date_type == 's':
        extract = r'개강'
    else:
        extract = r'성적제출'
    for i in soup.select("td")[15:]: # except previous winter class
        if re.findall(r'\d{4}년', i.get_text()):
            year = re.findall(r'\d{4}', i.get_text())[0]
        if re.search(extract, i.get_text()):
            date.append([int(year),
                         int(old_i.get_text().replace('ㆍ','').split('.')[0]),
                         int(old_i.get_text().replace('ㆍ','').split('.')[1])])
        old_i = i
    if default:
        if date_type == 'l' or date_type == 's':
            date = [[current_year, 3, 1], [current_year, 6, 20], [current_year, 9, 1], [current_year, 12, 20]]
        else:
            date = [[current_year, 7, 1], [current_year, 8, 1], [current_year+1, 1, 1], [current_year+1, 2, 1]]
    return date

def days_convert(mode=1, date_type=None, default=None):
    from functionCTL import snu_calendar
    from datetime import date, timedelta
    calendar = snu_calendar(date_type, default)
    regular_1st = date(year = calendar[0][0], month = calendar[0][1], day = calendar[0][2]) + timedelta(7)
    season_1st = date(year = calendar[1][0], month = calendar[1][1], day = calendar[1][2]) + timedelta(7)
    regular_2nd = date(year = calendar[2][0], month = calendar[2][1], day = calendar[2][2]) + timedelta(7)
    season_2nd = date(year = calendar[3][0], month = calendar[3][1], day = calendar[3][2]) + timedelta(7)
    if mode == 1:
        return regular_1st
    elif mode == 2:
        return season_1st
    elif mode == 3:
        return regular_2nd
    elif mode == 4:
        return season_2nd

# for column index ordering
def year_semester(year, mode=1, date_type=None, default=None):
    from functionCTL import days_convert
    from datetime import date
    from os import linesep
    today = date.today()
    if mode == 1: # for all semester
        semester = ['1학기', '2학기', '여름학기','겨울학기']
    elif mode == 2: # for only summer and winter semester
        semester = ['여름학기','겨울학기']
    index_order = []
    while year <= today.year:
        if days_convert(mode=4, date_type=date_type, default=default).year == today.year and year == today.year:
            if days_convert(mode=4, date_type=date_type, default=default) >= today:
                semester.remove('겨울학기')
                semester.remove('2학기')
                semester.remove('여름학기')
                semester.remove('1학기')
        elif days_convert(mode=1, date_type=date_type, default=default).year == today.year and year == today.year:
            if days_convert(mode=4, date_type=date_type, default=default) >= today:
                semester.remove('겨울학기')
            if days_convert(mode=3, date_type=date_type, default=default) >= today:
                semester.remove('2학기')
            if days_convert(mode=2, date_type=date_type, default=default) >= today:
                semester.remove('여름학기')
            if days_convert(mode=1, date_type=date_type, default=default) >= today:
                semester.remove('1학기')
        for i in semester:
            index_order.append(str(int(year))+linesep + i) # convert year to integer
        year += 1
    return index_order

def attend_check(data, subject, area, id, name):
    """
    :rtype : pandas.DataFrame
    :param data: input data
    :param subject: target subject
    :param area:
    :param id: extract column
    :param name: result column name
    """
    data_mod = data.copy()
    id_list = data_mod.where(data_mod[area]==subject).dropna(how='all')[id].unique()
    data_mod[name] = data_mod[id].isin(id_list)
    return data_mod

# from functionCTL import attend_check
# a = attend_check(data, '기초영어', '교과목이름_처리', '개인번호', '기초영어수강여부')
# a['기초영어수강여부'] = a['기초영어수강여부'].replace([True, False],['기초영어수강생', '기초영어미수강생'])
# a.to_csv('12333.csv', index=False, encoding='utf-8')

def attend_interval(data, list, id, sort=None):
    """

    :type data: pandas.DataFrame
    :type list: data list
    :type id: identifying column
    :type : identifying column
    """
    data_mod = data.copy()
    res_data = data_mod.where(data_mod[list[0]].isin(list[1])).dropna(how='all')
    if sort:
        res_data = res_data.sort_values(sort)
    interval_list = []
    for i in list[1]:
        interval_list.append([i + "_" + x for x in list[1]])
        interval_list[-1] = [i + '_최초수강'] + [i + '_최종수강'] + interval_list[-1]
    for i in interval_list:
        res_data[i[0].split('_')[0]] = 0
        for j in i:
            res_data[j] = 0
    index = 0
    for i in list[1]:
        index += 1
        for j in range(len(res_data)):
            try:
                if res_data.iloc[j,:][list[0]] == i and (res_data.iloc[j,:][id] != res_data.iloc[j-1,:][id] or j == 0):
                    res_data[i + '_최초수강'].iloc[j] = 1
            except:
                print('최초수강')
            try:
                if j == len(res_data) - 1:
                    if res_data.iloc[j,:][list[0]] == i and j == len(res_data)-1:
                        res_data[i + '_최종수강'].iloc[j] = 1
                elif j < len(res_data) - 1:
                    if res_data.iloc[j,:][list[0]] == i and res_data.iloc[j,:][id] != res_data.iloc[j+1,:][id]:
                        res_data[i + '_최종수강'].iloc[j] = 1
            except:
                print('최종수강')
            try:
                if j == len(res_data) - 1:
                    if j == len(res_data)-1 and res_data.iloc[j,:][id] != res_data.iloc[j-1,:][id]:
                        if res_data.iloc[j,:][list[0]] == i and checker:
                            res_data[i].iloc[j] = 1
                elif j < len(res_data) - 1:
                    checker = res_data.iloc[j,:][id] != res_data.iloc[j+1,:][id] and res_data.iloc[j,:][id] != res_data.iloc[j-1,:][id]
                    if res_data.iloc[j,:][list[0]] == i and checker:
                        res_data[i].iloc[j] = 1
            except:
                print('단독수강')
            try:
                if j < len(res_data) - 1:
                    if res_data.iloc[j,:][list[0]] == i and res_data.iloc[j,:][id] == res_data.iloc[j+1,:][id]:
                        res_data[i + '_' + res_data.iloc[j+1,:][list[0]]].iloc[j+1] = 1
            except:
                print('그외')
            # except IndexError:
            #     print('{0} is the end of the data point'.format(j))
        print('{0:.2f}% completed...'.format(index/len(list[1])*100))
    return res_data

def attend_interval_pivot(data, list):
    """

    :type data: pandas.DataFrame (which is processed)
    :type list: data list
    """
    import pandas as pd
    res_data = data.copy()
    res_pivot = pd.DataFrame(index = list, columns = ['해당교과'] + list)
    interval_list = []
    for i in list:
        interval_list.append([i + "_" + x for x in list])
        interval_list[-1] = [i + '_최초수강'] + [i + '_최종수강'] + interval_list[-1]
    for i in interval_list:
        res_pivot.loc[i[0].split('_')[0], '해당교과'] = sum(res_data[i[0].split('_')[0]])
        for j in i:
            tmp_var = j.split('_')
            res_pivot.loc[tmp_var[0], tmp_var[1]] = sum(res_data[j])
    return res_pivot

def auto_interval_calc(data, condition, id, list, sort):
    from functionCTL import attend_interval, attend_interval_pivot
    import pandas as pd
    data_mod = data.copy()
    res_pivot = pd.DataFrame()
    index = 0
    for i in condition[1]:
        sub_data = data_mod.where((data[condition[0]]==i) & data[list[0]].isin(list[1])).dropna(how='all')
        print(len(sub_data))
        tmp_data = attend_interval(sub_data, list, id, sort)
        tmp = attend_interval_pivot(tmp_data, list[1])
        tmp_res = pd.DataFrame(data = [i]*len(tmp), columns = [condition[0]], index = tmp.index)
        tmp_pivot = pd.concat([tmp_res, tmp], axis=1)
        res_pivot = pd.concat([res_pivot, tmp_pivot], axis=0)
        index += 1
        print('{0} (of {1}) is completed..'.format(index, len(condition[1])))
    return res_pivot

def remove_duplicate_index(pivot):
    res_pivot = pivot.reset_index()
    tmp = res_pivot.iloc[:,0].copy()
    for i in range(len(res_pivot)):
        if i != 0 and res_pivot.iloc[i,0] == res_pivot.iloc[i-1,0]:
            tmp[i] = ''
    res_pivot.iloc[:,0]= tmp
    return res_pivot

def reorder(input_text, target_list=None):
    if input_text == '대영역13':
        output_order = ['학문의 기초', '핵심교양', '일반교양']
    elif input_text == '소영역_처리' or input_text == '소영역14':
        output_order = ['사고와 표현', '기초영어', '외국어 Ⅰ(영어)', '외국어 Ⅱ(제2외국어)',
                        '수량적 분석과 추론', '과학적 사고와 실험', '컴퓨터와 정보 활용', '언어와 문학', '역사와 철학',
                        '문화와 예술', '정치와 경제', '인간과 사회', '자연과 기술', '생명과 환경',
                        '체육', '예술 실기', '대학과 리더십','창의와 융합','한국의 이해']
    elif input_text == '소영역_처리_기초영어':
        output_order = ['사고와 표현', '외국어 Ⅰ(영어)', '외국어 Ⅱ(제2외국어)',
                        '수량적 분석과 추론', '과학적 사고와 실험', '컴퓨터와 정보 활용', '언어와 문학', '역사와 철학',
                        '문화와 예술', '정치와 경제', '인간과 사회', '자연과 기술', '생명과 환경',
                        '체육', '예술 실기', '대학과 리더십','창의와 융합','한국의 이해']
    elif input_text == '대영역_처리_기초영어' or input_text == '대영역14':
        output_order = ['학문의기초', '학문의 세계', '선택교양']
    elif input_text == '대영역_처리':
        output_order = ['기초영어', '학문의기초', '학문의 세계', '선택교양']
    elif input_text == '통합개설학기':
        output_order = ['1학기', '2학기', '여름학기', '겨울학기']
    elif input_text == '학년그룹':
        output_order = ['1학년', '2학년', '3학년', '4학년', '5학년 이상']
    elif input_text == '교수직급_그룹화':
        output_order = ['전임', '강의교수', '시간강사', '명예·초빙교수 등']
    elif input_text == '개설대학':
        output_order = ['기초교육원', '인문대학', '사회과학대학', '자연과학대학', '공과대학']
    elif input_text == '소속단과대학':
        output_order = ['인문대학', '사회과학대학', '자연과학대학', '공과대학']
    elif input_text in ["소속부서", "개설부서"]:
        output_order = ["수리과학부",  "통계학과", "물리·천문학부", "화학부", "생명과학부", "지구환경과학부"]
    elif input_text == '교과목이름_처리':
        output_order = ["글쓰기의 기초", "인문학글쓰기", "사회과학글쓰기", "과학과 기술 글쓰기", "말하기와 토론",
                        "창의적 사고와 표현", "논리와 비판적 사고", # 사고와 표현
                        "초급한국어", "중급한국어 1", "중급한국어 2", "고급한국어", "초급한문 1", "초급한문 2", "중급한문",
                        "한문명작읽기", "역사와 철학 한문원전읽기", "초급중국어 1", "초급중국어 2", "중급중국어 1",
                        "중급중국어 2", "중국어회화 1", "중국어회화 2", "미디어중국어",
                        "기초영어", "대학영어 1", "대학영어 2: 글쓰기", "대학영어 2: 말하기",
                        "고급영어: 산문", "고급영어: 학술작문", "고급영어: 영화", "고급영어: 연극",
                        "고급영어: 문화와 사회", "고급영어: 발표", "고급영어: 문학", "초급프랑스어 1", "초급프랑스어 2",
                        "중급프랑스어 1", "중급프랑스어 2", "프랑스어 글쓰기", "프랑스어 말하기", "시사프랑스어",
                        "초급독일어 1", "초급독일어 2", "중급독일어 1", "중급독일어 2", "독일어 글쓰기",
                        "독일어로 읽는 문화와 예술", "시사독일어", "초급러시아어 1", "초급러시아어 2", "중급러시아어 1",
                        "중급러시아어 2", "러시아어 말하기", "러시아어로 읽는 문화와 예술", "시사 러시아어",
                        "초급스페인어 1", "초급스페인어 2", "중급스페인어 1", "중급스페인어 2", "스페인어 글쓰기",
                        "스페인어 말하기", "시사스페인어", "포르투갈어입문 1", "포르투갈어입문 2", "이태리어 1",
                        "이태리어 2", "스와힐리어 1", "스와힐리어 2", "몽골어 1", "몽골어 2", "산스크리트어 1",
                        "산스크리트어 2", "고급일본어 1", "고급일본어 2", "아랍어 1", "아랍어 2", "힌디어 1", "힌디어 2",
                        "말레이-인도네시아어 1", "말레이-인도네시아어 2", "터키어 1", "터키어 2", "베트남어 1", "베트남어 2",
                        "고전그리스어 1", "고전그리스어 2", "라틴어 1", "라틴어 2", # 외국어
                        "수학 및 연습 1", "수학 및 연습 2", "고급수학 및 연습 1", "고급수학 및 연습 2", "미적분학 및 연습 1",
                        "미적분학 및 연습 2", "생명과학을 위한 수학 1", "생명과학을 위한 수학 2", "경영학을 위한 수학",
                        "인문사회계를 위한 수학 1", "인문사회계를 위한 수학 2", "수학의 기초와 응용 1", "수학의 기초와 응용 2",
                        "공학수학 1", "공학수학 2", "기초수학 1", "기초수학 2", "미적분학의 첫걸음", "통계학", "통계학실험",
                        "통계학의 개념 및 실습", # 수량적 분석과 추론
                        "물리학 1", "물리학 2", "고급물리학 1", "고급물리학 2", "물리의 기본 1",
                        "물리의 기본 2", "물리학", "인문사회계를 위한 물리학", "물리학실험 1", "물리학실험 2",
                        "물리학실험", "기초물리학 1", "기초물리학 2", "천문학", "천문학실험", "화학 1", "화학 2",
                        "화학", "화학실험 1", "화학실험 2", "화학실험", "기초화학 1", "기초화학 2", "생물학 1",
                        "생물학 2", "생물학", "인문사회계를 위한 생물학", "생물학실험 1", "생물학실험 2", "생물학실험",
                        "기초생물학 1", "기초생물학 2", "지구환경과학", "지구환경과학실험", "대기과학", "대기과학실험",
                        "지구시스템과학", "지구시스템과학실험", "해양학", "해양학실험", "지구과학", "지구과학실험", # 과학적 사고와 실험
                        "컴퓨터의 개념 및 실습", "컴퓨터의 기초"] # 컴퓨터와 정보 활용
    elif input_text == '전형구분_분류':
        output_order = ["일반전형수시", "일반전형정시", "지역균형", "기회균형1", "기회균형2",
                        "글로벌인재1", "글로벌인재2", "기타"]
    elif input_text == '전형구분_세부분류':
        output_order = ["일반전형수시", "일반전형정시", "지역균형", "저소득", "기초생활수급권자",
                        "농어촌", "농업계열고교졸업예정", "정원외(특수교육대상자)",
                        "새터민", "전기정원외(영주자)", "전기정원외(외국인)",
                        "정부초청장학생", "외국인편입학", "후기정원외(영주자)", "후기정원외(외국인)",
                        "학사편입학", "학사편입학(의과대학)", "군위탁편입학", "약대전공과정(일반)",
                        "약대전공과정(정원외 기초생활)", "약대전공과정(정원외 농어촌)", "약대전공과정(정원외 재외국민)"]
    else:
        return sorted(target_list)
    if target_list:
        sorted_list = [x for x in output_order if x in target_list]
        unsorted_list = sorted([x for x in target_list if not x in output_order])
        final_order = sorted_list + unsorted_list
    elif not target_list:
        final_order = output_order
    return final_order

def make_doc_table(document, pivot, columns=1):
    table = document.add_table(rows=1, cols=pivot.shape[1])
    index = 0
    hdr_cells = table.rows[0].cells
    if 1 == columns:
        hdr_cells[0].text = pivot.columns.levels[0][0]
    elif 2 == columns:
        hdr_cells[0].text = pivot.columns.levels[0][1]
        hdr_cells[1].text = pivot.columns.levels[0][2]

    for i in pivot.columns.levels[1].tolist()[:-1]:
        hdr_cells[index + columns].text = str(i)
        index += 1
    for i in range(pivot.shape[0]):
        row_cells = table.add_row().cells
        for j in range(pivot.shape[1]):
            row_cells[j].text = pivot.iloc[i, j]
    table.style = 'Table Grid'

def zero_list(pivot_table): # delete zero total sum of row
    from numpy import nansum
    import pandas
    if type(pivot_table) == pandas.core.frame.DataFrame:
        exist_list = []
        for i in range(pivot_table.shape[0]):
            tmp = nan_process(pivot_table.iloc[i].copy(deep=True))
            try:
                if nansum(tmp.replace('', None).astype(float)):
                    exist_list.append(i)
            except:
                print("NaN value exists!\nCheck your data value! (it's warning!!)")
        return exist_list
    else:
        print("This is not a pandas data frame\nCheck your data type!")

def eliminate_zero(pivot_table, list):
    return pivot_table.iloc[list]

def nan_process(table):
    table = table.replace('\n(nan)', ' ')
    table = table.replace('nan', '0')
    table = table.replace('0\n(nan%)', ' ')
    table = table.replace('%', '', regex=True)
    return table