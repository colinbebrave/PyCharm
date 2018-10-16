# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
# from datetime import datetime as dt
import calendar
from sqlalchemy import create_engine
import psycopg2
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def insert2pgSql(result, database, table = [], table_list = []):
    conn = psycopg2.connect(database = database['name'], user = database['user'],
                            password = database['password'], host = database['ip'],
                            port = database['port'])
    cursor = conn.cursor()
    try:
        st = datetime.datetime.now()
        print('length of the table_list', len(table_list))
        table_list_str = str(table_list).replace('[', ' ').replace(']', ' ').replace("'", ' ')
        sql = "INSERTING INTO" + table + "(" + table_list_str + ") VALUES"

        for i in range(len(result) - 1):
            sql += "("
            for j in range(len(table_list) - 1):
                sql += str(result[table_list[j]].iloc[i]) + "','"
            sql += str(result[table_list[len(table_list) - 1]].iloc[i]) + "'),"
        sql += "('"
        for j in range(len(table_list) - 1):
            sql += str(result[table_list[j]].iloc[len(result) - 1]) + "','"
        sql += str(result[table_list[len(table_list) - 1]].iloc[len(result) - 1]) + "')"
        cursor.excute(sql)
        conn.commit()
        ed = datetime.datetime.now()
        print('insert time cost', ed - st)
    except Exception as err:
        print(err)
    finally:
        cursor.close()
        conn.close()

def get_interrange(mond_temp, date_start, date_end):
    if mond_temp == 'day':
        time_inter = '1 day'
        time_range = pd.date_range(date_start, date_end)
        time_range = time_range[::-1]
    elif mond_temp == 'month':
        time_inter = '1 month'
        time_range = pd.period_range(date_start, date_end, freq = 'm').to_datetime()
        time_range = time_range[::-1]
    return time_inter, time_range

def comdateclean(data, arg1, arg2):
    data = data[data[arg1] != -1]
    data = data[pd.notnull(data[arg1])]
    data[arg1] = data[arg1].apply(int)
    for i in arg2:
        data[i] = pd.to_datetime(data[i])
        data = data[data[i].notnull()]
    data.index = range(len(data))
    return data

def dbef(n, data_series, now_time = datetime.date.today()):
    idx = data_series[data_series['trade_date'] == now_time].index[0]

    return data_series.iloc[idx - n, :][0]

def mbef(n, now_time = datetime.date.today()):
    period = n // 12
    rest = np.mod(n, 12)
    last_year = (now_time.year - period) if (now_time.month > rest) else (now_time.year - period - 1)
    last_month = (now_time.month - rest) if (now_time.month > rest) else (12 + now_time.momth - rest)
    last_day = np.min([calendar.monthrange(last_year, last_month)[1], now_time.day])

    return datetime.date(last_year, last_month, last_day)

def ybef(n, now_time = datetime.date.today()):
    last_year = now_time.year - n
    last_month = now_time.month
    last_day = np.min([calendar.monthrange(last_year, last_month)[1], now_time.day])

    return datetime.date(last_year, last_month, last_day)

# 中债估值收益率的涨跌BP数 向前n_day个交易日
def defr_1441010001(n_day, date_this, cvs_data):
    df_signal = pd.DataFrame(
        columns = ['bcode', 'signal_code', 'signal_value', 'cal_formula',
                   'cal_value', 'update_date', 'data_date'])
    day_signal_dict = {1: 'r_1441010001', 2: 'r_1441010002', 3: 'r_1441010003', 7: 'r_1441010007',
                       14: 'r_1441010014', 21: 'r_1441010021', 30: 'r_1441010030'}

    b = cvs_data.groupby('bcode').apply(
        lambda x: x.loc[x['data_date'].idxmax(), 'yield'] - x.loc[x['data_date'].idxmin(), 'yield']).reset_index()
    c = cvs_data.groupby('bcode').apply(lambda y: np.min(y['data_date'])).reset_index()
    cvs_data = pd.merge(b, c, on = 'bcode', how = 'inner')
    cvs_data.columns = ['bcode', 'yield_diff' 'data_date']

    df_signal['bcode'] = cvs_data['bcode']
    df_signal['signal_code'] = day_signal_dict[n_day]
    df_signal['signal_value'] = cvs_data['yield_diff'] * 100
    df_signal['cal_formula'] = '中债估值收益率的涨跌BP数 向前{0}个交易日'.format(n_day)
    df_signal['cal_value'] = df_signal['signal_value']
    df_signal['update_date'] = date_this
    df_signal['data_date'] = cvs_data['data_date']

    return df_signal

# 中债估值收益率的涨跌幅度
def defr_1441020000(n_day, date_this, cvs_data):
    df_signal = pd.DataFrame(
        columns = ['bcode', 'signal_code', 'signal_value', 'cal_formula',
                 'cal_value', 'update_date', 'data_date'])
    day_signal_dict = {1: 'r_1441020001', 2: 'r_1441020002', 3: 'r_1441020003', 7: 'r_1441020007',
                       14: 'r_1441020014', 21: 'r_1441020021', 30: 'r_1441020030'}

    b = cvs_data.groupby('bcode').apply(
        lambda x:x.loc[x['data_date'].idxmax(), 'yield'] / x.loc[x['data_date'].idxmin(), 'yield'] - 1).reset_index()
    c = cvs_data.groupby('bcode').apply(lambda y: np.min(y['data_date'])).reset_index()
    cvs_data = pd.merge(b ,c, on = 'bcode', how = 'inner')
    cvs_data.columns = ['bcode', 'yield_diff', 'data_date']

    df_signal['bcode'] = cvs_data['bcode']
    df_signal['signal_code'] = day_signal_dict[n_day]
    df_signal['signal_value'] = cvs_data['yield_diff']
    df_signal['cal_formula'] = '中债估值收益率的涨跌幅度（向前{0}个交易日）'.format(n_day)
    df_signal['cal_value'] = df_signal['signal_value']
    df_signal['update_date'] = date_this
    df_signal['data_date'] = cvs_data['data_date']

    return df_signal

def judge_inc(data):
    if (data['signal_value_x'] > 0) & (data['signal_value_y'] < 0):
        return 1
    elif (data['signal_value_x'] < 0) & (data['signal_value_y'] > 0):
        return -1
    else:
        return 0

def defr_1441030000(date_this, r1, r2):
    df_signal = pd.DataFrame(
        columns = ['bcode', 'signal_code', 'signal_value', 'cal_formula',
                 'cal_value', 'update_date', 'data_date'])
    if len(r1) > 0 and len(r2) > 0:
        data = pd.merge(r1, r2, on = 'bcode', how = 'inner')
        data = data[['bcode', 'signal_value_x', 'signal_value_y']]

        df_signal['bcode'] = data['bcode']
        df_signal['signal_code'] = 'r_1441030000'
        df_signal['signal_value'] = data.apply(judge_inc, axis = 1)
        df_signal['cal_formula'] = '1表示由跌转涨；-1表示由涨转跌'
        df_signal['cal_value'] = df_signal['signal_value']
        df_signal['update_date'] = date_this
        df_signal['data_date'] = cvs_data['data_date']

        return df_signal

def handle(grouped):
    se = grouped.set_index('data_date')['yield'].sort_index()
    return pd.concat(
        {'diff': se - se.shift(1)}, axis = 1
    )

def longest_days(grouped):
    a_list = grouped.set_index('data_date')['diff'].sort_index()
    neg_longest_bp = 0
    neg_count_bp = 0
    neg_longest = 1
    neg_count = 0
    pos_longest_bp = 0
    pos_count_bp = 0
    pos_longest = 1
    pos_count = 0
    for i, x in enumerate(a_list):
        if x == 0:
            continue
        elif x < 0:
            if x < -0.1:
                neg_count_bp += 1
                if neg_count_bp > neg_longest_bp:
                    neg_longest_bp = neg_count_bp
            neg_count += 1
            if neg_count > neg_longest:
                neg_longest = neg_count
            pos_count = 0
            pos_count_bp = 0
        else:
            if x > 0.1:
                pos_count_bp += 1
                if pos_count_bp > pos_longest_bp:
                    pos_longest_bp = pos_count_bp
            pos_count += 1
            if pos_count > pos_longest:
                pos_longest = pos_count
            neg_count = 0
            neg_count_bp = 0

    return pos_longest, neg_longest, pos_longest_bp, neg_longest_bp

def defr_1441040010(n_day, date_this, cvs_data):
    df_signal = pd.DataFrame(
        columns = ['bcode', 'signal_code', 'signal_value', 'cal_formula',
                 'cal_value', 'update_date', 'data_date'])
    day_signal_dict = {10: ('r_1441040010', 'r_1441040011', 'r_1441050010', 'r_1441050011'),
                       20: ('r_1441040020', 'r_1441040021', 'r_1441050020', 'r_1441050021'),
                       30: ('r_1441040030', 'r_1441040031', 'r_1441050030', 'r_1441050031')}
    if len(cvs_data) > 0:
        cvs_data_temp = cvs_data.groupby('bcode').apply(handle).reset_index()
        cvs_data_new = cvs_data_temp.groupby('bcode').apply(longest_days).reset_index()
        cvs_data_new.columns = ['bcode', 'count']

        cvs_data_new['pos_longest'] = cvs_data_new['count'].apply(lambda x: x[0])
        cvs_data_new['neg_longest'] = cvs_data_new['count'].apply(lambda x: x[1])
        cvs_data_new['pos_longest_bp'] = cvs_data_new['count'].apply(lambda x: x[2])
        cvs_data_new['neg_longest_bp'] = cvs_data_new['count'].apply(lambda x: x[3])

        df_signal['bcode'] = cvs_data_new['bcode']
        df_signal['signal_value'] = day_signal_dict[n_day][0]
        df_signal['signal_value'] = cvs_data_new['pos_longest']
        df_signal['update_date'] = date_this
        df_signal['data_date'] = date_this

        df_signal_pos = df_signal.copy()
        df_signal_neg = df_signal.copy()
        df_signal_pos_bp = df_signal.copy()
        df_signal_neg_bp = df_signal.copy()

        df_signal_neg['signal_code'] = day_signal_dict[n_day][1]
        df_signal_neg['signal_value'] = cvs_data_new['neg_longest']

        df_signal_pos_bp['signal_code'] = day_signal_dict[n_day][2]
        df_signal_pos_bp['signal_value'] = cvs_data_new['pos_longest_bp']

        df_signal_neg_bp['signal_code'] = day_signal_dict[n_day][3]
        df_signal_neg_bp['signal_value'] = cvs_data_new['neg_longest_bp']
    else:
        df_signal_pos = df_signal.copy()
        df_signal_neg = df_signal.copy()
        df_signal_pos_bp = df_signal.copy()
        df_signal_neg_bp = df_signal.copy()

    return df_signal_pos, df_signal_neg, df_signal_pos_bp, df_signal_neg_bp

def execute(database, parameter, engine_oracle=None):
    print('Processing')
    signal_list = ['r_1441010001', 'r_1441010002', 'r_1441010003', 'r_1441010007', 'r_1441010014',
                   'r_1441010021', 'r_1441010030', 'r_1441020001', 'r_1441020002', 'r_1441020003',
                   'r_1441020007', 'r_1441020014', 'r_1441020021', 'r_1441020030', 'r_1441030000',
                   'r_1441040010', 'r_1441040011', 'r_1441040020', 'r_1441040021', 'r_1441040030',
                   'r_1441040031', 'r_1441050010', 'r_1441050011', 'r_1441050020', 'r_1441050021',
                   'r_1441050030', 'r_1441050031']

    # =============================================================== #
    # ======              <1> Connect the server               ====== #
    # =============================================================== #
    db = eval(str(database))
    # para = eval(str(parameter))
    # if para.has_key('com_list'):
    #     com_list = para['com_list']
    # else:
    #     com_list = []
    database_oracle = db['ebd_oracle']
    database_oracle['port'] = str(database_oracle['port'])
    engine_oracle = create_engine(''.join(
        ['oracle+cs_oracle://', database_oracle['user'], ':', database_oracle['password'],
         '@', database_oracle['ip'], ':', database_oracle['port'], '/', database_oracle['name']]))

    database_pg = db['ebd_pg']
    database_pg = str(database_pg['port'])
    engine_pg = create_engine(''.join(
        ['postgresql+psycopg2://', database_pg['user'], ':', database_pg['password'],
         '@', database_pg['ip'], ':', database_pg['port'], '/', database_pg['name']]))

    # =============================================================== #
    # ======                     <2> Read data                 ====== #
    # =============================================================== #
    print('Begin reading data..')
    mond = 'day'
    mond_temp = mond.strip().lower()
    date_end = datetime.date.today()
    # date_start = datetime.date(2017, 1, 1)
    date_start = ybef(3, date_end)

    time_inter, time_range = get_interrange(mond_temp, date_start, date_end)

    com_table = 'zzz_company_code_20170406'
    date_this = datetime.date(2017, 3, 8)

    '''
    cvs_raw = pd.read_sql_query(
    "SELECT tt.*, ss.issuance FROM \
    (SELECT bcode, company_id, yield, tdate::DATE as data_date FROM ebd_stgsmdb_bestimate t \
    WHERE exists(SELECT 1 FROM {0} s WHERE s.company_id = t.company_id) \ 
    AND t.tdate::DATE <= '{1}'::DATE \ 
    AND t.date::DATE >= to_date('{1}', 'YYYY-MM-DD') - INTERVAL '2' MONTH) tt \
    JOIN ebd_stgsmdb_bonddt ss on tt.bcode = ss.bcode".format(com_table, date_this), engine_pg)
    bdt_raw = pd.read_sql_query('SELECT bcode, issuance FROM ebd_stgsmdb_bonddt', engine_pg)
    bdt_raw = bdt_raw.drop_duplicates()
    a = pd.merge(cvs_raw, bdt_raw, on = 'bcode', how = 'inner')
    '''

    sql_date = 'SELECT t.tdate::DATE as trade_date FROM ebd_stgsmdb_bestimate t GROUP BY t.tdate'
    date_series = pd.read_sql_query(sql_date, engine_pg)
    date_series = date_series.sort_values('trade_date').reset_index(drop = True)
    print('Read date series done.')

    sql_data = "SELECT bcode, company_id, yield, tdate::DATE as data_date FROM ebd_stgsmdb_bestimate t \
                WHERE exists(SELECT 1 FROM {0} s WHERE s.company_id = t.company_id) \
                AND datasource = 1 \
                AND t.tdate::DATE <= '{1}'::DATE \
                AND t.tdate::DATE >= to_date('{1}', 'YYYY-MM-DD') - INTERVAL '2' MONTH LIMIT 50000".format(com_table, date_this)
    be_raw_old = pd.read_sql_query('sql_data', engine_pg)

    be_raw = be_raw_old.frop_duplicates().sort_values(by = ['bcode', 'data_date']).reset_index(drop = True)
    be_raw['data_date'] = pd.to_datetime(be_raw['data_date'], format = '%Y-%M-%D', errors = 'ignore')

    # =============================================================== #
    # ======                   <3> Let's do it                 ====== #
    # =============================================================== #
    print('Begin computing signals')

    # 1.中债估值收益率的涨跌BP数 向前n_day个交易日
    r_1441010001 = defr_1441010001(1, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(1, date_series, date_this))])
    r_1441010002 = defr_1441010001(2, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(2, date_series, date_this))])
    r_1441010003 = defr_1441010001(3, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(3, date_series, date_this))])
    r_1441010007 = defr_1441010001(7, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(7, date_series, date_this))])
    r_1441010014 = defr_1441010001(14, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(14, date_series, date_this))])
    r_1441010021 = defr_1441010001(21, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(21, date_series, date_this))])
    r_1441010030 = defr_1441010001(30, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(30, date_series, date_this))])

    # 2.中债估值收益率的涨跌幅度 向前n_day个交易日
    r_1441020001 = defr_1441020000(1, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(1, date_series, dat_this))])
    r_1441020002 = defr_1441020000(2, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(2, date_series, dat_this))])
    r_1441020003 = defr_1441020000(3, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(3, date_series, dat_this))])
    r_1441020007 = defr_1441020000(7, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(7, date_series, dat_this))])
    r_1441020014 = defr_1441020000(14, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(14, date_series, dat_this))])
    r_1441020021 = defr_1441020000(21, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(21, date_series, dat_this))])
    r_1441020030 = defr_1441020000(30, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(30, date_series, dat_this))])

    # 3.中债估值收益率连续涨跌拐点
    r_1441030000 = defr_1441030000(date_this, r_1441020001, r_1441020002)

    # 5.中债估值收益率10个交易日内连续涨跌超过10BP天数
    r_1441040010, r_1441040011, r_1441050010, r_1441050011 = defr_1441040010(10, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(10, date_series, date_this))])

    # 6.中债估值收益率20个交易日内连续涨跌超过10BP天数
    r_1441040020, r_1441040021, r_1441050020, r_1441050021 = defr_1441040010(20, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(20, date_series, date_this))])

    # 7.中债估值收益率30个交易日内连续涨跌超过10BP天数
    r_1441040020, r_1441040021, r_1441050020, r_1441050021 = defr_1441040010(30, date_this, be_raw[
        (be_raw['data_date'] <= date_this) & (be_raw['data_date'] >= dbef(30, date_series, date_this))])

    print('Good job, we are almost done')

    df_list = [eval(x) for x in signal_list]
    from functools import reduce
    result = reduce(lambda x, y: pd.concat([x, y], axis = 0), df_list)

    result = comdateclean(result, 'bcode', ['update_date', 'data_date'])
    # =============================================================== #
    # ======            <4> Insert result into Sql             ====== #
    # =============================================================== #
    insert_table = ['ebd_signal_company_all']
    table_list_all = ['company_id', 'signal_code', 'signal_value', 'cal_formula',
                      'cal_value', 'update_date', 'data_date']
    insert2pgSql(result, database_pg, table = insert_table[0], table_list = table_list_all)

    print('{0} we are done here'.format(date_this))

def main():

    # =============================================================== #
    # ======                Initiate Parameters                ====== #
    # =============================================================== #
    database = {"ebd_oracle": {"ip":"10.20.131.72", "port":"1536", "name":"d0eripab", "password":"paic56789","user":"pabdata"},
                "ebd_pg": {"ip":"10.20.130.122", "port":"7458", "name":"d0paebd", "password":"paic5678","user":"pierdata"}}
    parameter = {"_conc_index":1, "_conc_total":1}
    execute(database, parameter)

if __name__ == '__main__':
    main()