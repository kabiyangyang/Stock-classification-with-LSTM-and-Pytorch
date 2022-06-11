from pandas import DataFrame
from readdb1 import *
from util1 import *
import statsmodels.api as sm
import os
import pickle
import warnings
# warnings.filterwarnings("ignore")


#path=os.getcwd()


class BackTesting():
    def __init__( self, holdday : int, num : int, date_parameter : dict, index : str = '000906'):
            
        self.holdday = holdday
        self.num = num
        self.index = index
        self.start_date = date_parameter['train_end_date']
        self.end_date = date_parameter['valid_end_date']


    def get_stock_data(self):
        date_list =  get_trading_day_list(self.start_date, self.end_date, frequency='day') 
        all_stocks = get_periods_index_stocks(self.index, self.start_date, self.end_date)
        fac = pd.DataFrame(index=date_list,columns=all_stocks)

        factor_prices = Close_price(fac).fillna(method='bfill')
        self.factor_pro = factor_prices.pct_change()
        self.profitd = self.factor_pro.copy()
        self.profit10d = factor_prices.pct_change(periods=self.holdday).shift(-self.holdday)
        self.index_price = get_index_price(self.index, start_date=self.start_date, end_date=self.end_date)
    
    def __stand__(self, factor: DataFrame):
        factor = factor.sub(factor.mean(axis=1),axis=0).div(factor.std(axis=1),axis=0)
        return factor 

    def __MaxDrawdown__(self, return_list):
        '''最大回撤率'''
        i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(return_list[:i])  # 开始位置
        return (return_list[j] - return_list[i]) / (return_list[j])
    
    def draw_profit(self, factor: DataFrame, factor_save: str, model_param: dict, hyper_param : dict):
        factor_copy = factor.copy()
        profitd = self.profitd.applymap(lambda x: 0 if (x<=0.094) or (x>=0.101 and x <0.194) else np.nan)
        factor = factor + profitd
        factor = factor.dropna(axis=0, how='all')
        factor1 = factor.rank(axis=1,ascending= False)

        weight=factor1.applymap(lambda x: 1/self.num if x<=self.num else np.nan)   #多空就改这句话
        weight = weight.fillna(0)
        weight1 = weight.shift(1)
        for i in range(2,self.holdday+1):
            weight1 = weight1 + weight.shift(i).fillna(0)

        profit = self.factor_pro * weight1
        profit = profit.dropna(axis=0, how='all')
        profit1 = profit.sum(axis=1)/self.holdday +1- 0.0015/self.holdday
        rf = 0.03/250
        sharp = (profit1.mean()-rf -1)/(profit1-1).std()*(250**0.5)
        factor_copy.to_csv(os.path.join(factor_save, f'{sharp}{str(model_param)}_{str(hyper_param)}.csv'))
#####################################################################################  
  # 净值曲线
        fig = plt.figure(figsize=(15, 9), dpi=100)
        ax = fig.add_subplot(211)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        x = [pd.to_datetime(str(xx), format='%Y-%m-%d') for xx in profit1 .index.values.tolist()]
        ax.plot(x,  profit1.cumprod().values, color = 'y')

        tmp1 = self.index_price[(self.index_price['tradeDate']>=profit1 .index.values[0]) & (self.index_price['tradeDate']<=profit1 .index.values[-1])]
        close_index = pd.pivot_table(tmp1, index=['tradeDate'], columns=['ticker'], values='closeIndex')
        close_index = close_index/close_index.iloc[0]
        profit_index = close_index/close_index.shift(1)

        sup = profit1 - profit_index.iloc[:,0]
        ax.plot(x, close_index, color = 'b')
        plt.xlabel('time', fontsize=14)
        plt.ylabel("value", fontsize=16)
        plt.gcf().autofmt_xdate()
        ax.plot(x,  (sup+1).cumprod().values, color = 'g')
        ax.legend(['net_value', 'index_value', 'profit'])
        # sharp = (sup.mean()-rf)/(profit1-1).std()*(250**0.5)
        a=self.__MaxDrawdown__(profit1.cumprod().values.tolist())
        plt.title('sharp:'+str(sharp)+'MaxDrawdown'+str(a))
        plt.show()
        #plt.savefig(os.path.join(os.path.dirname(__file__),'净值曲线.jpg'))

########################################################################
# 直方图
        factor_comprofit = {}
        ax = fig.add_subplot(212)
        for i in range(1,factor.shape[0]):
            factor_cur = factor.iloc[i, :].dropna()
            factor_pro = self.profit10d.loc[factor.index[i],factor_cur.index]
            factor_df = pd.concat([factor_cur,factor_pro],axis=1)
            factor_df.columns = ['factor','profit']
            results = factor_df.sort_values(by='factor',ascending=False)

            group= 20
            num = math.ceil(results.shape[0] / group)

            profit1 = {}
            code1 = {}

            for m in range(0, group - 1):
                profit1[m] = results.iloc[m * num:num + m * num,1].mean()
                code1[m] = results.index[m * num:num + m * num].tolist()
            profit1[m + 1] = results.iloc[(m + 1) * num:,1].mean()
            code1[m + 1] = results.index[(m + 1) * num:].tolist()
            factor_comprofit[factor.index[i]] = profit1

        factor_comprofit_e = profit_2df(factor_comprofit)
        abar = factor_comprofit_e.mean(axis=0)-1
        abar.plot(ax = ax, kind='bar')
        plt.show()
        plt.savefig(os.path.join(factor_save, f'{sharp}_收益_分组_{str(model_param)}_{str(hyper_param)}.jpg'))











