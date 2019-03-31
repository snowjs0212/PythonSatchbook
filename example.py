get_ipython().magic('reset')
import pandas as pd
from scipy.stats import chi2_contingency

class WorkFlow():
    def __init__(self,file_name):
        self.input = pd.read_csv(file_name)

    def pre_proc(self,x_list,y):
        _ = self.input
        _['response'] = (_[y] > 2).astype(int)
        df = _[x_list]
        df['response'] = _['response']
        df = df.drop_duplicates()
        df = pd.get_dummies(data=df, columns=x_list)
        self.df = df

    def chi2_test(self,alpha):
        input = self.df
        stat,p,dof,ex,reject,failtoreject = dict(),dict(),dict(),dict(),dict(),dict()
        for i in list(input.loc[:, ~input.columns.isin(['response'])]):
            stat[i], p[i], dof[i], ex[i] = chi2_contingency(
                pd.crosstab(input['response'],
                            input[i]
                            )
            )
            if p[i] < alpha:
                reject[i] = p[i]
            else:
                failtoreject[i] = p[i]
            out = pd.Series(reject, index=reject.keys())
        return out

# Parameter set-up
file_name = 'final_data.csv' # Your final file name in .csv to run the testing
col_list = ['event','patient_sex_code','education_code','marital_status_code',
            'race_code','homeless_value'] # Modify this list to include any categorical variables

# 4 Cases: 'refills_auth', 'Drugs Per Refill', 'Drugs Per Event', 'Sum of Refill'
target = 'Sum of Refill' # Modify response variable
thred = 0.05 # Change chi-squared testing threshold

# Output
case = WorkFlow(file_name)
case.pre_proc(col_list,target)
case.chi2_test(thred).to_csv('chi2_' + target + '.csv''')
