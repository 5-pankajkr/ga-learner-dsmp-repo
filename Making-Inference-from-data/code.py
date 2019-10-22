# --------------
import pandas as pd
import math

data = pd.read_csv(path)

data_sample=data.sample(n=2000, random_state=0)
# print(data_sample.head())

true_mean=data.installment.mean()
print(true_mean)
sample_mean=data_sample.installment.mean()
sample_std=data_sample.installment.std()
print(sample_mean, sample_std)

z_critical=1.645
margin_of_error=z_critical*sample_std/math.sqrt(2000)
# margin_of_error2=sample_std/math.sqrt(2000)
print(margin_of_error, z_critical)

confidence_interval=(sample_mean-margin_of_error, sample_mean+margin_of_error)
print(confidence_interval)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fi, axes=plt.subplots(3, 1)

i=0
while (i < len(sample_size)):
    m =[]
    for j in range(1000):
        new_sample=data['installment'].sample(n=sample_size[i])
        mean_sample=new_sample.mean()
        m.append(mean_sample)

    mean_series=pd.Series(m)
    axes[i].hist(mean_series)
    i+=1

plt.show()


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate']=data['int.rate'].replace('\%', '', regex=True).astype(float)
data['int.rate'] = data['int.rate']/100
# print(data['int.rate'].head())

z_statistic, p_value=ztest(data[data['purpose']=='small_business']['int.rate'], x2=None, value=data['int.rate'].mean(), alternative='larger')

if p_value<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value=ztest(x1=data[data['paid.back.loan']=='No']['installment'], x2=data[data['paid.back.loan']=='Yes']['installment'])

if p_value<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


# --------------
#Importing header files
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
print(yes.head())
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()

observed=pd.concat([yes.transpose(), no.transpose()], axis=1, keys= ['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)

if chi2>critical_value:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


