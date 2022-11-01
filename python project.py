#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from sklearn import datasets,linear_model
from sklearn.linear_model import LinearRegression


# In[3]:


data=pd.read_excel('C:\\Users\\HP\\Downloads\\corona1.xlsx')


# In[4]:


data


# In[5]:


data.head()


# # AVERAGE IN INDIA

# In[6]:


data.iloc[:,1:5].apply(np.mean)


# # Graph for Confirmed Covid cases in India
# 

# In[7]:


plt.figure(figsize=(5,5))
data['Confirmed'].plot(kind="line")


# # Graph for Active cases in India

# In[8]:


plt.figure(figsize=(5,5))
data['Active'].plot(kind='line')


# # Graph for Death rate in India

# In[9]:


plt.figure(figsize=(5,5))
data['Deaths'].plot(kind='line')


# # Graph for Recovered cases in India

# In[10]:


plt.figure(figsize=(5,5))
data['Recovered'].plot(kind='line')


# # Correlation

# In[ ]:





# In[11]:


data


# # Correlation

# In[12]:


plt.figure(figsize=(9,8))
cor=data.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.RdYlBu)
plt.show()


# In[ ]:





# In[13]:


data


# In[14]:


bins=(-1,0,1,2,3,4,5)
health=['less infected','partially infected','moderately infected','more infected','highly infected','very much infected']
data['Active status']=pd.cut(data['Active status'],bins=bins,labels=health)


# In[15]:


data['Active status']


# In[16]:


data['Active status'].value_counts()


# In[17]:


sns.countplot(data['Active status'])


# In[18]:


sns.relplot(x='Active status',y='Deaths',hue='Active status',data=data,height=5,aspect=3)


# In[19]:


sns.relplot(x='Active status',y='Recovered',hue='Active status',data=data,height=5,aspect=3)


# In[20]:


confirmed=[5041,
895121,
16842,
218099,
263940,
25130,
329694,
651227,
56981,
292169,
282569,
61301,
129031,
121695,
975955,
1109908,
9942,
697,
280289,
2564881,
29362,
14019,
4454,
12226,
339246,
40645,
220276,
327175,
6210,
871440,
304791,
33480,
98880,
609443,
581865
]


# In[21]:


recovered=[4973,
884978,
16785,
215277,
261648,
22587,
313749,
635364,
55004,
278880,
272714,
58620,
125535,
119626,
946589,
1080803,
9711,
592,
266323,
2262593,
28935,
13848,
4425,
12133,
336337,
39521,
193280,
319695,
6032,
849064,
299427,
33045,
96062,
596286,
567771
]


# In[22]:


Active=[6,
2946,
1,
1719,
727,
2178,
11934,
4890,
1156,
8823,
6745,
1654,
1513,
969,
16905,
24578,
101,
104,
10047,
248604,
53,
22,
18,
2,
990,
445,
20522,
4672,
43,
9746,
3684,
43,
1112,
4388,
3782]


# In[23]:


Deaths=[62,
7197,
56,
1103,
1565,
365,
4011,
10973,
821,
4466,
3110,
1027,
1983,
1100,
12461,
4527,
130,
1,
3919,
53684,
374,
149,
11,
91,
1919,
679,
6474,
2808,
135,
12630,
1680,
392,
1706,
8769,
10312,
]


# #  Standard deviation 

# In[24]:


print('The standard deviation of Confirmed case is',stats.stdev(confirmed))
print('The standard deviation of  Recovered case is',stats.stdev(recovered))
print('The standard deviation of Active cases is',stats.stdev(Active))
print('The standard deviation of Death cases is',stats.stdev(Deaths))



# # Top 10 cases

# In[25]:


data.nlargest(10,['Recovered'])


# In[26]:


data.nlargest(10,["Deaths"])


# In[27]:


data.nlargest(10,['Total population'])


# In[28]:


data.nlargest(10,['Confirmed'])


# # Linear Regression

# In[29]:


data


# In[ ]:





# In[30]:


Active=[6,
2946,
1,
1719,
727,
2178,
11934,
4890,
1156,
8823,
6745,
1654,
1513,
969,
16905,
24578,
101,
104,
10047,
248604,
53,
22,
18,
2,
990,
445,
20522,
4672,
43,
9746,
3684,
43,
1112,
4388,
3782]


# In[31]:


Deaths=[62,
7197,
56,
1103,
1565,
365,
4011,
10973,
821,
4466,
3110,
1027,
1983,
1100,
12461,
4527,
130,
1,
3919,
53684,
374,
149,
11,
91,
1919,
679,
6474,
2808,
135,
12630,
1680,
392,
1706,
8769,
10312,
]


# In[32]:


active=pd.DataFrame(Active,index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],columns=['Active'])


# In[33]:


death=pd.DataFrame(Deaths,index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],columns=['Deaths'])


# In[34]:


regr=LinearRegression()
regr.fit(active,death)
#regression intercept
print(regr.intercept_)        


# In[35]:


#regression coefficient
print(regr.coef_)


# In[36]:


#analysis value for death
y=(regr.coef_*36000)+regr.intercept_
y


# In[37]:


confirmed=[5041,
895121,
16842,
218099,
263940,
25130,
329694,
651227,
56981,
292169,
282569,
61301,
129031,
121695,
975955,
1109908,
9942,
697,
280289,
2564881,
29362,
14019,
4454,
12226,
339246,
40645,
220276,
327175,
6210,
871440,
304791,
33480,
98880,
609443,
581865
]


# In[38]:


recovered=[4973,
884978,
16785,
215277,
261648,
22587,
313749,
635364,
55004,
278880,
272714,
58620,
125535,
119626,
946589,
1080803,
9711,
592,
266323,
2262593,
28935,
13848,
4425,
12133,
336337,
39521,
193280,
319695,
6032,
849064,
299427,
33045,
96062,
596286,
567771
]


# In[39]:


confirmed=pd.DataFrame(confirmed,index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],columns=['Confirmed'])


# In[40]:


recovered=pd.DataFrame(recovered,index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],columns=['Recovered'])


# In[41]:


regr=LinearRegression()
regr.fit(confirmed,recovered)
#regression intercept
print(regr.intercept_)        


# In[42]:


#regression coefficient
print(regr.coef_)


# In[43]:


#analysis value for recover
y=(regr.coef_*1000)+regr.intercept_
y


# In[82]:


states=['less infected<1000','partially infected<5000','moderately infected<10000','more infected<20000','highly infected<30000','very much infected<250000'
]


# In[85]:


plt.pie(confirmed,labels=states,radius=2,autopct='%0.1f%%')
plt.title("Active cases")


# In[84]:


plt.plot( 'Active', data=data, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'Deaths',  data=data, marker='', color='olive', linewidth=2)

plt.legend()

# show graph
plt.show()


# In[63]:


plt.plot( 'Recovered', data=data, marker='', markerfacecolor='blue', markersize=12, color='yellow', linewidth=4)
plt.plot( 'Confirmed',  data=data, marker='', color='olive', linewidth=2)

plt.legend()

# show graph
plt.show()


# In[64]:


data1=pd.read_excel('C:\\Users\\HP\\Downloads\\corona2.xlsx')


# In[65]:


data1


# In[66]:


covid=data1.groupby('Month')['new_cases'].sum()


# In[67]:


covid


# In[73]:


month=[1,2,3,4,5,6,7,8,9,10,11,12]
cases=[1,
     2,
      1394,
       33466,
      155746,
      394872,
     1110507,
     1995178,
     2621418,
    1871498,
    1278727,
    803865 ]


# In[74]:


month1=pd.DataFrame(month,index=[1,2,3,4,5,6,7,8,9,10,11,12],columns=['Month'])
cases1=pd.DataFrame(cases,index=[1,2,3,4,5,6,7,8,9,10,11,12],columns=['cases'])


# In[75]:


regr=LinearRegression()
regr.fit(month1,cases1)
y=(regr.coef_*14)+regr.intercept_
y


# 

# In[100]:


m,b=(month1,cases1)


# In[95]:


plt.scatter(month1,cases1,color='red')
plt.plot(month1,regr_line)


# In[71]:


month=[1,2,3,4,5,6,7,8,9,10,11,12]
cases=[1,
     2,
      1394,
       33466,
      155746,
      394872,
     1110507,
     1995178,
     2621418,
    1871498,
    1278727,
    803865 ]


# In[78]:


correl=[(1,1),(2,2),(3,1394),(4,33466),(5,155746),(6,394872),(7,1110507),(8,1995178),(9,2621418),(10,1871498),(11,1278727),(12,803865)]


# In[79]:


correl=pd.DataFrame(correl,columns=["Months","Covid cases"])
correl.corr()


# In[86]:


correl.plot(kind='scatter',x='Months',y='Covid cases')
plt.title("Covid cases in India in 2020")
plt.show()


# In[ ]:




