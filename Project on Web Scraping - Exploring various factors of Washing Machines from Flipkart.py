#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re 
import time

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


URL = 'https://www.flipkart.com/search?q=washing+machines&sid=j9e%2Cabm%2C8qx&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&as-pos=1&as-type=RECENT&suggestionId=washing+machines%7CWashing+Machines&requestId=565eafdf-ce09-41c2-b32b-e969088f53c2&as-searchtext=washing%20machines'


# In[3]:


page=requests.get(URL)


# In[4]:


page.content


# In[5]:


page_content = page.text
page_content


# In[6]:


soup=BeautifulSoup(page_content)


# In[7]:


soup


# In[8]:


print(soup.prettify())


# In[9]:


x=soup.find_all('div', attrs={'class':'_3pLy-c row'})


# In[10]:


x[0].text


# In[11]:


#productname(pname)
soup.find_all('div', attrs={'class':'_4rR01T'})


# In[12]:


soup.find_all('div', attrs={'class':'_4rR01T'})[0].text


# In[13]:


#Extracting product names 
for x in soup.find_all('div', attrs={'class':'_3pLy-c row'}):
    pname=x.find('div', attrs={'class':'_4rR01T'})
    if pname is None:
        print(np.NaN)
    else:
        print(pname.text)


# In[14]:


#price
soup.find_all('div', attrs={'class' :'_30jeq3 _1_WHN1'})


# In[15]:


#Extracting prices
for x in soup.find_all('div', attrs={'class':'_3pLy-c row'}):
    price=x.find('div', attrs={'class':'_30jeq3 _1_WHN1'})
    if price is None:
        print(np.NaN)
    else:
        print(price.text)


# In[16]:


#ratings
soup.find_all('div', attrs={'class':'_3LWZlK'})


# In[17]:


#Extracting ratings
for x in soup.find_all('div', attrs={'class':'_3pLy-c row'}):
    rat=x.find('div', attrs={'class':'_3LWZlK'})
    if rat is None:
        print(np.NaN)
    else:
        print(rat.text)


# In[18]:


#specs
soup.find_all('div', attrs={'class':'fMghEO'})


# In[19]:


soup.find_all('div', attrs='fMghEO')[0].text


# In[20]:


#Extracting specs
for x in soup.find_all('div', attrs={'class':'_3pLy-c row'}):
        spec=x.find('div', attrs={'class':'fMghEO'})
        if spec is None:
            print(np.NaN)
        else:
            print(spec.text)


# In[21]:


#Extracting all the pages
for i in range(1,24):
    URL = 'https://www.flipkart.com/search?q=washing+machines&sid=j9e%2Cabm%2C8qx&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&as-pos=1&as-type=RECENT&suggestionId=washing+machines%7CWashing+Machines&requestId=565eafdf-ce09-41c2-b32b-e969088f53c2&as-searchtext=washing%20machines&page={}'.format(i)
    print(URL)


# In[22]:


product_name=[]
prices=[]
ratings=[]
specifications=[]
pagenum=[]
for i in range(1,24):
    t=time.time()
    URL = 'https://www.flipkart.com/search?q=washing+machines&sid=j9e%2Cabm%2C8qx&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_16_na_na_na&as-pos=1&as-type=RECENT&suggestionId=washing+machines%7CWashing+Machines&requestId=565eafdf-ce09-41c2-b32b-e969088f53c2&as-searchtext=washing%20machines&page={}'.format(i)
    page=requests.get(URL)
    page_content=page.text
    soup=BeautifulSoup(page_content)
    for x in soup.find_all('div', attrs={'class':'_3pLy-c row'}):
        pname=x.find('div', attrs={'class':'_4rR01T'})
        price=x.find('div', attrs={'class':'_30jeq3 _1_WHN1'})
        rat=x.find('div', attrs={'class':'_3LWZlK'})
        spec=x.find('div', attrs={'class':'fMghEO'})
        if pname is None:
            product_name.append(np.NaN)
        else:
            product_name.append(pname.text)
            
        if price is None:
            prices.append(np.NaN)
        else:
            prices.append(price.text)
            
        if rat is None:
            ratings.append(np.NaN)
        else:
            ratings.append(rat.text)
            
        if spec is None:
            specifications.append(np.NaN)
        else:
            specifications.append(spec.text)
        pagenum.append(i)
    e=time.time()
    print('Page {} is completed in {} seconds'.format(i,e-t))


# In[23]:


len(product_name)
len(prices)
len(ratings)
len(specifications)


# In[24]:


df_wm=pd.DataFrame({'Product_Name':product_name, 'Prices':prices, 'Ratings':ratings, 'Specifications':specifications, 'Page_No':pagenum})


# In[25]:


df_wm


# ## Regular Expression

# In[26]:


#Extracting Brand name from Prodcut name


# In[27]:


df_wm.Product_Name[0]


# In[28]:


re.findall(r'^\w+', df_wm.Product_Name[0])


# In[29]:


' '.join(re.findall(r'^\w+', df_wm.Product_Name[0]))


# In[30]:


df_wm['Brand']=df_wm['Product_Name'].apply(lambda x : ' '.join(re.findall(r'^\w+',x)))


# In[31]:


df_wm


# In[32]:


df_wm['Brand'].value_counts()


# In[33]:


df_wm.Product_Name[11]


# In[34]:


df_wm.Product_Name[68]


# In[35]:


#Extracting Capacity from Product Name
re.findall('[0-9+.]{1,}\s(?:kg|Kg)', df_wm.Product_Name[360])


# In[36]:


" ".join(re.findall('[0-9+.]{1,}\s(?:kg|Kg)', df_wm.Product_Name[389]))


# In[37]:


df_wm['Capacity_in_kg']=df_wm['Product_Name'].apply(lambda x :" ".join(re.findall('[0-9+.]{1,}\s(?:kg|Kg)', x)))


# In[38]:


df_wm


# In[39]:


#Extracting Load Type from Product Name
df_wm['Product_Name'][5]


# In[40]:


re.findall('(?:Top|Front)+\s(?:Load)', df_wm['Product_Name'][5] )


# In[41]:


' '.join(re.findall('(?:Top|Front)+\s(?:Load)', df_wm['Product_Name'][5] ))


# In[42]:


df_wm['Load_Type']=df_wm['Product_Name'].apply(lambda x : ' '.join(re.findall('(?:Top|Front)+\s(?:Load)', x)))


# In[43]:


pd.options.display.max_rows=600
df_wm


# In[44]:


df_wm['Specifications'][54]


# In[45]:


df_wm['Specifications'][7]


# In[46]:


#Extracting Rotations per Minute(rpm) from Specifications
re.findall('[0-9]+\s(?:rpm|RPM)', df_wm['Specifications'][7])


# In[47]:


df_wm['Specifications'][367]


# In[48]:


' '.join(re.findall('[0-9]+\s(?:rpm|RPM)', df_wm['Specifications'][367]))


# In[49]:


df_wm['RPM']=df_wm["Specifications"].apply(lambda x : ' '.join(re.findall('[0-9]+\s(?:rpm|RPM)+', x )))


# In[50]:


df_wm


# In[51]:


#Extracting star from data
df_wm['Specifications'][0]


# In[52]:


re.findall('[0-9][\s]+Star', df_wm['Specifications'][0])


# In[53]:


' '.join(re.findall('[0-9][\s]+Star', df_wm['Specifications'][6]))


# In[54]:


df_wm['Star']=df_wm['Specifications'].apply(lambda x : ' '.join(re.findall('[0-9][\s]+Star',x)))


# In[55]:


df_wm


# In[56]:


import os
os.getcwd()


# In[57]:


df_wm.to_csv('C:\\Users\\dell\\df_mw.csv')


# ## Pickling

# In[58]:


import joblib
joblib.dump(df_wm,'C:\\Users\\dell\\webscrape.pkl')


# In[59]:


pd.options.display.max_rows=600
df_wm


# ## Data Cleaning

# In[60]:


df_wm.isnull().sum()


# In[61]:


type(df_wm['Ratings'][4])


# In[62]:


#Columns
df_wm.columns


# In[63]:


#Data Types of Columns
df_wm['Brand'][0]
type(df_wm['Brand'][0])


# In[64]:


df_wm['Prices'][0]
type(df_wm['Prices'][0])


# In[65]:


df_wm['Ratings'][3]
type(df_wm['Ratings'][0])


# In[66]:


df_wm['Capacity_in_kg'][0]
type(df_wm['Capacity_in_kg'][0])


# In[67]:


df_wm['Load_Type'][0]
type(df_wm['Load_Type'][0])


# In[68]:


df_wm['RPM'][0]
type(df_wm['RPM'][0])


# In[69]:


df_wm['Star'][0]
type(df_wm['Star'][0])


# In[70]:


# Replacing '₹' with '' and ',' with '' in Price
float(df_wm['Prices'][0].replace('₹', '').replace(',',''))


# In[71]:


df_wm['Prices']=df_wm['Prices'].apply(lambda x : x.replace('₹','').replace(',', '')).astype(float)


# In[72]:


# Replacing 'kg' with '' in Capacity_in_kg
float(df_wm['Capacity_in_kg'][1].replace('kg', '').replace(' ', ''))


# In[73]:


df_wm['Capacity_in_kg']=df_wm['Capacity_in_kg'].apply(lambda x : x.replace('kg','').replace('Kg','').replace(' ', ''))


# In[74]:


for i in range(0,len(df_wm['Capacity_in_kg'])):
    a=df_wm['Capacity_in_kg'][i]
    if a=='NaN':
        print(i,a)


# In[75]:


for i in range(0,len(df_wm['Capacity_in_kg'])):
    a=df_wm['Capacity_in_kg'][i]
    if a=='':
        df_wm['Capacity_in_kg'][i]=(np.NaN)


# In[76]:


df_wm['Capacity_in_kg']=df_wm['Capacity_in_kg'].astype(float)


# In[77]:


#Replacing 'rpm' with '' 
float(df_wm['RPM'][7].replace('RPM', 'rpm').replace('rpm', '').replace(' ', ''))


# In[78]:


df_wm['RPM']=df_wm['RPM'].apply(lambda x : x.replace('RPM','rpm').replace('rpm', '').replace(' ', ''))


# In[79]:


df_wm['RPM']=df_wm['RPM'].astype(float)


# In[80]:


df_wm['Star']=df_wm['Star'].apply(lambda x : x.replace('Star','').replace(' ', ''))


# In[81]:


for i in range(0,len(df_wm['Star'])) :
    c=df_wm['Star'][i]
    if c=='':
        df_wm['Star'][i]=0


# In[82]:


df_wm['Star']=df_wm['Star'].astype(float)


# In[83]:


df_wm['Ratings']=df_wm['Ratings'].astype(float)


# In[92]:


df_wm


# In[5]:


df_wm.to_csv('C:\\Users\\dell\\df_wm.csv')


# In[3]:


df_wm=pd.read_csv(r"C:\Users\dell\df_wm.csv")


# In[4]:


df_wm


# In[6]:


df_wm.dtypes


# In[8]:


df_wm=df_wm.drop(['Unnamed: 0'],axis=1)


# In[9]:


df_wm


# In[10]:


df_wm.shape


# In[11]:


df_wm.isnull().sum()


# In[12]:


#filling null values in Ratings column with mean of Ratings
df_wm['Ratings'].mean()


# In[13]:


round(df_wm['Ratings'].mean(),1)


# In[10]:


df_wm['Ratings']=df_wm['Ratings'].fillna(round(df_wm['Ratings'].mean(),1))


# In[11]:


df_wm['Ratings'].isnull().sum()


# In[12]:


#filling null values in Capacity_in_kg column with mode of Capacity_in_kg
df_wm['Capacity_in_kg'].mode()[0]


# In[13]:


df_wm['Capacity_in_kg']=df_wm['Capacity_in_kg'].fillna(df_wm['Capacity_in_kg'].mode()[0])


# In[14]:


df_wm['Capacity_in_kg'].isnull().sum()


# In[15]:


df_wm=df_wm.dropna(how='any', subset=['Load_Type'])


# In[16]:


df_wm['Load_Type'].isnull().sum()


# In[17]:


df_wm.isnull().sum()


# In[18]:


df_wm.shape


# In[19]:


df_wm[df_wm["RPM"].isin([0.0])]


# ## Data Visualisation

# In[20]:


df_wm


# ## Univariate Analysis

# ### Bar Plot

# In[21]:


df_wm['Brand'].value_counts().plot.bar()


# ### Count Plot

# In[22]:


plt.figure(figsize=(15,5))
plt.title('Number of products from different Brands')
sns.countplot(df_wm['Brand'])
plt.xticks(rotation=90);


# ### Insights:
# #### In this countplot, we can see the frequency of the products of each Brand and we can also observe 'LG' has the maximum number of products.

# In[23]:


plt.title('Count of products in each Load Type')
sns.countplot(df_wm['Load_Type'])


# ### Insights:
# #### Here when we plot the graph on Load Type, we can observe that the washing machines with Top Load are more when compared with the Front Load.

# In[24]:


plt.figure(figsize=(15,5))
plt.title("Count of products with different RPM's")
sns.countplot(df_wm['RPM'])
plt.xticks(rotation=90);


# ### Insights :
# #### In this countplot, we can say that the washing machines with 1400 RPM has the highest count while the machines with 55.0 RPM, 705.0 RPM,  730.0 RPM, 850.0 RPM and 1320.0 RPM has the least count.

# ### Box Plot

# In[25]:


sns.boxplot(df_wm.Prices)


# ### Insights :
# #### In this boxplot, the data is right_skewed and we can also observe that minimum price is below 10000 and the maximum price is 50000 and there are outliers present in this data. 

# In[26]:


sns.boxplot(df_wm.Ratings)


# ### Insights :
# #### From this boxplot, we can say that the distribution is fairly even with the median being in the middle of the box and the whiskers are of similar size.There are also low and high outliers.

# In[27]:


df_wm[df_wm['Star'].isin([1.0, 2.0])]


# In[ ]:





# ### Histogram

# In[28]:


plt.title('Histogram on "Star" rating')
plt.hist(df_wm['Star'])


# ### Insights :
# #### Here Histogram is plotted for 'Star' ratings of the product. 'Star' rating represents the Efficiency of the product. From the graph we can observe that there are no products with 1 Star and 2 Star while the products with 5 Star rating has the highest count.

# In[29]:


plt.title('Histogram on Prices of the product')
plt.hist(df_wm['Prices']);


# ### Insights :
# #### From the above histogram plot, we can say that the price of washing machine approximately ranges b/w 5000 to 75000, while the maximum number washing machines price ranges b/w 15000 to 20000 

# In[30]:


plt.figure(figsize=(10,4))
plt.title('Histogram Plot on Capacity of Washing Machines')
plt.hist(df_wm['Capacity_in_kg']);


# ### Insights :
# #### From the above Histogram bar, we can observe that the Capacity of Washing machines ranges b/w 5kg to 18kg and the capacity of more number of washing machines ranges b/w 7kg to 9kg.

# ### Distribution Plot

# In[31]:


plt.figure(figsize=(10,4))
sns.distplot(df_wm['Ratings'])
plt.title('Distribution Plot for Ratings')


# ### Insights :
# #### From this ditribution plot we can observe that the more number of ratings are b/w 4.0 to 4.5 and less number of ratings are b/w 2.0 to 3.5

# In[32]:


plt.figure(figsize=(10,4))
sns.distplot(df_wm['Prices'])
plt.title("Distribution of Price")


# ### Insights :
# #### The distribution plot is plotted for Price of washing machines. Here we can observe that the price distribution is high in b/w 10000 to 20000.

# In[33]:


plt.figure(figsize=(10,4))
sns.distplot(df_wm['RPM'])
plt.title('Distribution Plot for RPM')


# ### Insights :
# #### From this plot we can say that the more number of machines RPM ranges b/w 600 to 750 and 1350 to 1500 and there are less number of machines with RPM ranges b/w 250 to 500.

# ### Violin Plot

# In[35]:


plt.title('Variation in Price')
sns.violinplot(df_wm["Prices"])


# ### Insights :
# #### Here we can observe that maximum number of Washing machines are in b/w the price range of 10000 to 20000.

# In[36]:


plt.title('Variation in Capacity')
sns.violinplot(df_wm["Capacity_in_kg"])


# ### Insights :
# #### From this violin plot, we can observe that th maximum number of washing machines are with Capacity of 6kg tp 8kg. 

# ## Bivariate Analysis

# ### Box Plot

# In[37]:


plt.figure(figsize=(15,5))
sns.boxplot(x=df_wm.Brand, y = df_wm.Prices)
plt.xticks(rotation=90);


# ### Insights:
# #### In this box plot, we can observe that 'Whirlpool' Brand has the more number of outliers while the Brands 'Gangnam', 'Micromax', 'Toshiba' and 'Candy' has the stable prices.

# In[38]:


sns.boxplot(x = df_wm['Load_Type'],
            y = df_wm['Prices'])


# ### Insights : 
# #### From this boxplot, it is observed that the Top Load Washing Machines price ranges from 10000 to 20000 and Front Load Washing Machines price ranges from 28000 to 35000.

# ### Scatter Plot

# In[39]:


sns.scatterplot(df_wm["Ratings"],df_wm["Prices"])
plt.title('Rating vs Price')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.xticks(rotation=90);


# ### Insights :
# #### Here we can observe that there are more number of washing machine Ratings are b/w 4.0 to 4.5 with Price range b/w 10000 to 35000. 

# In[40]:


sns.scatterplot(df_wm["Capacity_in_kg"],df_wm["Prices"])
plt.title('Capacity vs Price')
plt.xlabel('Capacity')
plt.ylabel('Price')
plt.xticks(rotation=90);


# ### Insights :
# #### From the plot, we can observe that washing machines with capacity 7kg to 9kg and with price range of 10000 to 35000 are more in number. 

# ### Pie Chart : Diagram consisting of a circle divided into parts to show the size of particular parts in relation.

# In[41]:


data = df_wm.groupby('Load_Type')['Prices'].sum()


# In[42]:


data.plot.pie(autopct='%.1f%%', startangle=80)


# ### Insights :
# #### From the above pie chart, we can say that the machines with Top Load type are more than machines with Front load type.

# In[22]:


df_wm_50=df_wm.head(50)


# In[23]:


df_wm_50


# In[ ]:





# In[24]:


data = df_wm_50.groupby('Ratings')['Prices'].sum()


# In[25]:


data.plot.pie(autopct='%.1f%%')


# ### Insights :
# #### The Pie Chart for Ratings of 50 washing machines says that 93.8% of the machines are with Ratings of 4.2 to 4.4 while the machines with Rating 4.4 alone contributes 41.5% of the machines and the rest of the % of machines are with 4.0, 4.1 and 4.5 ratings.

# In[26]:


data = df_wm_50.groupby('Capacity_in_kg')['Prices'].sum()


# In[27]:


data.plot.pie(autopct='%.1f%%')


# ### Insights:
# #### This is a piechart of first 50 Washing machines for Capacity. Here we can observe that highest number of washing machines are with capacity 8.0kg and 7.0kg and least number of washing machines are with capacity 7.8kg.

# ### Heat Map

# In[47]:


plt.figure(figsize=(15,7))
sns.heatmap(df_wm.corr(), annot=True)
plt.title('Correlation of different Components of Washing Machine')
plt.show();


# ### Insights :
# #### In this Heat Map, the lighter color represents the strongest correlation b/w two variables and the darker color represents the weakest correlation b/w two variables.

# ### Pair plot :
# #### It is used to plot on multiple data in one plot.In this we are showing the three data variables of Price, Star and Rating, you can the differences in each plot.

# In[48]:


sns.pairplot(data = df_wm, vars=['Prices','Star','Ratings'])
plt.show();


# In[49]:


df_wm.groupby(['Brand'], as_index=False).agg({'Prices':['min', 'max', 'mean']})


# ### Continuous vs Categorical

# ### Bar Plot

# In[50]:


plt.figure(figsize=(15,4))
df_wm.groupby(['Brand'])['Prices'].max().plot(kind = 'bar')
plt.title('Maximum price of Each brand')


# ### Insights :
# #### In this plot we are using Barplot, we took two components Brand and Price.The plot shows the maximum values of the price in each brand by using groupby method.
# 

# In[51]:


plt.figure(dpi = 100, figsize=(15,4))
df_wm.groupby(['Brand'])['RPM'].mean().plot(kind = 'bar');
plt.title('Average RPM from each Brand')


# ### Insights :
# #### The plot shows the maximum values of the RPM in each brand. Here we can observe that almost all the Brands have the washing machines with maximum RPM greater than 1200.¶

# In[52]:


plt.figure(dpi = 100, figsize=(15,4))
df_wm.groupby(['Brand'])['Capacity_in_kg'].min().plot(kind = 'bar');
plt.title('Minimum Capacity from each Brand')


# ### Insights :
# #### The plot shows the minimum values of the Capacity in each brand. Here we can observe that almost all the Brands have the washing machines with minimum Capacity of 6kg.

# ### Categorical vs Categorical

# In[53]:


crosstab=pd.crosstab(df_wm['Brand'], df_wm['Load_Type'])


# In[54]:


crosstab


# In[55]:


crosstab.plot(kind='bar', stacked=True)


# ### Insights :
# #### The Stacked Bar Plot is plotted b/w 'Brand' and 'Load Type'. Here Blue color shows the count of Front Load Washing Machines in each Brand and the Orange color shows the Top Load Washing machines in each brand.

# ### Continuous vs Continuous

# In[56]:


sns.scatterplot(df_wm['Prices'], df_wm['Capacity_in_kg'])


# In[57]:


df_wm[["Prices", 'RPM', 'Capacity_in_kg']].corr()


# In[58]:


sns.heatmap(df_wm[["Prices", 'RPM', 'Capacity_in_kg']].corr(),annot=True)


# ### Multivariate Analysis :

# In[59]:


sns.scatterplot(df_wm['Prices'], df_wm['Capacity_in_kg'], hue=df_wm['Load_Type'])
plt.legend(facecolor='pink', edgecolor='black', shadow=True)


# ### Insights :
# #### Scatter Plot is plotted b/w 'Price' and 'Capacity' by differentiating 'Load Type'. Here Blue color represents the Top Load Washing Machines and Orange color represents the Front Load Washing Machines.

# In[ ]:




