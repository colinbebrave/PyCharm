
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[6]:

import numpy as np
import pandas as pd
import os
os.chdir('/Users/wangqi/Documents/PyCharm/Data_Science_in_Python')


# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[3]:

def answer_one():
    #loading the .xls file of interest
    energy = pd.read_excel('Energy Indicators.xls','Energy', na_values = ["..."])
    #slicing data, we just keep one part of the whole worksheet for that the others contains irrelevant information
    energy = energy.iloc[16:243, 2:]
    #rename the columns
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy['Energy Supply'] = energy['Energy Supply'] * 1000000
    #wo need to have a look at the names before we start transforming them
    #'United States of America' in energy['Country'].unique()
    #maybe there are some other issues, i.e., maybe the country names are not standard
    #remove numbers from strings represents countries.
    energy['Country'] = energy['Country'].str.replace('\d+', '')
    #'United States of America' in energy['Country'].unique()
    #ok, now numbers have been removed
    #but there are still parentheses.
    energy['Country'] = energy['Country'].str.replace(r'\(.*\)+', '')
    the_dict = {"Republic of Korea": "South Korea",
                "United States of America": "United States",
                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                "China, Hong Kong Special Administrative Region": "Hong Kong"}
    #rename countries using dictionary
    energy['Country'] = energy['Country'].replace(the_dict, regex = True)
    energy['Country'] = energy['Country'].str.strip()
    energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].astype(np.float64)
    
    
    #read gdp data
    GDP = pd.read_csv('world_bank.csv')
    #GDP.head()
    
    #change column names
    GDP.columns = GDP.iloc[3,]
    
    #slice the part that is of interest
    GDP = GDP.iloc[4:, ].reset_index(drop = True)
    
    GDP.columns = list(GDP.columns[:4]) + list(map(int, GDP.columns[4:]))
    another_dict = {"Korea, Rep.": "South Korea", 
                    "Iran, Islamic Rep.": "Iran",
                    "Hong Kong SAR, China": "Hong Kong"}
    GDP['Country Name'] = GDP['Country Name'].replace(another_dict, regex = True)
    
    
    #read ScimEn data
    ScimEn = pd.read_excel('scimagojr-3.xlsx', 'Sheet1')
    
    #--------------------preparing for merging dataframeS-------------------
    ScimEn_15 = ScimEn.iloc[:15, :]
    GDP_10 = GDP.loc[:,['Country Name']+ list(range(2006, 2016))]
    GDP_10.columns = ['Country'] + list(map(str, list(range(2006, 2016))))
    #ScimEn = ScimEn.set_index('Country', drop = True)
    #energy = energy.set_index('Country', drop = True)
    #GDP_10 = GDP_10.set_index('Country', drop = True)

    return pd.merge(pd.merge(ScimEn_15, energy, on = 'Country'), GDP_10, on = 'Country').set_index('Country', drop = 'True')


# In[7]:

df = answer_one(); df


# In[8]:

energy = pd.read_excel('Energy Indicators.xls','Energy', na_values = ["..."])
energy = energy.iloc[16:243, 2:]
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy['Energy Supply'] = energy['Energy Supply'] * 1000000
#wo need to have a look at the names before we start transforming them
#'United States of America' in energy['Country'].unique()
#maybe there are some other issues, i.e., maybe the country names are not standard
#remove numbers from strings represents countries.
energy['Country'] = energy['Country'].str.replace('\d+', '')
#'United States of America' in energy['Country'].unique()
#ok, now numbers have been removed
#but there are still parentheses.
energy['Country'] = energy['Country'].str.replace(r'\(.*\)+', '')
the_dict = {"Republic of Korea": "South Korea",
            "United States of America": "United States",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "China, Hong Kong Special Administrative Region": "Hong Kong"}
#rename countries using dictionary
energy['Country'] = energy['Country'].replace(the_dict, regex = True)
energy['Country'] = energy['Country'].str.strip()


#read gdp data
GDP = pd.read_csv('world_bank.csv')
#GDP.head()

#change column names
GDP.columns = GDP.iloc[3,]

#slice the part that is of interest
GDP = GDP.iloc[4:, ].reset_index(drop = True)

GDP.columns = list(GDP.columns[:4]) + list(map(int, GDP.columns[4:]))
another_dict = {"Korea, Rep.": "South Korea", 
                "Iran, Islamic Rep.": "Iran",
                "Hong Kong SAR, China": "Hong Kong"}
GDP['Country Name'] = GDP['Country Name'].replace(another_dict, regex = True)

    
#read ScimEn data
ScimEn = pd.read_excel('scimagojr-3.xlsx', 'Sheet1')


# In[9]:

'China' in energy['Country']


# In[10]:

'China' in energy['Country'].unique()


# In[11]:

'Iran' in energy['Country'].unique()


# In[12]:

energy['Country'] = energy['Country'].str.strip()


# Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[15]:

def answer_two():
    Top15 = answer_one()
    return np.mean(Top15.loc[:,'2006':], axis = 1).sort_values(ascending = False)


# In[16]:

answer_two()


# In[17]:

df = answer_one()


# In[18]:

np.mean(df, axis = 1).sort_values(ascending = False)


# In[19]:

np.mean(df.loc['Iran'][10:])


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[20]:

def answer_three():
    Top15 = answer_one()
    avg_Top15 = answer_two()
    gdp_uk_2006 = Top15.loc[avg_Top15.index[5], '2006']
    gdp_uk_2015 = Top15.loc[avg_Top15.index[5], '2015']
    return gdp_uk_2015 - gdp_uk_2006


# In[21]:

answer_three()


# In[22]:

Top15 = answer_one()
avg_Top15 = answer_three()


# In[23]:

avg_Top15.index[5]


# In[24]:

Top15.loc[avg_Top15.index[5]]


# In[25]:

gdp_uk_2006 = Top15.loc[avg_Top15.index[5], '2006']
gdp_uk_2015 = Top15.loc[avg_Top15.index[5], '2015']


# In[318]:

#growth_gdp_uk = (gdp_uk_2015 - gdp_uk_2006) / gdp_uk_2015; growth_gdp_uk


# In[26]:

gdp_uk_2015 - gdp_uk_2006


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[27]:

def answer_four():
    Top15 = answer_one()
    avg_energy_per_capita = np.mean(Top15['Energy Supply per Capita'])
    return avg_energy_per_capita


# In[28]:

answer_four()


# In[29]:

Top15


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[30]:

def answer_five():
    Top15 = answer_one()
    Top15 = Top15.sort_values(by = '% Renewable', ascending = False).reset_index()
    return Top15['Country'][0], Top15['% Renewable'][0]


# In[31]:

answer_five()


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[32]:

Top15.columns


# In[33]:

def answer_six():
    Top15 = answer_one()
    Top15['self_citations_ratio'] = Top15['Self-citations'] / Top15['Citations']
    Top15 = Top15.sort_values(by = 'self_citations_ratio', ascending = False).reset_index()
    return Top15['Country'][0], Top15['self_citations_ratio'][0]


# In[34]:

answer_six()


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[35]:

def answer_seven():
    Top15 = answer_one()
    if (Top15['Energy Supply'].isnull().values.any() == False) & (Top15['Energy Supply per Capita'].isnull().values.any() == False):
        Top15['Estimated_pop_by_energy'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
        Top15 = Top15.sort_values('Estimated_pop_by_energy', ascending = False)
        Top15_1 = Top15.iloc[:3]
    return Top15_1.index[2]


# In[36]:

answer_seven()


# In[37]:

Top15 = answer_one()
Top15['Estimated_pop_by_energy'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
Top15 = Top15.sort_values('Estimated_pop_by_energy', ascending = False)
Top15_1 = Top15.iloc[:3]


# In[38]:

Top15_1.index[2]


# In[39]:

Top15['Energy Supply'].isnull().values.any()


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[40]:

Top15.head(2)


# In[41]:

def answer_eight():
    Top15 = answer_one()
    # if (Top15['Energy Supply'].isnull().values.any() == False) & (Top15['Energy Supply per Capita'].isnull().values.any() == False):
    Top15['Estimated_pop_by_energy'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['citable_documents_per_capita'] = Top15['Citable documents'] / Top15['Estimated_pop_by_energy']
    the_corr_coef = Top15['citable_documents_per_capita'].corr(Top15['Energy Supply per Capita'])

    return the_corr_coef


# In[42]:

answer_eight()


# In[43]:

def plot9():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[19]:

#plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[44]:

def answer_nine():
    Top15 = answer_one()
    Top15['HighRenew'] = Top15['% Renewable'] >= Top15['% Renewable'].median()
    Top15['HighRenew_num'] = Top15['HighRenew'] * 1
    return Top15['HighRenew_num']

answer_nine()
# In[534]:

# Top15_1 = Top15[Top15['HighRenew_num'] == 1]
# Top15_1


# In[45]:

Top15['HighRenew'] = Top15['% Renewable'] >= Top15['% Renewable'].median()
Top15['HighRenew_num'] = Top15['HighRenew'] * 1
Top15.head(2)


# In[533]:

# Top15['HighRenew'] = [1 if Top15['% Renewable'] >= Top15['% Renewable'].mean() else 0]


# In[46]:

Top15['% Renewable'] >= Top15['% Renewable'].median()


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[47]:

def answer_ten():
    Top15 = answer_one()
    Top15['Estimated_pop_by_energy'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Country'] = Top15['Country'].replace(ContinentDict, regex = True)
    return Top15[['Country', 'Estimated_pop_by_energy']].groupby(['Country']).agg(['size', 'sum', 'mean', 'std'])['Estimated_pop_by_energy']


# In[48]:

answer_ten()


# In[50]:

ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}


# In[51]:

dddd = answer_one()
dddd['Estimated_pop_by_energy'] = dddd['Energy Supply'] / dddd['Energy Supply per Capita']
dddd = dddd.reset_index()
dddd['Country'] = dddd['Country'].replace(ContinentDict, regex = True)
dddd


# In[52]:

dddd[['Country', 'Estimated_pop_by_energy']].groupby(['Country']).agg(['sum', 'mean', 'std', 'size'])['Estimated_pop_by_energy']


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[53]:

def answer_eleven():
    Top15 = answer_one()
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Continent'] = Top15['Country'].replace(ContinentDict, regex = True)
    Top15['binning_renew'] = pd.cut(Top15['% Renewable'], 5)
    return Top15[['Continent', 'Country', 'binning_renew']].groupby(['Continent', 'binning_renew']).size()


# In[54]:

type(answer_eleven())
answer_eleven()

# In[55]:

Top15 = answer_one()
ContinentDict  = {'China':'Asia', 
              'United States':'North America', 
              'Japan':'Asia', 
              'United Kingdom':'Europe', 
              'Russian Federation':'Europe', 
              'Canada':'North America', 
              'Germany':'Europe', 
              'India':'Asia',
              'France':'Europe', 
              'South Korea':'Asia', 
              'Italy':'Europe', 
              'Spain':'Europe', 
              'Iran':'Asia',
              'Australia':'Australia', 
              'Brazil':'South America'}
Top15 = Top15.reset_index()
Top15['Continent'] = Top15['Country'].replace(ContinentDict, regex = True)
Top15['binning_renew'] = pd.cut(Top15['% Renewable'], 5)


# In[56]:

Top15.head()


# In[57]:

Top15[['Continent', 'Country', 'binning_renew']].groupby(['Continent', 'binning_renew']).size()


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[58]:

def answer_twelve():
    Top15 = answer_one()
    Top15['Estimated_pop_by_energy'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Estimated_pop'] = Top15['Estimated_pop_by_energy'].apply(lambda x: "{:,}".format(x))
    return Top15['Estimated_pop']


# In[59]:

answer_twelve()


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[24]:

def plot_optional():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")


# In[25]:

#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!

