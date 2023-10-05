#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis of the Ames dataset

# In[1]:


import pathlib
import pickle
import requests

import pandas as pd


# It is a good idea to define a variable for the base data directory, and to construct the exact filenames from that variable. Another good idea is to use the `pathlib` library for manipulating paths in Python, as it will make your code work in both Windows and Linux/MacOS.

# In[2]:


DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)


# Make sure the path exists:

# In[3]:


DATA_DIR.mkdir(parents=True, exist_ok=True)


# ## Before we begin

# Let's download the data, read it, check if things appear ok (if we at least read the data), list the columns and their data type, and correct their types in Pandas.

# ### Download the data
# 
# It is a good idea to automate the downloading of the data, it makes everyone's life easier. Just be careful to avoid downloading multiple times.

# In[4]:


raw_data_dir = DATA_DIR / 'raw'
raw_data_dir.mkdir(parents=True, exist_ok=True)
print(raw_data_dir)


# In[5]:


raw_data_file_path = DATA_DIR / 'raw' / 'ames.csv'
print(raw_data_file_path)


# In[6]:


if not raw_data_file_path.exists():
    source_url = 'https://www.openintro.org/book/statdata/ames.csv'
    headers = {
        'User-Agent': \
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) ' \
            'AppleWebKit/537.36 (KHTML, like Gecko) ' \
            'Chrome/39.0.2171.95 Safari/537.36',
    }
    response = requests.get(source_url, headers=headers)
    csv_content = response.content.decode()
    with open(raw_data_file_path, 'w', encoding='utf8') as file:
        file.write(csv_content)


# ### Reading the data, sanity checks

# Check the size of the file for good measure.

# In[7]:


filesize = raw_data_file_path.stat().st_size
print(f'This file has {filesize} bytes')


# You should see that the file has 1182303 bytes. All good, let's read the file.

# In[8]:


raw_data = pd.read_csv(raw_data_file_path)


# Check the number of rows and columns, and print a few samples to see if the reading went well.

# In[9]:


raw_data.shape


# In[10]:


raw_data.head()


# ### Analyze the column types

# From now on, `raw_data` is untouchable. Make a copy of `raw_data`:

# In[11]:


data = raw_data.copy()


# Let's see the columns and types:

# In[12]:


data.info()


# Pay attention to the various types of data:
# 
# - `int64`: integer-valued variable. May be a categorical variable (when the integer number actually represents a non-ordered category) or a numerical variable (it should have been a `float64` instead).
# - `float64`: real-valued variable.
# - `object`: a generic object in Pandas - could be a list, a dictionary, an object from a complicated class, etc. But if your DataFrame was just read from a CSV file, then the `object` type actually refers to a _string_ (type `str`). Usually represents a categorical variable but be careful - sometimes we have ordinal variables, where the categories actually have an ordering. Ordinal variables may be modeled as categorical or converted to numerical, and it is not always clear which approach is better.

# In[13]:


data.dtypes.value_counts()


# Go back to the documentation for this dataset and analyze the columns. The documentation is well written, and it is an example of how to document variables. Notice that the file describes the variables by meaningful groups, somehow: 
# 
# - `Order` and `PID` are line identifiers, and should have no bearing on house price prediction.
# 
# - The next set of variable relates to the conditions of the surroundings of the house: `MS.SubClass`, `MS.Zoning`, `Lot.Frontage`, `Lot.Area`, `Street`, `Alley`, `Lot.Shape`, `Land.Contour`, `Utilities`, `Lot.Config`, `Land.Slope`, `Neighborhood`, `Condition.1`, `Condition.2`.
# - After that, the documentation presents the (quite extensive) list of features of the house: `Bldg.Type`, `House.Style`, `Overall.Qual`, `Overall.Cond`, `Year.Built`, `Year.Remod.Add`, `Roof.Style`, `Roof.Matl`, `Exterior.1st`, `Exterior.2nd`, `Mas.Vnr.Type`, `Mas.Vnr.Area`, `Exter.Qual`, `Exter.Cond`, `Foundation`, `Bsmt.Qual`, `Bsmt.Cond`, `Bsmt.Exposure`, `BsmtFin.Type.1`, `BsmtFin.SF.1`, `BsmtFin.Type.2`, `BsmtFin.SF.2`, `Bsmt.Unf.SF`, `Total.Bsmt.SF`, `Heating`, `Heating.QC`, `Central.Air`, `Electrical`, `X1st.Flr.SF`, `X2nd.Flr.SF`, `Low.Qual.Fin.SF`, `Gr.Liv.Area`, `Bsmt.Full.Bath`, `Bsmt.Half.Bath`, `Full.Bath`, `Half.Bath`, `Bedroom.AbvGr`, `Kitchen.AbvGr`, `Kitchen.Qual`, `TotRms.AbvGrd`, `Functional`, `Fireplaces`, `Fireplace.Qu`, `Garage.Type`, `Garage.Yr.Blt`, `Garage.Finish`, `Garage.Cars`, `Garage.Area`, `Garage.Qual`, `Garage.Cond`, `Paved.Drive`, `Wood.Deck.SF`, `Open.Porch.SF`, `Enclosed.Porch`, `X3Ssn.Porch`, `Screen.Porch`, `Pool.Area`, `Pool.QC`, `Fence`, `Misc.Feature`, `Misc.Val`.
# - Finally, attributes related to the sale are given: `Mo.Sold`, `Yr.Sold`, `Sale.Type`, `Sale.Condition`, `SalePrice`.

# Make a list of all categorical, ordinal, continuous, and discrete variables (and the two variables to ignore). Follow the documentation.

# In[14]:


ignore_variables = [
    'Order',
    'PID',
]

continuous_variables = [
    'Lot.Frontage',
    'Lot.Area',
    'Mas.Vnr.Area',
    'BsmtFin.SF.1',
    'BsmtFin.SF.2',
    'Bsmt.Unf.SF',
    'Total.Bsmt.SF',
    'X1st.Flr.SF',
    'X2nd.Flr.SF',
    'Low.Qual.Fin.SF',
    'Gr.Liv.Area',
    'Garage.Area',
    'Wood.Deck.SF',
    'Open.Porch.SF',
    'Enclosed.Porch',
    'X3Ssn.Porch',
    'Screen.Porch',
    'Pool.Area',
    'Misc.Val',
    'SalePrice',
]

discrete_variables = [
    'Year.Built',
    'Year.Remod.Add',
    'Bsmt.Full.Bath',
    'Bsmt.Half.Bath',
    'Full.Bath',
    'Half.Bath',
    'Bedroom.AbvGr',
    'Kitchen.AbvGr',
    'TotRms.AbvGrd',
    'Fireplaces',
    'Garage.Yr.Blt',
    'Garage.Cars',
    'Mo.Sold',
    'Yr.Sold',
]

ordinal_variables = [
    'Lot.Shape',
    'Utilities',
    'Land.Slope',
    'Overall.Qual',
    'Overall.Cond',
    'Exter.Qual',
    'Exter.Cond',
    'Bsmt.Qual',
    'Bsmt.Cond',
    'Bsmt.Exposure',
    'BsmtFin.Type.1',
    'BsmtFin.Type.2',
    'Heating.QC',
    'Electrical',
    'Kitchen.Qual',
    'Functional',
    'Fireplace.Qu',
    'Garage.Finish',
    'Garage.Qual',
    'Garage.Cond',
    'Paved.Drive',
    'Pool.QC',
    'Fence',
]

categorical_variables = [
    'MS.SubClass',
    'MS.Zoning',
    'Street',
    'Alley',
    'Land.Contour',
    'Lot.Config',
    'Neighborhood',
    'Condition.1',
    'Condition.2',
    'Bldg.Type',
    'House.Style',
    'Roof.Style',
    'Roof.Matl',
    'Exterior.1st',
    'Exterior.2nd',
    'Mas.Vnr.Type',
    'Foundation',
    'Heating',
    'Central.Air',
    'Garage.Type',
    'Misc.Feature',
    'Sale.Type',
    'Sale.Condition',
]


# Drop the droppable variables:

# In[15]:


data.drop(columns=['Order', 'PID'], inplace=True)


# Now let's perform a first attempt to correct the data types for the given variables, setting the types as:
# 
# - continuous: `float64`
# - categorical: `category`
# - ordinal: This is the difficult one, it has to be of type `category`, but we need to define the order. One-by-one.
# - discrete: Let's set it as being of type `float64`, that is, we will interpret them as numerical quantities.

# In[16]:


for col in continuous_variables:
    data[col] = data[col].astype('float64')


# In[17]:


for col in categorical_variables:
    data[col] = data[col].astype('category')


# In[18]:


for col in discrete_variables:
    data[col] = data[col].astype('float64')


# Now the painful one...

# In[19]:


data[ordinal_variables].info()


# Attention to the `Overall.Qual` and `Overall.Cond` columns, their original values are integers, enter them accordingly. 
# 
# Also, some variables are said to have the category `NA` for situations when a feature is not available (e.g. condition of the basement when there is no basement). But, in reality, these occurences have been represented by the Pandas reading function as the Python constant `None`, correctly indicating that there is no value there. Therefore, you don't need to encode the `NA` category, it does not exist in our dataset.

# In[20]:


category_orderings = {
    'Lot.Shape': [
        'Reg',
        'IR1',
        'IR2',
        'IR3',
    ],
    'Utilities': [
        'AllPub',
        'NoSewr',
        'NoSeWa',
        'ELO',
    ],
    'Land.Slope': [
        'Gtl',
        'Mod',
        'Sev',
    ],
    'Overall.Qual': [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    'Overall.Cond': [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],
    'Exter.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Exter.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Bsmt.Exposure': [
        'Gd',
        'Av',
        'Mn',
        'No',
        'NA',
    ],
    'BsmtFin.Type.1': [
        'GLQ',
        'ALQ',
        'BLQ',
        'Rec',
        'LwQ',
        'Unf',
    ],
    'BsmtFin.Type.2': [
        'GLQ',
        'ALQ',
        'BLQ',
        'Rec',
        'LwQ',
        'Unf',
    ],
    'Heating.QC': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Electrical': [
        'SBrkr',
        'FuseA',
        'FuseF',
        'FuseP',
        'Mix',
    ],
    'Kitchen.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Functional': [
        'Typ',
        'Min1',
        'Min2',
        'Mod',
        'Maj1',
        'Maj2',
        'Sev',
        'Sal',
    ],
    'Fireplace.Qu': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Garage.Finish': [
        'Fin',
        'RFn',
        'Unf',
    ],
    'Garage.Qual': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',
    ],
    'Garage.Cond': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
        'Po',    
    ],
    'Paved.Drive': [
        'Y',
        'P',
        'N',
    ],
    'Pool.QC': [
        'Ex',
        'Gd',
        'TA',
        'Fa',
    ],
    'Fence': [
        'GdPrv',
        'MnPrv',
        'GdWo',
        'MnWw',
    ],
}


# In[21]:


for col, orderings in category_orderings.items():
    data[col] = data[col] \
        .astype('category') \
        .cat \
        .set_categories(orderings, ordered=True)


# ### Check the variable summaries using `.describe()`

# Now that our variables are painstakingly organized, with proper types, let's see a summary of each. The Pandas `DataFrame` method `.describe()` is smart: it will provide a summary that is appropriate for the type of variable if all columns are categorical or all are numerical. Use `.select_dtypes()` to select only categoricals or numericals.

# In[22]:


data \
    .select_dtypes('category') \
    .describe() \
    .transpose() \
    .sort_values(by='count', ascending=True)


# Looks like some features are very sparse, we may have to analyze their value to the project's goals later.

# In[23]:


data \
    .select_dtypes('category') \
    .describe() \
    .transpose() \
    .sort_values(by='unique', ascending=True)


# In[24]:


data \
    .select_dtypes('number') \
    .describe() \
    .transpose() \
    .sort_values(by='count', ascending=True)


# Apart from `Lot.Frontage`, all numerical features do not present many missing values, that's good!

# ### Save the data with the correct types

# Make a directory `processed` in the `data` folder of our project, save your processed dataframe there.

# In[25]:


processed_dir = DATA_DIR / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)


# Let's save the data as `pickle`, it's simpler. 
# 
# IMPORTANT NOTE: when saving data that you expect to last for many years, choose a simpler and more stable format - despite the myriad of file types, the good-ole CSV still stands.

# In[26]:


processed_file_path = processed_dir / 'ames_with_correct_types.pkl'


# In[27]:


with open(processed_file_path, 'wb') as file:
    pickle.dump(
        [
            data,
            continuous_variables,
            discrete_variables,
            ordinal_variables,
            categorical_variables,
        ],
        file,
    )


# Now we are ready to start analyzing this data!
