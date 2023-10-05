#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis of the Ames dataset

# In[1]:


import pathlib
import pickle


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', 500)


# It is a good idea to define a variable for the base data directory, and to construct the exact filenames from that variable. Another good idea is to use the `pathlib` library for manipulating paths in Python, as it will make your code work in both Windows and Linux/MacOS.

# In[2]:


DATA_DIR = pathlib.Path.cwd().parent / 'data'
print(DATA_DIR)


# ## Analyzing the columns individually

# Let's load the data from the previous section - this way we don't need to re-run all of the previous data adjusting:

# In[3]:


processed_file_path = DATA_DIR / 'processed' / 'ames_with_correct_types.pkl'

with open(processed_file_path, 'rb') as file:
    (
        data,
        continuous_variables,
        discrete_variables,
        ordinal_variables,
        categorical_variables,
    ) = pickle.load(file)


# ### A first look at the categorical variables

# In[4]:


def plot_categoricals(data, cols, sorted=True):
    summary = data[cols] \
        .describe() \
        .transpose() \
        .sort_values(by='count')

    print(summary)

    for k, (col, val) in enumerate(summary['count'].items()):
        plt.figure()
        ser = data[col].value_counts()
        if sorted:
            ser = ser.sort_values()
        else:
            ser = ser.sort_index()
        ax = ser.plot.barh()
        for container in ax.containers:
            ax.bar_label(container)
        plt.title(f'{col}, n={int(val)}')
        plt.show()


# In[5]:


plot_categoricals(data, categorical_variables)


# It is important to notice:
# 
# - There are variables that have many categories with little representation. 
#     - It may be interesting to remove the minor categories, and make a note that the model that we are developing is not suitable to process houses of these categories. 
#     - Or we may decide to ignore columns of this nature altogether.
#     - A third option is to group the minor categories into a new category named `Other`, to indicate that we are not ignoring these properties, but we don't have enough evidence to infer the effect of the precise minor categories into the sale price.
# - Some variables contain a great number of missing values. 
#     - It may be better to drop those columns,
#     - or to assign all missing values to a newly created `Unknown` category
# 
# 

# In order to simplify the data (CAREFUL: we may end up damaging the model here! If the future model is not performing well, it could be interesting to revisit these assumptions), we will process each variable, remove outliers, etc.

# #### Residential zoning, sales types and conditions (`MS.Zoning`, `Sale.Type`, and `Sale.Condition`)

# Lets concentrate first on residential sales types:

# In[6]:


data['MS.Zoning'].unique()


# In[7]:


data['MS.Zoning'].value_counts()


# We observe that a small number of sales are for non-residential properties, namely the categories `C (all)`, `A (agr)`, and `I (all)`. Let's remove them.

# In[8]:


selection = ~(data['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))
selection.value_counts()


# So far, so good.

# In[9]:


data = data[selection]


# After removing those undesired rows, lets remove the unused categories

# In[10]:


data['MS.Zoning'] = data['MS.Zoning'].cat.remove_unused_categories()


# In[11]:


data['MS.Zoning'].value_counts()


# Lets analyze the types of sale and condition. The documentation states:
# 
# ```
# Sale Type (Nominal): Type of sale
# 		
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
# 		
# Sale Condition (Nominal): Condition of sale
# 
#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)
# ```
# 
# Lets look at the representation of each category in the dataset.

# In[12]:


data['Sale.Type'].value_counts()


# In[13]:


data['Sale.Type'].unique()


# *AHA!* Careful with the name of the categories! The category `"WD "` has a space in it! Too many hours were spent debugging this kind of thing.

# In[14]:


data['Sale.Condition'].value_counts()


# Upon careful analysis, looks like all types of sales and conditions are valid. But we need to be careful with the low representativity of some categories in both `Sale.Type` and `Sale.Condition`. Let's do some category reassigning:
# 
# - All warranty deed types will go into a `GroupedWD` category;
# - Category `New` stays as-is.
# - All the remaining minor categories go into an `Other` category.

# In[15]:


processed_data = data.copy()


# In[16]:


def remap_categories(
    series: pd.Series,
    old_categories: tuple[str],
    new_category: str,
) -> pd.Series:
    # Add the new category to the list of valid categories.
    series = series.cat.add_categories(new_category)

    # Set all items of the old categories as the new category.
    remapped_items = series.isin(old_categories)
    series.loc[remapped_items] = new_category

    # Clean up the list of categories, the old categories no longer exist.
    series = series.cat.remove_unused_categories()

    return series


# In[17]:


processed_data['Sale.Type'] = remap_categories(
    series=processed_data['Sale.Type'],
    old_categories=('WD ', 'CWD', 'VWD'),
    new_category='GroupedWD',
)

processed_data['Sale.Type'] = remap_categories(
    series=processed_data['Sale.Type'],
    old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),
    new_category='Other',
)


# In[18]:


processed_data['Sale.Type'].value_counts()


# Much better!

# In[19]:


data = processed_data


# #### Street paving (`Street`)

# Now lets focus on street paving (`Street`):

# In[20]:


data['Street'].value_counts()


# The very low representativity of the minor class `Grvl` forces us to ignore this column altogether:

# In[21]:


data = data.drop(columns='Street')


# #### House surroundings (`Condition.1` and `Condition.2`)

# Let's check for conditions (`Condition.1` and `Condition.2`) pertaining to the house surroundings:

# In[22]:


data['Condition.1'].value_counts()


# In[23]:


data['Condition.2'].value_counts()


# In[24]:


pd.crosstab(data['Condition.1'], data['Condition.2'])


# Again, we observe very low representation of the minor classes. Lets reassign some categories to group similar features:
# 
# - The railroad proximity categories (`RRAn`, `RRAe`, `RRNn`, and `RRNe`) will go into a single `Railroad` category;
# - The `Feedr` and `Artery` categories refer to larger streets that collect traffic from local streets and connect neighborhoods (e.g. large avenues), lets place them into a `Roads` category;
# - The `PosA` and `PosN` refer to positive features adjacent or nearby the building, lets place them into a `Positive` category.

# In[25]:


processed_data = data.copy()


# In[26]:


for col in ('Condition.1', 'Condition.2'):
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),
        new_category='Railroad',
    )
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('Feedr', 'Artery'),
        new_category='Roads',
    )
    processed_data[col] = remap_categories(
        series=processed_data[col],
        old_categories=('PosA', 'PosN'),
        new_category='Positive',
    )


# In[27]:


processed_data['Condition.1'].value_counts()


# In[28]:


processed_data['Condition.2'].value_counts()


# In[29]:


pd.crosstab(processed_data['Condition.1'], processed_data['Condition.2'])


# Looks like we can recombine the `Condition.1` and `Condition.2` columns into a single categorical column with the categories:
# 
# - `Norm`: `Condition.1` is `Norm`;
# - `Railroad`: `Condition.1` is `Railroad` and `Condition.2` is `Norm`;
# - `Roads`: `Condition.1` is `Roads` and `Condition.2` is not `Railroad`;
# - `Positive`: `Condition.1` is `Positive`;
# - `RoadsAndRailroad`: (`Condition.1` is `Railroad` and `Condition.2` is `Roads`) or (`Condition.1` is `Roads` and `Condition.2` is `Railroad`).

# In[30]:


processed_data['Condition'] = pd.Series(
    index=processed_data.index,
    dtype=pd.CategoricalDtype(categories=(
        'Norm',
        'Railroad',
        'Roads',
        'Positive',
        'RoadsAndRailroad',
    )),
)


# In[31]:


norm_items = processed_data['Condition.1'] == 'Norm'
processed_data['Condition'][norm_items] = 'Norm'


# In[32]:


railroad_items = \
    (processed_data['Condition.1'] == 'Railroad') \
    & (processed_data['Condition.2'] == 'Norm')
processed_data['Condition'][railroad_items] = 'Railroad'


# In[33]:


roads_items = \
    (processed_data['Condition.1'] == 'Roads') \
    & (processed_data['Condition.2'] != 'Railroad')
processed_data['Condition'][roads_items] = 'Roads'


# In[34]:


positive_items = processed_data['Condition.1'] == 'Positive'
processed_data['Condition'][positive_items] = 'Positive'


# In[35]:


roads_and_railroad_items = \
    ( \
        (processed_data['Condition.1'] == 'Railroad') \
        & (processed_data['Condition.2'] == 'Roads')
    ) \
    | ( \
        (processed_data['Condition.1'] == 'Roads') \
        & (processed_data['Condition.2'] == 'Railroad') \
    )
processed_data['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'


# In[36]:


processed_data['Condition'].value_counts()


# There is no apparent need, for now, to go into further simplification of this variable. Lets drop the original variables:

# In[37]:


processed_data = processed_data.drop(columns=['Condition.1', 'Condition.2'])


# In[38]:


data = processed_data


# #### Columns with many missing values (`Misc.Feature` and `Alley`)

# The columns `Misc.Feature` and `Alley` are mostly formed by missing values!

# In[39]:


plot_categoricals(data, ['Misc.Feature', 'Alley'])


# But it looks like we can reuse this information, still. We can transform the `Misc.Feature` variable into a `HasShed` variable that indicates whether the house has a shed:

# In[40]:


data['HasShed'] = data['Misc.Feature'] == 'Shed'
data = data.drop(columns='Misc.Feature')


# In[41]:


data['HasShed'].value_counts()


# Likewise, we can mutate the `Alley` feature into a `HasAlley` feature:

# In[42]:


data['HasAlley'] = ~data['Alley'].isna()
data = data.drop(columns='Alley')


# In[43]:


data['HasAlley'].value_counts()


# #### Exterior coverings (`Exterior.1st` and `Exterior.2nd`)

# The exterior coverings have a lot of categories, some with very low representativity:

# In[44]:


plot_categoricals(data, ['Exterior.1st', 'Exterior.2nd'])


# Also, it looks like there are a few typos!

# 
# | `Exterior.1st` | `Exterior.2nd` | `Correct value` |
# |----------------|----------------|-----------------|
# | `BrkComm`      | `Brk Cmn`      | `BrkComm`       |
# | `CemntBd`      | `CmentBd`      | `CemntBd`       |
# | `WdShing`      | `Wd Shng`      | `WdShing`       |

# Lets fix those

# In[45]:


data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('Brk Cmn', ),
    new_category='BrkComm',
)
data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('CmentBd', ),
    new_category='CemntBd',
)
data['Exterior.2nd'] = remap_categories(
    series=data['Exterior.2nd'],
    old_categories=('Wd Shng', ),
    new_category='WdShing',
)


# In[46]:


for col in ('Exterior.1st', 'Exterior.2nd'):
    categories = data[col].cat.categories
    data[col] = data[col].cat.reorder_categories(sorted(categories))


# In[47]:


pd.crosstab(data['Exterior.1st'], data['Exterior.2nd'])


# It looks like there are a few popular options and lots of poorly represented materials beyond the popular ones. Due to lack of representativity, lets keep only the popular categories as-is, and group the rest into an `Other` category.
# 
# Also, looks like it is often the case that the first material is the same as the second material, probably to indicate that the house exterior contains only one material. Therefore, we will keep only the `Exterior.1st` variable.

# In[48]:


processed_data = data.copy()


# In[49]:


mat_count = processed_data['Exterior.1st'].value_counts()
mat_count


# In[50]:


rare_materials = list(mat_count[mat_count < 40].index)
rare_materials


# In[51]:


processed_data['Exterior'] = remap_categories(
    series=processed_data['Exterior.1st'],
    old_categories=rare_materials,
    new_category='Other',
)
processed_data = processed_data.drop(columns=['Exterior.1st', 'Exterior.2nd'])


# In[52]:


processed_data['Exterior'].value_counts()


# In[53]:


data = processed_data


# #### `Heating`

# In[54]:


plot_categoricals(data, ['Heating',])


# This column does not have missing values, and an overwhelming amount of items belong to the same category. As such, there is not much information here, lets discard the column.

# In[55]:


data = data.drop(columns='Heating')


# #### `Roof.Matl` and `Roof.Style`

# In[56]:


plot_categoricals(data, ['Roof.Matl', 'Roof.Style'])


# Due to the low representativity of the minor categories in the `Roof.Matl` feature, we will drop it.

# In[57]:


data = data.drop(columns='Roof.Matl')


# In the `Roof.Style` feature we have two substantial categories, and a few very minor ones. Lets group the minor categories into an `Other` category:

# In[58]:


data['Roof.Style'] = remap_categories(
    series=data['Roof.Style'],
    old_categories=[
        'Flat',
        'Gambrel',
        'Mansard',
        'Shed',
    ],
    new_category='Other',
)


# In[59]:


data['Roof.Style'].value_counts()


# #### `Mas.Vnr.Type`

# In[60]:


data['Mas.Vnr.Type'].info()


# In[61]:


data['Mas.Vnr.Type'].value_counts()


# Lets group the two minor classes into an `Other` class:

# In[62]:


data['Mas.Vnr.Type'] = remap_categories(
    series=data['Mas.Vnr.Type'],
    old_categories=[
        'BrkCmn',
        'CBlock',
    ],
    new_category='Other',
)


# Also, lets add the missing entries to the `None` category, since we have no evidence of the veneer type here.

# In[63]:


data['Mas.Vnr.Type'] = data['Mas.Vnr.Type'].cat.add_categories('None')
data['Mas.Vnr.Type'][data['Mas.Vnr.Type'].isna()] = 'None'


# In[64]:


data['Mas.Vnr.Type'].value_counts()


# #### `MS.SubClass`

# In[65]:


plot_categoricals(data, ['MS.SubClass'])


# This is a complicated feature, lets look at the documentation:
# 
# ```
# MS SubClass (Nominal): Identifies the type of dwelling involved in the sale.	
# 
#        020	1-STORY 1946 & NEWER ALL STYLES
#        030	1-STORY 1945 & OLDER
#        040	1-STORY W/FINISHED ATTIC ALL AGES
#        045	1-1/2 STORY - UNFINISHED ALL AGES
#        050	1-1/2 STORY FINISHED ALL AGES
#        060	2-STORY 1946 & NEWER
#        070	2-STORY 1945 & OLDER
#        075	2-1/2 STORY ALL AGES
#        080	SPLIT OR MULTI-LEVEL
#        085	SPLIT FOYER
#        090	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
# ```

# This is the moment where you reach out to a real-estate agent and ask what is the meaning of these categories, and whether they can they be meaningfully grouped together. For now lets just reassign the minor categories to an `Other` category:

# In[66]:


data['MS.SubClass'] = remap_categories(
    series=data['MS.SubClass'],
    old_categories=[75, 45, 180, 40, 150],
    new_category='Other',
)


# In[67]:


data['MS.SubClass'].value_counts()


# #### `Foundation`

# In[68]:


plot_categoricals(data, ['Foundation'])


# Same story: minor categories grouped into an `Other` category.

# In[69]:


data['Foundation'] = remap_categories(
    series=data['Foundation'],
    old_categories=['Slab', 'Stone', 'Wood'],
    new_category='Other',
)


# #### `Neighborhood`

# In[70]:


data['Neighborhood'].value_counts()


# We can either group the minor categories into an `Other` category, or drop the rows. Both approaches have their pros and cons:
# 
# - If we delete the rows we are being more precise, in the sense that we restrict our model to fewer neighborhoods.
# - If we mantain the rows we have more data to construct the model.
# 
# Lets drop the rows and make a note that this model does not work for those neighborhoods.

# In[71]:


selection = ~data['Neighborhood'].isin([
    'Blueste',
    'Greens',
    'GrnHill',
    'Landmrk',
])
data = data[selection]


# In[72]:


data['Neighborhood'] = data['Neighborhood'].cat.remove_unused_categories()


# In[73]:


data['Neighborhood'].value_counts()


# #### `Garage.Type`

# In[74]:


data['Garage.Type'].info()


# In[75]:


data['Garage.Type'].value_counts()


# Looks like there are a few residences that do not have a garage, lets create a `NoGarage` category for them.

# In[76]:


data['Garage.Type'] = data['Garage.Type'].cat.add_categories(['NoGarage'])
data['Garage.Type'][data['Garage.Type'].isna()] = 'NoGarage'


# In[77]:


data['Garage.Type'].value_counts()


# ### A final look at the categorical variables

# After all of this processing, even the set of categorical variables has changed, so lets make a new list:

# In[78]:


all_categorical = data.select_dtypes('category').columns

new_categorical_variables = [ \
    col for col in all_categorical \
    if not col in ordinal_variables \
]


# In[79]:


plot_categoricals(data, new_categorical_variables)


# Looks like we are done with the categorical variables, yay! Notice that there are no more missing values among the categorical variables!

# ### Analyzing the ordinal variables

# Lets take a first look:

# In[80]:


plot_categoricals(data, ordinal_variables, sorted=False)


# Ouch! Again, we suffer from the existence of categories with very low representativity! But here, different from the nominal variables, we have a choice. If we model the ordinal variables as increaasing numbers (we'll do that when preparing the data for the model), than the low representativity of some categories is no longer a problem: it is just a number that does not occur often!
# 
# So the only problems to address here are:
# 
# - Extreme cases of low representativity: the `Utilities` feature.
# - Large number of missing values.

# #### `Utilities`

# This one is easy: the low representativity here is so extreme that we will just drop this column.

# In[81]:


data = data.drop(columns='Utilities')


# #### Large number of missing values

# There are aa few different cases here:

# 
# ##### `Pool.QC`
# 
# Negligible information here, drop the column.

# In[82]:


data = data.drop(columns='Pool.QC')


# ##### `Fence`
# 
# This is interesting. The documentation says:
# 
# ```
# Fence (Ordinal): Fence quality
# 		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
# ```
# 
# Since the `Fence` feature means to convey the level of privacy that a fence brings to the residence, we can create a new category `NoFence` and set it to be the category with the lowest privacy level! Then we mark all missing values as `NoFence`!

# In[83]:


data['Fence'].value_counts().sort_index()


# In[84]:


old_categories = list(data['Fence'].cat.categories)
old_categories


# In[85]:


new_categories = old_categories + ['NoFence']
new_categories


# In[86]:


data['Fence'] = data['Fence'].cat.set_categories(new_categories)


# In[87]:


data['Fence'].dtype


# In[88]:


data['Fence'][data['Fence'].isna()] = 'NoFence'


# In[89]:


data['Fence'].value_counts().sort_index()


# ##### `Fireplace.Qu`
# 
# This is a hard one. There is a lot of information there, but also a lot of missing values with no obvious way to transform them into something meaningful (like in the `Fence` case).

# In[90]:


data['Fireplace.Qu'].value_counts().sort_index()


# Observe that most fireplace quality indicators are between "good" and "typical". Also notice that there is a variable `Fireplaces` that list how many fireplaces a house has.

# In[91]:


data['Fireplaces'].value_counts()


# These observations give us an alternative here: drop the `Fireplace.Qu` column. Reasons:
# 
# - If there is a fireplace, it will usually be "good" or "typical" - there is a fair amount of lower and higher level fireplaces, but we have to choose to ignore something;
# - The `Fireplaces` variable already conveys the idea of whether there is a fireplace (or many).

# In[92]:


data = data.drop(columns='Fireplace.Qu')


# ##### `Garage.Cond`, `Garage.Qual`, `Garage.Finish`
# 
# All of these variables (plus the `Garage.Yr.Blt`) have the same number of non-missing entries:

# In[93]:


plot_categoricals(
    data,
    [
        'Garage.Cond',
        'Garage.Qual',
        'Garage.Finish',
    ],
    sorted=False,
)


# While the `Garage.Cond` and `Garage.Qual` features can be ignored, the `Garage.Finish` feature seems to have relevant information. What to do?

# In[94]:


data = data.drop(columns=['Garage.Cond', 'Garage.Qual'])


# 
# As always, some information will be discarded in one way or another. In this case we can transform the `Garage.Finish` variable from an ordinal variable to a nominal variable - that is, we discard the ordering of the finishing levels. This way we can create a new category `NoGarage` to account for the missing values.

# In[95]:


data['Garage.Finish'] = data['Garage.Finish'] \
    .cat \
    .as_unordered() \
    .cat \
    .add_categories(['NoGarage'])
data['Garage.Finish'][data['Garage.Finish'].isna()] = 'NoGarage'


# In[96]:


data['Garage.Finish'].value_counts()


# In[97]:


data['Garage.Finish'].dtype


# In[98]:


data['Garage.Finish'].cat.ordered


# ##### `Electrical`

# In[99]:


data['Electrical'].isna().value_counts()


# In[100]:


plot_categoricals(data, ['Electrical'], sorted=False)


# So we can drop the only row with a missing value for the `Electrical` variable, or we can fill it with something. Given the high prevalence of the `SBrkr` category, lets set the missing value as that.

# In[101]:


data['Electrical'][data['Electrical'].isna()] = 'SBrkr'


# In[102]:


ordinal_columns = [col for col in data.select_dtypes('category') if data[col].cat.ordered]


# In[103]:


data[ordinal_columns].info()


# ##### `Bsmt.Qual`, `Bsmt.Cond`, `Bsmt.Exposure`, `BsmtFin.Type.1`, `BsmtFin.Type.2`

# In[104]:


plot_categoricals(
    data,
    [
        'Bsmt.Qual',
        'Bsmt.Cond',
        'Bsmt.Exposure',
        'BsmtFin.Type.1',
        'BsmtFin.Type.2',
    ],
    sorted=False,
)


# For `Bsmt.Exposure`, lets assign the missing entries to the `NA` category. For all the other columns (`Bsmt.Qual`, `Bsmt.Cond`, `BsmtFin.Type.1`, `BsmtFin.Type.2`) lets also create the `NA` category and assign the missing entries to that category. Finally, since the `NA` category does not fit into a sequence with the other classes, lets convert these columns to nominal (that is, categoricals without category order). Notice that in this case we need to eliminate the unused categories.

# In[105]:


data['Bsmt.Exposure'].unique()


# In[106]:


data['Bsmt.Exposure'][data['Bsmt.Exposure'].isna()] = 'NA'
data['Bsmt.Exposure'] = data['Bsmt.Exposure'] \
    .cat \
    .as_unordered() \
    .cat \
    .remove_unused_categories()


# In[107]:


for col in ('Bsmt.Qual', 'Bsmt.Cond', 'BsmtFin.Type.1', 'BsmtFin.Type.2'):
    data[col] = data[col].cat.add_categories(['NA'])
    data[col][data[col].isna()] = 'NA'
    data[col] = data[col] \
        .cat \
        .as_unordered() \
        .cat \
        .remove_unused_categories()


# In[108]:


plot_categoricals(
    data,
    [
        'Bsmt.Qual',
        'Bsmt.Cond',
        'Bsmt.Exposure',
        'BsmtFin.Type.1',
        'BsmtFin.Type.2',
    ],
    sorted=False,
)


# Lets also place the entries from the minor categories of `Bsmt.Cond` into neighboring categories, to simplify the dataset.
# 

# In[109]:


data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Po'] = 'Fa'
data['Bsmt.Cond'][data['Bsmt.Cond'] == 'Ex'] = 'Gd'
data['Bsmt.Cond'] = data['Bsmt.Cond'].cat.remove_unused_categories()


# In[110]:


data['Bsmt.Cond'].value_counts()


# In[111]:


data[ordinal_columns].info()


# Great! All ordinal variables are corrected!

# ### Analyzing the continuous variables

# Lets look for missing values, anomalies, outliers, and all sorts of things that may hinder our modeling.

# In[112]:


def plot_numericals(data, cols):
    summary = data[cols] \
        .describe() \
        .transpose() \
        .sort_values(by='count')

    print(summary)

    n = data.shape[0]
    b = int(np.sqrt(n))
    for k, (col, val) in enumerate(summary['count'].items()):
        plt.figure()
        data[col].plot.hist(bins=b)
        plt.title(f'{col}, n={int(val)}')
        plt.show()

plot_numericals(data, data.select_dtypes('number').columns)


# #### `SalePrice`

# Ah, the target! The most special feature of all! Sale prices have a few special properties:
# 
# - They are always non-negative, and most likely have a min value that is positive.
# - The *importance of the difference* of sale prices is not absolute: a difference of $10k in a house valued at $100k is a lot (10%), but in a house valued at $1M is negligible.
# 
# These characteristics suggest something important: maybe our model should predict the *logarithm* of the sale price, rather than the actual price! This way, errors are associated with proportions, rather than absolute values! 
# 
# Lets analyze the previous sentence mathematically. Call $P$ the real value of a house, and $\hat{P}$ the value given by our (yet to be developed) model. However, since we will work with the logarithm of $P$ and $\hat{P}$ instead, lets name the log-values as well: $y = \log_{10}{(P)}$ and $\hat{y} = \log_{10}{(\hat{P})}$.
# 
# (Why base $10$? Mere convenience: the value of $y$ or $\hat{y}$ can be immediately associated with the base-$10$ magnitude of the house price they represent. For instance: $y = 5.6$ implies $P = 10^{5.6} = 10^{0.6} \times 10^{5}$, which is something in the range of hundreds of thousands. So, take the integer part of the base-$10$ logarithm and you automatically have the order of magnitude of the price).
# 
# Now, when we have a prediction value $\hat{y}$ for a house with log-price $y$, the error is $\varepsilon = (\hat{y} - y)$. Substituting the definitions of $y$ and $\hat{y}$ we have:
# 
# $$
# \varepsilon = \hat{y} - y = \log_{10}{(\hat{P})} - \log_{10}{(P)} = \log_{10}{\left(\frac{\hat{P}}{P}\right)}
# $$
# 
# Therefore we can obtain the prediction error for this house as a fraction:
# 
# $$
# 10^{\varepsilon} = 10^{\log_{10}{\left(\frac{\hat{P}}{P}\right)}} = \frac{\hat{P}}{P}
# $$
# 
# And we can convert it into a percentage:
# 
# $$
# \varepsilon_{\%} = 100 * (10^{\varepsilon} - 1) = 100 * \left(\frac{\hat{P}}{P} - 1\right) = 100 * \left(\frac{\hat{P} - P}{P}\right)
# $$
# 
# Percentual errors are much more informative of the practical value of our (future) model. If I tell you that the average error of a model is 50k USD, is this good or bad? It depends on the price of the house we are considering: an error of 50k USD in a house valued at 250k USD is very significant, whereas if the house costed 10 million USD, then an error of 50k USD is quite acceptable. Now, if I tell you that the model has an average error of $5\%$, this is understandable across all ranges of house prices.
# 
# 
# Lets work this out:

# In[113]:


data['SalePrice'].describe()


# Lets remap the `SalePrice` variable to represent the base-10 logarithm of the original value. Again, why base 10? Any base will do, but the base 10 is a bit more interpretable: a value of 3 means thousands, 4 means tens of thousands, 6 means million, etc.

# In[114]:


data['SalePrice'] = data['SalePrice'].apply(np.log10)


# In[115]:


data['SalePrice'].describe()


# #### `Lot.Frontage`

# In[116]:


data['Lot.Frontage'].info()


# Plenty of missing values here, what do they mean? The documentation is not very helpful:
# 
# ```
# Lot Frontage (Continuous): Linear feet of street connected to property
# ```
# 
# Maybe the relation between this feature and some other feature will explain it.

# In[117]:


missing_lot_frontage = data['Lot.Frontage'].isna()


# In[118]:


data['MS.SubClass'][missing_lot_frontage].value_counts()


# In[119]:


data['Lot.Config'][missing_lot_frontage].value_counts()


# In[120]:


data['Land.Contour'][missing_lot_frontage].value_counts()


# Nope, no obvious relationship. Is this column even useful for prediction? Lets compare it with `SalePrice`:

# In[121]:


data.plot.scatter(x='Lot.Frontage', y='SalePrice', alpha=0.1)
plt.ylabel('$\log_{10} SalePrice$')


# Yikes, no way to ignore this feature! Can we at least relate it to `Lot.Area`?

# In[122]:


data[['Lot.Frontage', 'Lot.Area']].corr()


# There is some correlation there, maybe the square-root of the lot area shows better correlation?

# In[123]:


aux_data = data[['Lot.Frontage', 'Lot.Area']].copy()
aux_data['Sqrt.Lot.Area'] = aux_data['Lot.Area'].apply(np.sqrt)


# In[124]:


aux_data[['Lot.Frontage', 'Sqrt.Lot.Area']].corr()


# Nice! There is a healthy relationship between `Lot.Frontage` and the square root of `Lot.Area`!

# In[125]:


x = np.sqrt(data['Lot.Area'])
y = data['Lot.Frontage']
plt.scatter(x, y, alpha=0.2)
plt.xlim([0, 200])
plt.ylim([0, 200])
plt.xlabel('$\sqrt{Lot.Area}$')
plt.ylabel('$Lot.Frontage$')
plt.title('Relationship between lot frontage and area')
plt.show()


# We could *predict* the `Lot.Frontage` for the mising entries from the `Lot.Area` feature!
# 
# The task of filling up missing values is called **Imputation**. There are several imputation strategies, some of them are:
# 
# - Replace with some meaningful constant independent of the dataset - e.g. zero.
# - Replace with a constant derived from the dataset, like the mean or median.
# - Replace with a predicted value from other variables.
# 
# So, fitting a linear model to predict `Lot.Frontage` from the square root of `Lot.Area` could be a good alternative. But for now lets do the simplest thing: impute the missing values of `Lot.Frontage` simply with the median of `Lot.Frontage`. Once we are more familiar with modelling we can revisit this decision.
# 
# **Caution**: There is also the possibility that the missing values represent residences that *do not have a frontage*! No way to tell from the documentation or the relationship with other relevant variables! Another moment here where the really best course of action is to go back to the expert and ask questions, ok?

# In[126]:


data['Lot.Frontage'] = data['Lot.Frontage'].fillna(data['Lot.Frontage'].median())


# In[127]:


data['Lot.Frontage'].info()


# #### `Garage.Yr.Blt` 

# In[128]:


data['Garage.Yr.Blt'].describe()


# Oops! Looks like someone built a garage in the future!
# 
# The documentation states that the residential sales were collected from 2006 to 2010. We have the `Yr.Sold` variable to let us know when the house was sold. So we can obtain from those the information of how old the garage was when the house was sold! That is a better variable than `Garage.Yr.Blt` by itself.

# In[129]:


garage_age = data['Yr.Sold'] - data['Garage.Yr.Blt']
garage_age.describe()


# Lets look at the garages "from the future"

# In[130]:


data[garage_age < 0.0].transpose()


# Only two rows. One is an obvious mistake, the other looks somewhat legitimate: house was sold new in 2007, garage was only finished in 2008. We can safely correct these errors, but if there was any doubt about the validity of fixing them, it is better to get rid of the rows. In our case we will set the garage age to zero.

# In[131]:


garage_age[garage_age < 0.0] = 0.0


# And now we remove the `Garage.Yr.Blt` column, replacing it with a new `Garage.Age` column.

# In[132]:


data = data.drop(columns='Garage.Yr.Blt')
data['Garage.Age'] = garage_age


# What about the missing values of the new `Garage.Age` column?

# In[133]:


data['Garage.Age'].info()


# In[134]:


data['Garage.Type'][data['Garage.Age'].isna()].value_counts()


# Seems like they correspond to the absence of garage. What is a reasonable value for imputation of `Garage.Age` when there is no garage? Zero? Median? 
# 
# Or drop the rows altogether? (Seems excessive!)
# 
# Lets just impute the median here. The logic is that we want to cause the least harm to the data by doing this.

# In[135]:


data['Garage.Age'] = data['Garage.Age'].fillna(data['Garage.Age'].median())


# #### `Year.Remod.Add`, `Year.Built`
# 
# Lets apply the same age treatment to these variables

# In[136]:


data[['Year.Remod.Add', 'Year.Built', 'Yr.Sold']].describe()


# In[137]:


remod_age = data['Yr.Sold'] - data['Year.Remod.Add']
remod_age.describe()


# Oops, another weirdness, lets check

# In[138]:


data[remod_age < 0.0].transpose()


# Same story, lets set the remodeling age to zero for these cases

# In[139]:


remod_age[remod_age < 0.0] = 0.0


# In[140]:


house_age = data['Yr.Sold'] - data['Year.Built']
house_age.describe()


# Again...

# In[141]:


data[house_age < 0.0].transpose()


# Same approach, lets set the age to zero in this case.

# In[142]:


house_age[house_age < 0.0] = 0.0


# In[143]:


data = data.drop(columns=['Year.Remod.Add', 'Year.Built'])
data['Remod.Age'] = remod_age
data['House.Age'] = house_age


# #### `Mas.Vnr.Area`

# In[144]:


data['Mas.Vnr.Area'].info()


# Just a few missing values here. Do they correspond to the absence of veneer?

# In[145]:


data['Mas.Vnr.Type'][data['Mas.Vnr.Area'].isna()].value_counts()


# Yep. Impute with zeros.

# In[146]:


data.loc[data['Mas.Vnr.Area'].isna(), 'Mas.Vnr.Area'] = 0.0


# #### Features that contain a lot of zeros

# In the case of features that contain a lot of zeros, it could be that the feature contains actual information in the non-zero cases, and zero to indicate that the feature is absent. Consider, for instance, the `Pool.Area` feature:

# In[147]:


num_houses = data.shape[0]
num_houses_with_pool = data[data['Pool.Area'] > 0].shape[0]
print(f'Out of {num_houses} houses, only {num_houses_with_pool} have a pool.')


# What to do? Discard the column? Discard the rows (we don't appraise houses with pools!)? Transform the column into a categorical `HasPool`?
# 
# Actually, it depends on the model. A linear model, for instance, can safely process this column: since the pool area for houses without pools is zero, the contribution of the pool area to the house price will be zero (remember that a linear model associates a multiplicative coefficient for each variable).
# 
# So we leave them all as is! We can revisit our decisions once we play a bit with the predictive model *as long as we only use the training dataset* - more on this later.

# #### Final cleanup

# In[148]:


data.info()


# After all this effort, only a few columns have just one missing value! We suffered enough, lets just drop those rows.

# In[149]:


data = data.dropna(axis=0)


# In[150]:


data.info()


# Nice, not a single missing value! Let us also remove unused categories, for good measure.

# In[151]:


for col in data.select_dtypes('category').columns:
    data[col] = data[col].cat.remove_unused_categories()


# ## Joint feature analysis

# There are two types of joint feature analysis:
# 
# - feature versus feature
# - feature versus target
# 
# Lets start with the "feature versus target" analysis.

# ### Feature versus target

# In this scenario we are exploring the relationship between an individual feature and the target. Since we are involving the target, it is best to do this after splitting the dataset into training and testing subsets, to avoid inferring deep relations between the features and the target without an "out-of-sample" dataset to use to measure the generalization capacity of our (soon-to-be-developed) model. 
# 
# Why? Because this involuntary "peeking into the test dataset" may cause us to make design decisions for our model that are way too adapted to the entire dataset, leaving no way to investigate if our decisions truly lead us to a model that makes effective predictions for unseen data, or if it is only "memorizing" the entire dataset. Looking into test data before all model-design decisions are made is called **data snooping**.
# 
# So the only thing we will look at prior to train-test splitting is a simple numerical and visual correlation between the features and the target.
# 
# Here are the scatter plots between the numerical features and the target. Look for perfect correlations: they may indicate that one of the features is the target in disguise!

# In[152]:


numerical_data = data.select_dtypes('number').drop(columns='SalePrice').copy()
target = data['SalePrice'].copy()


# In[153]:


numerical_data.corrwith(target).sort_values()


# Unsurprisingly:
# 
# - The older the house, the lower the price
# - The larger the house, the higher the price
# 
# Surprisingly, the `Lot.Area` has low correlation with the target!

# In[154]:


for column, series in numerical_data.items():
    plt.figure()
    plt.scatter(series, target, alpha=0.3)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('SalePrice')
    plt.show()


# Looks like no feature is perfectly predicting the target, which could indicate an obvious error in understanding the dataset.

# For the categorical features we will use box plots:

# In[155]:


categorical_columns = data.select_dtypes('category').columns


# In[156]:


for column in categorical_columns:
    aux_dataframe = data[[column, 'SalePrice']]
    aux_dataframe.plot.box(by=column)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('SalePrice')
    plt.show()


# Looks like there is no suspicious relationship between the target and the features, moving on:

# ### Feature versus feature

# This topic is a bit more complicated. First of all, there is a quadratic number of relationships for feature-versus-feature comparisons here, so it is not feasible to really analyze them all. We will limit ourselves to the analysis of correlations between numerical variables for now:

# In[157]:


corr = data.corr(numeric_only=True)
corr


# In[158]:


sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


# Why look for high correlations? In some models, the presence of high correlations may prevent the good fitting of the data onto the model. This is the case for the simple linear model. For now it seems that there are no damaging correlations here.

# ## Save the cleaned data

# In[159]:


clean_data_path = DATA_DIR / 'processed' / 'ames_clean.pkl'


# In[160]:


with open(clean_data_path, 'wb') as file:
    pickle.dump(data, file)

