import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from sklearn.preprocessing import RobustScaler
from category_encoders import OrdinalEncoder

def remap_objects(series, old_categories, new_category):
    if not pd.api.types.is_categorical_dtype(series):
        series = series.astype('category')
    series = series.cat.add_categories(new_category)
    remapped_items = series.isin(old_categories)
    series.loc[remapped_items] = new_category
    series = series.cat.remove_unused_categories()
    return series

def sc_preprocessing(item: dict):
    try:
        df = pd.DataFrame(item, index=[0])

        df.drop(columns=['Order', 'PID'], inplace=True)
        
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

        fill_none = [
            'Pool.QC',
            'Misc.Feature',
            'Alley',
            'Fence',
            'Fireplace.Qu',
            'Garage.Finish',
            'Garage.Qual',
            'Garage.Cond',
            'Garage.Finish',
            'Garage.Qual',
            'Garage.Cond',
            'Garage.Type'
        ]

        fill_no_bsmt = [
            'Bsmt.Qual',
            'Bsmt.Cond',
            'Bsmt.Exposure',
            'BsmtFin.Type.1',
            'BsmtFin.Type.2',
        ]

        fill_mode = [
            'Electrical',
            'Functional',
            'Kitchen.Qual',
            'Exterior.1st',
            'Exterior.2nd',
            'MS.Zoning',
            'Sale.Type',
            'Mas.Vnr.Type'
        ]

        for col in fill_none:
            df[col].fillna('None', inplace=True)

        for col in fill_no_bsmt:
            df[col].fillna('NoBsmt', inplace=True)

        # for col in fill_mode:
        #     df[col].fillna(df[col].mode()[0], inplace=True)

        df_nan = df.isna().sum().to_frame('nan_count')

        df_nan['type'] = df.dtypes

        df_nan.sort_values(by='nan_count', ascending=False, inplace=True)
        cols_nan = df_nan[df_nan['nan_count'] != 0].index.tolist()
        cols_nonan = df_nan[df_nan['nan_count'] == 0].index.tolist()

        df_num = df.select_dtypes(include=np.number)
        df_cat = df.select_dtypes(exclude=np.number)

        # for col in cols_nan:
        #     impute = df_num[df_num[col].isna()]
        #     imputer_train = df_num[~df_num[col].isna()]
            
        #     # Certifique-se de que cols_nonan esteja presente em ambos os DataFrames
        #     cols_nonan = [col for col in cols_nonan if col in df_num.columns]
            
        #     imputer = KNeighborsRegressor(n_neighbors=5)
        #     knr = imputer.fit(imputer_train[cols_nonan], imputer_train[col])
        #     df_num.loc[impute.index, col] = knr.predict(impute[cols_nonan])

        df = pd.concat([df_num, df_cat], axis=1)

        selection = ~(df['MS.Zoning'].isin(['A (agr)', 'C (all)', 'I (all)']))
        df = df[selection]

        df.drop(columns=['Garage.Cars'], inplace=True)

        df['Sale.Type'].replace({'WD ':'WD'}, inplace=True)

        df['Sale.Type'] = remap_objects(
            series=df['Sale.Type'],
            old_categories=('WD ', 'CWD', 'VWD'),
            new_category='GroupedWD',
        )

        df['Sale.Type'] = remap_objects(
            series=df['Sale.Type'],
            old_categories=('COD', 'ConLI', 'Con', 'ConLD', 'Oth', 'ConLw'),
            new_category='Other',
        )

        df.drop(columns=['Street'], inplace=True)

        for col in ('Condition.1', 'Condition.2'):
            df[col] = remap_objects(
                series=df[col],
                old_categories=('RRAn', 'RRAe', 'RRNn', 'RRNe'),
                new_category='Railroad',
            )
            df[col] = remap_objects(
                series=df[col],
                old_categories=('Feedr', 'Artery'),
                new_category='Roads',
            )
            df[col] = remap_objects(
                series=df[col],
                old_categories=('PosA', 'PosN'),
                new_category='Positive',
            )
        
        df['Condition.1'] = df['Condition.1'].astype('category')
        df['Condition.2'] = df['Condition.2'].astype('category')

        df['Condition'] = pd.Series(
            index=df.index,
            dtype=pd.CategoricalDtype(categories=(
                'Norm',
                'Railroad',
                'Roads',
                'Positive',
                'RoadsAndRailroad',
            )),
        )

        norm_items = df['Condition.1'] == 'Norm'
        df['Condition'][norm_items] = 'Norm'

        railroad_items = \
            (df['Condition.1'] == 'Railroad') \
            & (df['Condition.2'] == 'Norm')
        df['Condition'][railroad_items] = 'Railroad'

        roads_items = \
            (df['Condition.1'] == 'Roads') \
            & (df['Condition.2'] != 'Railroad')
        df['Condition'][roads_items] = 'Roads'

        positive_items = df['Condition.1'] == 'Positive'
        df['Condition'][positive_items] = 'Positive'

        roads_and_railroad_items = \
            ( \
                (df['Condition.1'] == 'Railroad') \
                & (df['Condition.2'] == 'Roads')
            ) \
            | ( \
                (df['Condition.1'] == 'Roads') \
                & (df['Condition.2'] == 'Railroad') \
            )
        df['Condition'][roads_and_railroad_items] = 'RoadsAndRailroad'

        df.drop(columns=['Condition.1', 'Condition.2'], inplace=True)

        df['HasShed'] = df['Misc.Feature'] == 'Shed'
        df.drop(columns=['Misc.Feature'], inplace=True)

        mat_count = df['Exterior.1st'].value_counts()

        df['Exterior'] = remap_objects(
            series=df['Exterior.1st'],
            old_categories=mat_count[mat_count < 40].index.tolist(),
            new_category='Other',
        )

        df.drop(columns=['Exterior.1st', 'Exterior.2nd'], inplace=True)

        df.drop(columns=['Heating'], inplace=True)

        df.drop(columns=['Roof.Matl'], inplace=True)

        df['Roof.Style'] = remap_objects(
            series=df['Roof.Style'],
            old_categories=('Flat', 'Gambrel', 'Mansard', 'Shed'),
            new_category='Other',
        )

        df['Mas.Vnr.Type'] = remap_objects(
            series=df['Mas.Vnr.Type'],
            old_categories=[
                'BrkCmn',
                'CBlock',
            ],
            new_category='Other',
        )

        df['Mas.Vnr.Type'] = df['Mas.Vnr.Type'].cat.add_categories('None')
        df['Mas.Vnr.Type'].fillna('None', inplace=True)

        df.drop(columns=['Utilities', 'Pool.QC', 'Fireplace.Qu', 'Garage.Cond', 'Garage.Qual'], inplace=True)

        df['Garage.Finish'] = df['Garage.Finish'] \
            .astype('category') \
            .cat \
            .as_unordered() \
            .cat \
            .add_categories(['NoGarage'])
        
        df['Garage.Finish'][df['Garage.Finish'].isna()] = 'NoGarage'
        df['Sq.Ft.PerRoom'] = df['Gr.Liv.Area'] / (df['TotRms.AbvGrd'] + df['Full.Bath'] + df['Half.Bath'] + df['Kitchen.AbvGr'])
        df['Ttl.Home.Qual'] = np.sqrt((df['Overall.Qual'] ** 2 + df['Overall.Cond'] ** 2) / 2)
        df['Ttl.Bath'] = df['Full.Bath'] + 0.5 * df['Half.Bath'] + df['Bsmt.Full.Bath'] + 0.5 * df['Bsmt.Half.Bath']
        df['Age'] = df['Yr.Sold'] - df['Year.Built']
        df['Age.Remod'] = df['Yr.Sold'] - df['Year.Remod.Add']

        # removendo a assimetria e a curtose com teste jarque-bera
        num_fts = df.select_dtypes(include=np.number).columns.tolist()
        non_normal = list()

        # realizar teste jarque-bera para cada feature numerica
        for ft in num_fts:
            _, p_value = stats.jarque_bera(df[ft])
            if p_value < 0.05:
                non_normal.append(ft)

        # transformar features nao normais
        for ft in non_normal:
            df[ft] = np.log10(df[ft])
        
        df['Age.Remod'].fillna(df['Age.Remod'].median(), inplace=True)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df_num = df.select_dtypes(include=np.number)
        df_cat = df.select_dtypes(exclude=np.number)

        df_num.fillna(df_num.median(), inplace=True)

        df = pd.concat([df_num, df_cat], axis=1)

        scaler = RobustScaler()
        df[num_fts] = scaler.fit_transform(df[num_fts])

        ordinal_variables = [col for col in ordinal_variables if col in df.columns]

        ordencoder = OrdinalEncoder(cols=ordinal_variables)
        df = ordencoder.fit_transform(df)
        print("Cheguei aqui")

        df = pd.get_dummies(df, drop_first=True)

        return df
    except:
        return None