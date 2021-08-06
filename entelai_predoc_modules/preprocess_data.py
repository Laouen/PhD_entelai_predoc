from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import json


def json_to_csv(json_data):
    """
    Converts json responses to dataframe format
    """
    df = pd.DataFrame()
    for elem in json_data:
        df = pd.concat([df, pd.DataFrame(elem)], axis=1, sort=False)
    df = df.transpose()
    return df


def filter_diagnostics(df):
    error_col = 'En cuanto a la intensidad del dolor, señale en una escala de 1 a 10 la intensidad máxima a la que han llegado sus dolores'
    # Remove nan in condition
    new_df = df.copy()
    #remove all test ids
    new_df = new_df[new_df['practitioner_id'] != 15]
    new_df.dropna(subset=['condition'], inplace=True)
    # Format condition values
    new_df.condition = new_df.condition.apply(
        lambda x: x.replace('\n', ' ').replace('  ', ' ').strip())

    # Unify diagnostics using diagnostic_dict
    new_df.condition.replace(diagnostics_dict, inplace=True)
    # Remove None condition again
    new_df.dropna(subset=['condition'], inplace=True)

    error_col_values = new_df[error_col].apply(extract_value).astype(float)
    print(error_col_values)
    # Get index of inconsisten rows
    rows_to_drop = error_col_values[error_col_values > 10.0].index

    new_df.drop(rows_to_drop, inplace=True)

    print(new_df.condition.value_counts())
    data = new_df.drop(columns=['ai_condition', 'ai_eval', 'medical_consultation_id',
                                'condition', 'practitioner_id', 'questionnarie_id'])
    target = new_df['condition']
    target[target.isin(['Cefalea secundaria',
                        'Cefalea atribuída a disfunción temporomandibular',
                        'Cefalea por abuso de medicación', 'Neuralgia del intermediario',
                        'Cefalea asociada al ejercicio', 'Síndroma de boca ardiente'])] = 'Otras'
    return data, target


def extract_value(x):
    try:
        if len(x) == 1:
            return x[0]
        return x.replace("['", "").replace("']", "")
    except:
        return x


def one_hot_encode_multi_data(data):
    # Replace list values
    multiple_value_cols = set()
    for label, content in data.items():
        try:
            if any(map(lambda x: (type(x) == list and len(x) > 1), content)):
                multiple_value_cols.add(label)
        except:
            continue
    multiple_cols_data = data[multiple_value_cols]

    # OneHotEncode over multiple_values
    res_multiple_value_data = pd.DataFrame()
    for c in multiple_value_cols:
        
        partial_df = multiple_cols_data[c].str.join(sep='*').str.get_dummies(sep='*')
        
        partial_df.rename(columns=dict(
            map(lambda x: (x, '{}_{}'.format(c, x)), partial_df.columns)), inplace=True)

        res_multiple_value_data = pd.concat(
            [res_multiple_value_data, partial_df], axis=1, sort=False)
    return res_multiple_value_data


def one_hot_encode_cat_data(cat_data):
    cat_data = cat_data.applymap(extract_value)
    return pd.get_dummies(cat_data)


def preprocess_training(data):
    cat_dummies = one_hot_encode_cat_data(data[object_features_names])
    multi_dummies = one_hot_encode_multi_data(data[multi_features_names])
    num = data[num_features_names].applymap(extract_value).astype(float)
    return cat_dummies.join(multi_dummies).join(num)  # [boruta_cols]


def preprocess_instance(instance):
    # Create empty df with boruta columns

    instance_df = pd.DataFrame(columns=boruta_cols)
    instance_cat = pd.DataFrame(columns=object_features_names)
    instance_cat = one_hot_encode_cat_data(instance_cat.append(
        instance)[object_features_names].applymap(extract_value)).dropna(axis=1)

    instance_num = instance[num_features_names].applymap(
        extract_value).astype(float).astype(int)

    instance_mul = pd.DataFrame(columns=multi_features_names)
    instance_mul = one_hot_encode_multi_data(instance_mul.append(instance)[
                                             multi_features_names]).dropna(axis=1)

    return instance_df.append(instance_cat.join(instance_mul).join(instance_num))[boruta_cols].fillna(0).values


array_features = [
    ''
]

subject_info_cols = [
    'ID consulta',
    'ID paciente',
    'DNI',
    'ID médico',
    'Fecha de consulta'
]

feature_cols = [

]


def extract_value(x):
    if x.startswith('['):
        return json.loads(x.replace("'",'"'))
    elif x == '':
        return []
    else:
        return [x]

def transform_column_values(col):
    return col.apply(extract_value)

def preprocess_data(curated_curated, data_entelai_predoc_server):
    
    # Read curated and server data and merge them 
    df_curated_targets = pd.read_excel(
        curated_curated,
        columns=['Diagnóstico del médico', *subject_info_cols]
    )

    df_predoc_responses = pd.read_csv(
        data_entelai_predoc_server,
        columns=[
            *subject_info_cols,
            *feature_cols
        ]
    )

    df = pd.merge(
        df_curated_targets,
        df_predoc_responses,
        on=subject_info_cols,
        how='inner',
        validate='1:1'
    )

    # Transform column with string representations of array to actual arrays
    array_columns = [
        c for c in df.columns
        if any([type(x) == str and x.startswith('[') for x in df[c]])
    ]

    df[array_columns] = df[array_columns].apply(transform_column_values, axis=1)

    # Binarize multi labels
    for col in array_columns:
        mlb = MultiLabelBinarizer()
        new_cols = mlb.fit_transform(df[col].values)
        df[[f'{col}_{c}' for c in mlb.classes_]] = new_cols
        df.drop(col, axis=1, inplace=True)

    # TODO: definir las columnas que no hay que parsear, las que son listas de string y las que son listas de int o float

    single_value_cols = [
        col for col in df.columns
        if all(df[col].apply(lambda x: len(x) == 1).tolist())
    ]

    # TODO: poner un assert que mire que todas las columnas están contenidas
    
    # Me quedo con la primer visita de cada paciente para evitar diagnosticos sesgados
    df.drop_duplicates(subset='DNI', keep='first', inplace=True)


    return {
        'X': df[feature_cols].values,
        'y': df['Diagnóstico del médico'].values
    }
