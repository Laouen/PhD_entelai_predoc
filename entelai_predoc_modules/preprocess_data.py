from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
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


subject_info_cols = [
    'medical_consultation_id',
    'patient_id',
    'patient_identifier',
    'practitioner_id',
    'questionnarie_date'
]

subject_info_cols_rename = {
    'ID consulta': 'medical_consultation_id',
    'ID paciente': 'patient_id',
    'DNI': 'patient_identifier',
    'ID médico': 'practitioner_id',
    'Fecha de consulta': 'questionnarie_date',
    'Diagnóstico del médico': 'target'
}

demografic_cols = [
    'patient_age',
    'patient_gender',
    'patient_height',
    'patient_weight'
]

numerical_features = [
    'En cuanto a la intensidad del dolor, señale en una escala de 1 a 10 la intensidad máxima a la que han llegado sus dolores',
    '¿Cuántos días en los últimos 3 meses no ha podido ir a trabajar por su cefalea?',
    '¿Cuántos días en los últimos 3 meses no ha realizado sus tareas domésticas por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses se perdió actividades familiares, sociales o lúdicas por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses se redujo por la mitad su productividad en el trabajo por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses se redujo por la mitad su productividad en la realización de tareas domésticas por la presencia de cefalea?',
    '¿Cuántos días en los últimos 3 meses se redujo por la mitad su productividad en el trabajo por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses se redujo por la mitad su productividad en la realización de tareas domésticas por la presencia de cefalea?',
    '¿Cuántos días en los últimos 3 meses se perdió actividades familiares, sociales o lúdicas por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses no ha realizado sus tareas domésticas por sus cefaleas?',
    '¿Cuántos días en los últimos 3 meses no ha podido ir a trabajar por su cefalea?'
]

binary_features = [
    'Actualmente bebo alcohol',
    'Actualmente fumo tabaco en cualquiera de sus formas',
    'Uso drogas ilícitas'
]

cat_features = [
    'Cuando tiene dolor de cabeza, ¿con qué frecuencia desearía poder acostarse?',
    'Cuando usted tiene dolor de cabeza, ¿con qué frecuencia el dolor es intenso?',
    'Cuán seguido bebe alcohol',
    'En las últimas 4 semanas, ¿con qué frecuencia el dolor de cabeza ha limitado su capacidad para concentrarse en el trabajo o en las actividades diarias?',
    'En las últimas 4 semanas, ¿con qué frecuencia se ha sentido demasiado cansada/o para trabajar o realizar las actividades diarias debido a su dolor de cabeza?',
    'En las últimas 4 semanas, ¿con qué frecuencia se ha sentido harta/o o irritada/o debido a su dolor de cabeza?',
    'En relación a la actividad física, marque la que corresponda',
    'En relación a mis hábitos de sueño',
    'He identificado los siguientes desencadenantes de mi dolor que siempre o casi siempre que me expongo, tengo dolor de cabeza',
    'Indique cuál/cuáles de las siguientes afirmaciones es correcta',
    'Por favor indique que parte de la cabeza o cara le duele',
    'Señale la opción correcta con respecto a la frecuencia de sus dolores (cuantas veces tiene dolor por semana o por mes, por ejemplo)',
    'Señale si el o los síntomas que preceden su dolor de cabeza cumple alguno de las siguientes características',
    'Señale si presenta alguna de las siguientes características durante su dolor',
    'Señale si presenta alguna de las siguientes características previo a su dolor de cabeza',
    '¿Con qué frecuencia el dolor de cabeza limita su capacidad para realizar actividades diarias habituales como las tareas domésticas, el trabajo, los estudios o actividades sociales?',
    '¿Cuánto tiempo le dura la cefalea o dolor de cabeza si no toma ningún remedio, o si lo toma pero no tiene respuesta?',
    '¿Cómo considera que es su respuesta a la medicación Diclofenac?',
    '¿Cómo considera que es su respuesta a la medicación Ibuprofeno (Ibupirac)?',
    '¿Cómo considera que es su respuesta a la medicación Ketorolac?',
    '¿Cómo considera que es su respuesta a la medicación Paracetamol?',
    '¿Durante el episodio de dolor de cabeza, ha notado que le molestan las luces o los sonidos y que trata de evitarlos?',
    '¿Este dolor es algo que comenzó recientemente (en el último mes)?',
    '¿Ha notado que su dolor empeora con el ejercicio, como caminar, correr o subir escaleras?',
    '¿Ha sentido náuseas, ganas de vomitar o ha llegado a vomitar durante o después de su dolor de cabeza?',
    '¿Le dieron alguna vez medicación preventiva? ',
    '¿Qué medicación/es toma actualmente cuando tiene dolor de cabeza?',
    '¿Siente que su dolor de cabeza es pulsátil (le late la cabeza) u opresivo (siente como si le estuviesen apretando la cabeza)?',
    '¿Cómo considera que es su respuesta a la medicación preventiva actual?',
    '¿Qué medicación preventiva se encuentra usando?',
    '¿Utilizó alguna vez para su dolor aplicaciones de botox?',
    '¿Cómo considera que es su respuesta a la medicación Ergotamina (migral)?',
    '¿Cómo considera que es su respuesta a la medicación Sumatriptán (migratriptan, micranil, imigran, rontadol)?',
    '¿Cómo considera que es su respuesta a la medicación Ácido Tolfenámico (Flocur)?',
    '¿Cómo considera que es su respuesta a la medicación Naratriptán  (Naramig)?',
    '¿Qué medicación preventiva usó?',
    '¿Cómo considera que es su respuesta a la medicación Sumatriptán Naproxeno (Naprux)?',
    '¿Cómo considera que es su respuesta a la medicación Eletriptán (Relpax)?',
    '¿Cómo considera que es su respuesta a la medicación Indometacina?',
    '¿Cómo considera que es su respuesta al botox?',
    'Señale si junto con su dolor de cabeza, experimentó alguno de los siguientes síntomas',
    'Señale la o las afirmaciones correctas en cuanto a las características  de su dolor luego del golpe en la cabeza'
]

columns_to_drop = [
    'ai_condition',
    'ai_eval',
    'condition',
    'organization_id',
    'questionnarie_id',
    '¿Esta es su primer vez con el sistema de Pre-Consulta?'
]

# Convert values to arrays and clean values
def extract_value(x):
    if x.startswith('['):
        return [item.strip().replace(' ','_') for item in json.loads(x.replace("'",'"'))]
    elif x == '':
        return []
    elif type(x) == str:
        return [x.strip().replace(' ','_')]
    else:
        return [x]

def transform_column_values(col):
    return col.apply(extract_value)

def extract_numerical_value(x):
    int(float(x.replace('[','').replace(']',''))))

def preprocess_data(curated_curated, data_entelai_predoc_server):
    
    # Read curated and server data and merge them 
    df_curated_targets = pd.read_excel(
        curated_curated,
        columns=['Diagnóstico del médico', *subject_info_cols]
    )

    df_curated_targets.rename(
        subject_info_cols_rename,
        inplace=True
    )

    df_predoc_responses = pd.read_csv(
        data_entelai_predoc_server,
        keep_default_na=False
    ).drop(columns_to_drop, axis=1)

    # Remove subject with incorrect data in some cols
    df_predoc_responses = df_predoc_responses[df_predoc_responses.patient_identifier == 17560351]

    # Me quedo con la primer visita de cada paciente para evitar diagnosticos sesgados
    df_predoc_responses.drop_duplicates(
        subset='patient_identifier',
        keep='first',
        inplace=True
    )

    # Mergeo el dataframe con el target con el dataframe con los features
    df = pd.merge(
        df_curated_targets,
        df_predoc_responses,
        on=subject_info_cols,
        how='inner',
        validate='1:1'
    ).drop(subject_info_cols, axis=1)

    # clean column names
    df.rename(lambda x: x.strip().replace(' ','_'), inplace=True)

    # process numerical cols
    for col in numerical_features:
        df[col] = df[col].apply(extract_numerical_value)

    # scale numerical features between 0 and 1
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features].values)

    # Transform column with string representations of array to actual arrays
    array_columns = [
        c for c in df.columns
        if any([type(x) == str and x.startswith('[') for x in df[c]])
    ]

    # One hot encode categorical features
    df[array_columns] = df[array_columns].apply(transform_column_values, axis=1)

    # Binarize multi labels
    for col in array_columns:
        mlb = MultiLabelBinarizer()
        new_cols = mlb.fit_transform(df[col].values)
        df[[f'{col}_{c}' for c in mlb.classes_]] = new_cols
        df.drop(col, axis=1, inplace=True)

    # extract final X, y values
    X = df.drop('target', axis=1).values
    y = df['target'].values

    return {'X': X, 'y': y}
