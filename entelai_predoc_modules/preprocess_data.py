from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
import pandas as pd
import numpy as np
import json


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
    if type(x) == str:
        if x.startswith('['):
            try:
                return [
                    item.strip().replace(' ', '_')
                    for item in json.loads(x.replace('"','').replace("'",'"'))
                ]
            except Exception as e:
                print(x)
                raise e 
        elif x == '':
            return []
        else:
            return [x.strip().replace(' ','_')]
    else:
        return [x]


def extract_numerical_value(x):
    if type(x) == str:
        return int(float(x.replace('[','').replace(']','').replace("'","").replace('"','')))
    else:
        return x


def preprocess_data(curated_targets, data_entelai_predoc_server):
    
    # Read curated and server data and merge them 
    df_curated_targets = pd.read_excel(curated_targets)
    df_curated_targets = df_curated_targets[subject_info_cols_rename.keys()]

    df_curated_targets.rename(
        subject_info_cols_rename,
        inplace=True,
        axis=1
    )

    df_predoc_responses = pd.read_csv(
        data_entelai_predoc_server,
        sep=';',
        keep_default_na=False
    ).drop(columns_to_drop, axis=1)

    df_predoc_responses['questionnarie_date'] = pd.to_datetime(df_predoc_responses['questionnarie_date']).dt.date.astype('datetime64[ns]')

    # Remove subject with incorrect data in some cols
    df_predoc_responses = df_predoc_responses[df_predoc_responses.patient_identifier != 17560351]

    # Mergeo el dataframe con el target con el dataframe con los features
    df = pd.merge(
        df_curated_targets,
        df_predoc_responses,
        on=subject_info_cols,
        how='inner',
        validate='1:1'
    )

    # Me quedo con la primer visita de cada paciente para evitar diagnosticos sesgados
    df.drop_duplicates(
        subset='patient_identifier',
        keep='first',
        inplace=True
    )

    # Elimino las columnas que son info del paciente
    df.drop(
        subject_info_cols,
        axis=1,
        inplace=True
    )

    df = df.reset_index(drop=True)

    # clean column names
    df.rename(
        lambda x: x.strip().replace(' ','_'),
        inplace=True,
        axis=1
    )

    numerical_features = [c.strip().replace(' ','_') for c in numerical_features]

    df[numerical_features] = df[numerical_features].apply(
        lambda col: col.apply(extract_numerical_value),
        axis=1
    )

    # scale numerical features between 0 and 1
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features].values)

    # Transform column with string representations of array to actual arrays
    array_columns = [
        c for c in df.columns
        if any([type(x) == str and x.startswith('[') for x in df[c]])
    ]

    # One hot encode categorical features
    df[array_columns] = df[array_columns].apply(
        lambda col: col.apply(extract_value), 
        axis=1
    )

    # Binarize multi labels
    new_cols = []
    for col in array_columns:
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(df[col].values)
        new_cols.append([f'{col}_{c}' for c in mlb.classes_])
        df[new_cols[-1]] = X
        df.drop(col, axis=1, inplace=True)

    feature_cols = np.concatenate(new_cols + [numerical_features])

    # Extract final X
    X = df[feature_cols]

    # Encode y label
    y = df['target'].values
    y = LabelEncoder().fit_transform(y)

    return {'X': X, 'y': y, 'features': feature_cols}
