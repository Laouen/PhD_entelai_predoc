from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler, OneHotEncoder
import pandas as pd
import numpy as np
import json

features_to_drop = [
    '¿Qué medicación/es toma actualmente cuando tiene dolor de cabeza?', # Posible biased feature by the patient doctor
    '¿Qué medicación preventiva se encuentra usando?', # Posible biased feature by the patient doctor
    '¿Qué medicación preventiva usó?', # Posible biased feature by the patient doctor
    '¿Utilizó alguna vez para su dolor aplicaciones de botox?', # Posible biased feature by the patient doctor
    'Señale la o las afirmaciones correctas en cuanto a las características  de su dolor luego del golpe en la cabeza', # This feature has almost no data
    'Cuán seguido bebe alcohol', # This feature has not enough data
    '¿Cómo considera que es su respuesta a la medicación preventiva actual?', # This feature has not enough data and it can algo be biased by the doctor
]

columns_to_drop = [
    'ai_condition',
    'ai_eval',
    'condition',
    'organization_id',
    'questionnarie_id',
    '¿Esta es su primer vez con el sistema de Pre-Consulta?'
]

subject_info_cols = {
    'ID consulta': 'medical_consultation_id',
    'ID paciente': 'patient_id',
    'DNI': 'patient_identifier',
    'ID médico': 'practitioner_id',
    'Fecha de consulta': 'questionnarie_date'
}

target_col = {
    'Diagnóstico del médico': 'condition'
}

demografic_features = [
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
]

multi_label_features = [
    'Señale si el o los síntomas que preceden su dolor de cabeza cumple alguno de las siguientes características',
    'Señale si presenta alguna de las siguientes características durante su dolor',
    'Señale si presenta alguna de las siguientes características previo a su dolor de cabeza',
    'Señale si junto con su dolor de cabeza, experimentó alguno de los siguientes síntomas',
    'He identificado los siguientes desencadenantes de mi dolor que siempre o casi siempre que me expongo, tengo dolor de cabeza',
    'Indique cuál/cuáles de las siguientes afirmaciones es correcta'
]

#key: feature col name, value: The label to drop to avoid linear correlation between new one hot encoded labels in the one vs all fashion 
single_label_features = {
    'Cuando tiene dolor de cabeza, ¿con qué frecuencia desearía poder acostarse?': 'Nunca',
    'En las últimas 4 semanas, ¿con qué frecuencia el dolor de cabeza ha limitado su capacidad para concentrarse en el trabajo o en las actividades diarias?': 'Nunca',
    'En las últimas 4 semanas, ¿con qué frecuencia se ha sentido demasiado cansada/o para trabajar o realizar las actividades diarias debido a su dolor de cabeza?': 'Nunca',
    'En las últimas 4 semanas, ¿con qué frecuencia se ha sentido harta/o o irritada/o debido a su dolor de cabeza?': 'Nunca',
    'En relación a la actividad física, marque la que corresponda': 'No hago ejercicio',
    'En relación a mis hábitos de sueño': 'Duermo bien de noche',
    'Por favor indique que parte de la cabeza o cara le duele': 'empty',
    'Señale la opción correcta con respecto a la frecuencia de sus dolores (cuantas veces tiene dolor por semana o por mes, por ejemplo)': 'Tengo_dolor_todos_los_días',
    '¿Con qué frecuencia el dolor de cabeza limita su capacidad para realizar actividades diarias habituales como las tareas domésticas, el trabajo, los estudios o actividades sociales?': 'Nunca',
    '¿Cuánto tiempo le dura la cefalea o dolor de cabeza si no toma ningún remedio, o si lo toma pero no tiene respuesta?': 'empty',
    '¿Durante el episodio de dolor de cabeza, ha notado que le molestan las luces o los sonidos y que trata de evitarlos?': 'empty',
    '¿Le dieron alguna vez medicación preventiva?': 'No',
    '¿Siente que su dolor de cabeza es pulsátil (le late la cabeza) u opresivo (siente como si le estuviesen apretando la cabeza)?': 'empty',
    '¿Cómo considera que es su respuesta a la medicación preventiva actual?': 'Mala_(sin_ninguna_respuesta)',
    'Cuando usted tiene dolor de cabeza, ¿con qué frecuencia el dolor es intenso?': 'Nunca'
}

# key: feature col name, value: The possitive label
binary_features = {
    'Actualmente bebo alcohol': 'Si',
    'Actualmente fumo tabaco en cualquiera de sus formas': 'Si',
    'Uso drogas ilícitas': 'Si',
    '¿Este dolor es algo que comenzó recientemente (en el último mes)?': 'Sí, es un dolor que nunca tuve en mi vida y que comenzó hace menos de un mes',
    '¿Ha notado que su dolor empeora con el ejercicio, como caminar, correr o subir escaleras?': 'Si, empeora si me muevo. Trato de quedarme quieto o acostado',
    '¿Ha sentido náuseas, ganas de vomitar o ha llegado a vomitar durante o después de su dolor de cabeza?': 'Si, he tenido náuseas o ganas de vomitar y/o he llegado a vomitar'
}

# all features that are strings codifying arrays of features
array_features = numerical_features + multi_label_features + single_label_features + list(binary_features.keys())

# Columns where empty values must be replaced by a token 'empty'
to_fill_empty_columns = [
    'Por favor indique que parte de la cabeza o cara le duele',
    '¿Cuánto tiempo le dura la cefalea o dolor de cabeza si no toma ningún remedio, o si lo toma pero no tiene respuesta?',
    '¿Durante el episodio de dolor de cabeza, ha notado que le molestan las luces o los sonidos y que trata de evitarlos?',
    '¿Siente que su dolor de cabeza es pulsátil (le late la cabeza) u opresivo (siente como si le estuviesen apretando la cabeza)?'
]

response_to_drug_cols = [
    '¿Cómo considera que es su respuesta a la medicación Diclofenac?',
    '¿Cómo considera que es su respuesta a la medicación Ibuprofeno (Ibupirac)?',
    '¿Cómo considera que es su respuesta a la medicación Ketorolac?',
    '¿Cómo considera que es su respuesta a la medicación Paracetamol?',
    '¿Cómo considera que es su respuesta a la medicación Ergotamina (migral)?',
    '¿Cómo considera que es su respuesta a la medicación Sumatriptán (migratriptan, micranil, imigran, rontadol)?',
    '¿Cómo considera que es su respuesta a la medicación Ácido Tolfenámico (Flocur)?',
    '¿Cómo considera que es su respuesta a la medicación Naratriptán  (Naramig)?',
    '¿Cómo considera que es su respuesta a la medicación Sumatriptán Naproxeno (Naprux)?',
    '¿Cómo considera que es su respuesta a la medicación Eletriptán (Relpax)?',
    '¿Cómo considera que es su respuesta a la medicación Indometacina?',
    '¿Cómo considera que es su respuesta al botox?'
]

invalid_conditions = [
    'dni mal',
    'duplicado',
    'mal dni',
    'mal el dni',
    'moigraña episodica midas mal',
    'no aparece el dni',
    'sin información sobre cefalea'
]

condition_group = {
    'migraña sin aura': 'migraña sin aura',
    'migraña cronica': 'migraña sin aura',
    'migraña episodica': 'migraña sin aura',
    'migraña hemiplejica': 'migraña sin aura',
    'migraña vestibular': 'migraña sin aura',
    'migraña vestibular/migraña sin aura': 'migraña sin aura',
    'cefalea mixta': 'migraña sin aura',
    'cefalea secundaria': 'cefalea secundaria',
    'cefalea 2ria': 'cefalea secundaria',
    'cefalea postrcranectomia': 'cefalea secundaria',
    'algia facial': 'algia facial',
    'algia craneal': 'algia facial',
    'algia facial atipica': 'algia facial',
    'neuralgia occipital': 'algia facial',
    'neuralgia del trigemino': 'algia facial',
    'disfuncion atm': 'algia facial',
    'cefalea trigemino autonomica': 'CTA',
    'cefalea trigémino-autonómica': 'CTA',
    'cefalea en racimos': 'CTA',
    'cefalea tensional': 'cefalea tensional',
    'cefalea al esfuerzo': 'cefalea tensional',
    'cefalea en puntadas': 'cefalea tensional',
    'migraña con aura': 'migraña con aura'
}

desencadenantes_col = 'He identificado los siguientes desencadenantes de mi dolor que siempre o casi siempre que me expongo, tengo dolor de cabeza',
desencadenantes_values = [
    'No tengo ningún desencadenante',
    'Muchas horas de sueño',
    'Pocas horas de sueño',
    'Saltearse comidas',
    'Calor intenso',
    'Cambios de clima o de la presión atmosférica',
    'Algunos alimentos u olores',
    'El período o menstruación',
    'Situaciones de stress',
    'Ejercicio moderado',
    'Luces brillantes o titilantes',
    'Vehículo en movimiento',
    'Viaje en avión',
    'Alcohol',
    'Ruido'
]

affirmations_col = 'Indique cuál/cuáles de las siguientes afirmaciones es correcta'
affirmations_values = [
    'Tengo_diagnóstico_previo_de_cáncer_con_o_sin_tratamiento',
    'Tengo_diagnóstico_previo_de_alguna_de_las_siguientes_condiciones:_HIV,_meningitis,_toxoplasmosis,_linfoma,_malfomación_vascular/cerebral',
    'Estoy_recibiendo_tratamiento_inmunosupresor_por_cualquier_causa',
    'Tengo_más_de_50_años_y_hasta_ahora_nunca_había_tenido_este_dolor_de_cabeza',
    'Tengo_fiebre_sin_una_causa_aparente_clara_(diagnóstico_de_gripe_o_resfrío,_neumonía_u_otra_infección_diagnosticada_por_un_médico)',
    'He_perdido_más_de_5_kilos_en_el_último_mes_sin_hacer_dieta_o_incrementado_actividad_física',
    'La_cefalea_o_dolor_de_cabeza_apareció_de_golpe_y_llegó_a_su_máxima_intensidad_en_tan_solo_segundos',
    'Tuve_un_golpe_muy_fuerte_en_la_cabeza_y_este_dolor_apareció_luego_o_hasta_7_días_después_del_golpe',
    'La_cefalea_solo_me_ocurre_durante_o_luego_de_toser_o_hacer_un_esfuerzo_físico',
    'El_dolor_de_cabeza_empeora_luego_de_15_minutos_parado_o_sentado_y_suele_mejorar_si_me_acuesto_por_más_de_15_minutos',
    'El_dolor_de_cabeza_me_aparece_o_apareció_durante_la_actividad_sexual',
    'Me_realizaron_recientemente_una_punción_lumbar'
]


def parse_desencadenantes_value(values):
    return [
        'otro'
        if value not in desencadenantes_values.values else value
        for value in values
    ]

def clean_affirmation_values(values):
    return [
        value for value in values
        if value in affirmations_values
    ]

# parse array values
def parse_array_value(x):
    return [
        item
        for item in json.loads(x.replace('"','').replace("'",'"'))
    ]

# parse numerical valua
def parse_numerical_value(x):
    if len(x) != 1:
        raise ValueError('numerical value should be unique')

    return int(float(x[0]))
    

def clean_condition_value(x):
    return x.replace('\n', ' ').lower().strip()


def calculate_response_to_drugs(x):
    x_str = ' '.join(x)
    for option in ['Excelente', 'Buena', 'Regular', 'Mala']:
        if option in x_str:
            return option
    return None


def check_and_extract_single_value(x):
    if len(x) > 1:
        raise ValueError(f'{x} should be an array of length 1.')
    return x[0];

def fill_empty_array_rows(x):
    return "['empty']" if x == '' else x

def preprocess_data(curated_targets_file, df_predoc_responses_file):

    ###################################################
    ## ****** Read abd clean-up curated data ******* ##
    ###################################################

    curated_targets_cols = {
        **subject_info_cols,
        **target_col
    }

    df_curated_targets = pd.read_excel(curated_targets_file)[curated_targets_cols.keys()]

    df_curated_targets.rename(
        curated_targets_cols,
        inplace=True,
        axis=1
    )

    df_curated_targets.dropna(
        subset=['condition'],
        inplace=True,
        axis=0
    )

    df_curated_targets.loc[:, 'condition'] = df_curated_targets['condition'].apply(
        clean_condition_value
    )

    valid_conditions = ~df_curated_targets['condition'].isin(
        invalid_conditions
    )
    df_curated_targets = df_curated_targets[valid_conditions]
    df_curated_targets['condition'] = df_curated_targets['condition'].apply(
        lambda x: condition_group[x]
    )

    #####################################################
    ## ****** Read and clean-up responses data ******* ##
    #####################################################
    df_predoc_responses = pd.read_csv(
        df_predoc_responses_file,
        sep=';',
        keep_default_na=False
    )

    df_predoc_responses.drop(
        columns_to_drop + features_to_drop,
        inplace=True,
        axis=1
    )

    # Remove subject with incorrect data in some cols
    df_predoc_responses = df_predoc_responses[df_predoc_responses.patient_identifier != 17560351]

    # Calculate the response to drugs without discriminate between drugs (we take the better response to drugs)
    df_predoc_responses['respuesta a medicamento'] = df_predoc_responses.loc[:,response_to_drug_cols].apply(
        calculate_response_to_drugs,
        axis=1
    )
    df_predoc_responses.drop(
        response_to_drug_cols,
        inplace=True,
        axis=1
    )

    # Parse columns with string representations of arrays to python arrays
    df_predoc_responses[to_fill_empty_columns] = df_predoc_responses[to_fill_empty_columns].replace('', "['empty']")
    df_predoc_responses[array_features] = df_predoc_responses[array_features].replace('', '[]')
    df_predoc_responses[array_features] = df_predoc_responses[array_features].apply(
        lambda col: col.apply(parse_array_value),
        axis=1
    )

    # Parse and scale numerical features between 0 and 1
    df_predoc_responses[numerical_features] = df_predoc_responses[numerical_features].apply(
        lambda col: col.apply(parse_numerical_value),
        axis=1
    )
    scaler = MinMaxScaler()
    df_predoc_responses[numerical_features] = scaler.fit_transform(
        df_predoc_responses[numerical_features].values
    )

    # extract value of excluyent values features (binary features are excluyents)
    single_value_cols = list(binary_features.keys()) + list(single_label_features.keys())
    for col in single_value_cols:
        df_predoc_responses.loc[:, col] = df_predoc_responses.loc[:, col].apply(check_and_extract_single_value)

    # Convert binary col to 0 = False, 1 = True values
    for col, pos_label in binary_features.items():
        df_predoc_responses.loc[:, col] = (df_predoc_responses.loc[:, col] == pos_label).astype(int)
    
    # One hot encode single label features droping one value per feature to break linear correlation:
    single_features, drop = zip(*single_label_features.items())
    enc = OneHotEncoder(drop=drop)
    X = enc.fit_transform(df_predoc_responses[single_features])

    new_single_label_features = [
        f'{col}"__{c}' 
        for i, col in enumerate(single_features)
        for c in enc.categories_[i] if c != drop[i]
    ]
    
    df_predoc_responses[new_single_label_features] = X
    df_predoc_responses.drop(
        single_features,
        axis=1,
        inplace=True
    )

    # Clean feature with other option
    df_predoc_responses[desencadenantes_col] = df_predoc_responses[desencadenantes_col].apply(parse_desencadenantes_value)

    # Clean feature with some incorrect values we are not sure where they come from
    df_predoc_responses[affirmations_col] = df_predoc_responses[affirmations_col].apply(clean_affirmation_values)

    # Multi label binarizer for multi label features
    new_multi_label_features = []
    for col in multi_label_features:
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(df_predoc_responses[col].values)
        new_multi_label_features.append([f'{col}__{c}' for c in mlb.classes_])
        df_predoc_responses[new_multi_label_features[-1]] = X
        df_predoc_responses.drop(col, axis=1, inplace=True)

    df_predoc_responses['questionnarie_date'] = pd.to_datetime(df_predoc_responses['questionnarie_date']).dt.date.astype('datetime64[ns]')

    # Mergeo el dataframe con el target con el dataframe con los features
    df = pd.merge(
        df_curated_targets,
        df_predoc_responses,
        on=list(subject_info_cols.values()),
        how='inner',
        validate='1:1'
    )

    # Me quedo con la primer visita de cada paciente para evitar diagnosticos sesgados
    df.sort_values(
        by='questionnarie_date',
        ascending=True,
        ignore_index=True,
        inplace=True
    )
    df.drop_duplicates(
        subset='patient_identifier',
        keep='first',
        ignore_index=True,
        inplace=True
    )

    # Elimino las columnas que son info del paciente
    df.drop(
        list(subject_info_cols.values()),
        axis=1,
        inplace=True
    )

    # final feature cols
    feature_cols = (
        demografic_features
        + numerical_features 
        + list(binary_features.keys()) 
        + new_multi_label_features 
        + new_single_label_features 
        + ['respuesta a medicamento']
    )


    # Extract final X and y matrixes
    if 'migrañas vs otras':
        y = df['condition'].isin(['migraña sin aura', 'migraña con aura']).astype(int)
        X = df[feature_cols].values
    elif 'Migraña sin aura vs otras':
        y = (df['condition'] == 'migraña sin aura').astype(int)
        X = df[feature_cols].values
    elif 'Cefalea segundaria vs resto':
        y = (df['condition'] == 'cefalea secundaria').astype(int)
        X = df[feature_cols].values
    elif 'Migraña vs CTA':
        df = df[df['condition'].isin(['migraña sin aura', 'migraña con aura', 'CTA'])]
        y = df['condition'].isin(['migraña sin aura', 'migraña con aura']).astype(int)
        X = df[feature_cols].values
    elif 'Migraña sin aura vs cefalea tensional':
        df = df[df['condition'].isin(['migraña sin aura', 'cefalea tensional'])]
        y = (df['condition'] == 'migraña sin aura').astype(int)
        X = df[feature_cols].values

    return {
        'X': X,
        'y': y,
        'features': feature_cols
    }
