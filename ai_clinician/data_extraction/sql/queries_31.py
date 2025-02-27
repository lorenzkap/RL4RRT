ANTIBIOTIC_GSN_CODES = (
    '002542','002543','007371','008873','008877','008879','008880','008935',
    '008941','008942','008943','008944','008983','008984','008990','008991',
    '008992','008995','008996','008998','009043','009046','009065','009066',
    '009136','009137','009162','009164','009165','009171','009182','009189',
    '009213','009214','009218','009219','009221','009226','009227','009235',
    '009242','009263','009273','009284','009298','009299','009310','009322',
    '009323','009326','009327','009339','009346','009351','009354','009362',
    '009394','009395','009396','009509','009510','009511','009544','009585',
    '009591','009592','009630','013023','013645','013723','013724','013725',
    '014182','014500','015979','016368','016373','016408','016931','016932',
    '016949','018636','018637','018766','019283','021187','021205','021735',
    '021871','023372','023989','024095','024194','024668','025080','026721',
    '027252','027465','027470','029325','029927','029928','037042','039551',
    '039806','040819','041798','043350','043879','044143','045131','045132',
    '046771','047797','048077','048262','048266','048292','049835','050442',
    '050443','051932','052050','060365','066295','067471',
    '006816', '006817', '017956', '066110', '006705', '051558',  '006704', '006812'
)

CHARTEVENT_CODES = (
    226707, 581, 198, 228096, 211, 220179, 220181, 8368, 220210, 220277, 3655, 
    223761, 220074, 492, 491, 8448, 116, 626, 467, 223835, 190, 470, 220339, 
    224686, 224687, 224697, 224695, 224696, 226730, 580, 220045, 225309, 220052, 
    8441, 3337, 646, 223762, 678, 113, 1372, 3420, 471, 506, 224684, 450, 444, 
    535, 543, 224639, 6701, 225312, 225310, 224422, 834, 1366, 160, 223834, 505, 
    684, 448, 226512, 6, 224322, 8555, 618, 228368, 727, 227287, 224700, 224421, 
    445, 227243, 6702, 8440, 3603, 228177, 194, 3083, 224167, 443, 615, 224691, 
    2566, 51, 52, 654, 455, 456, 3050, 681, 2311, 220059, 220061, 220060, 226732,
    224690, 226457, 224144,
    224191, 224154, 226748, 226499, 228004, 228005, 228006
)

# COMORBIDITY_FIELDS = [
#     'congestive_heart_failure', 'cardiac_arrhythmias', 'valvular_disease',
#     'pulmonary_circulation', 'peripheral_vascular', 'hypertension', 'paralysis',
#     'other_neurological', 'chronic_pulmonary', 'diabetes_uncomplicated',
#     'diabetes_complicated', 'hypothyroidism', 'renal_failure', 'liver_disease',
#     'peptic_ulcer', 'aids', 'lymphoma', 'metastatic_cancer', 'solid_tumor',
#     'rheumatoid_arthritis', 'coagulopathy', 'obesity', 'weight_loss',
#     'fluid_electrolyte', 'blood_loss_anemia', 'deficiency_anemias', 'alcohol_abuse',
#     'drug_abuse', 'psychoses', 'depression'
# ]

CULTURE_CODES = (
    6035,3333,938,941,942,4855,6043,2929,225401,225437,225444,225451,225454,
    225814,225816,225817,225818,225722,225723,225724,225725,225726,225727,
    225728,225729,225730,225731,225732,225733,227726,70006,70011,70012,70013,
    70014,70016,70024,70037,70041,225734,225735,225736,225768,70055,70057,70060,
    70063,70075,70083,226131,80220
)

INPUTEVENT_CODES = (
    225158,225943,226089,225168,225828,225823,220862,220970,220864,225159,
    220995,225170,225825,227533,225161,227531,225171,225827,225941,225823,
    225825,225941,225825,228341,225827,30018,30021,30015,30296,30020,30066,
    30001,30030,30060,30005,30321,3000630061,30009,30179,30190,30143,30160,
    30008,30168,30186,30211,30353,30159,30007,30185,30063,30094,30352,30014,
    30011,30210,46493,45399,46516,40850,30176,30161,30381,30315,42742,30180,
    46087,41491,30004,42698,42244,
    223258, 221347, 221794, 225152, 229299, 221653, 221429
)

LABS_CE_CODES = (
    223772, 829, 1535, 227442, 227464, 4195, 3726, 3792, 837, 220645, 4194, 
    3725, 3803, 226534, 1536, 4195, 3726, 788, 220602, 1523, 4193, 3724, 226536,
    3747, 225664, 807, 811, 1529, 220621, 226537, 3744, 781, 1162, 225624, 3737,
    791, 1525, 220615, 3750, 821, 1532, 220635, 786, 225625, 1522, 3746, 816,
    225667, 3766, 777, 787, 770, 3801, 769, 3802, 1538, 848, 225690, 803, 1527, 
    225651, 3807, 1539, 849, 772, 1521, 227456, 3727, 227429, 851, 227444, 814, 
    220228, 813, 220545, 3761, 226540, 4197, 3799, 1127, 1542, 220546, 4200, 
    3834, 828, 227457, 3789, 825, 1533, 227466, 3796, 824, 1286, 1671, 1520, 
    768, 220507, 815, 1530, 227467, 780, 1126, 3839, 4753, 779, 490, 3785, 3838, 
    3837, 778, 3784, 3836, 3835, 776, 224828, 3736, 4196, 3740, 74, 225668, 1531, 
    227443, 1817, 228640, 823, 227686, 220587, 227465, 220224, 226063, 226770, 
    227039, 220235, 226062, 227036,
    227073, 220580, 227468, 220644, 229355, 225677, 220227, 225693, 226541, 220632,
    227445, 227446
)

LABS_LE_CODES = (
     50971, 50822, 50824, 50806, 50931, 51081, 50885, 51003, 51222, 50810, 51301, 50983,
     50902, 50809, 51006, 50912, 50960, 50893, 50808, 50804, 50878, 50861, 51464, 50883,
     50976, 50862, 51002, 50889, 50811, 51221, 51279, 51300, 51265, 51275, 51274, 51237,
     50820, 50821, 50818, 50802, 50813, 50882, 50803, 52167, 52166, 52165, 52923, 51624,
     52647,
     50952, 50993, 51082, 51097, 51100, 51104, 51080, 51001, 50927, 51695, 51229, 52044,
     51676, 50868, 52500, 50866, 51623, 52116, 52075, 51000, 50908, 51580
)

MECHVENT_MEASUREMENT_CODES = (
    445, 448, 449, 450, 1340, 1486, 1600, 224687, # minute volume
    639, 654, 681, 682, 683, 684,224685,224684,224686, # tidal volume
    218,436,535,444,459,224697,224695,224696,224746,224747, # High/Low/Peak/Mean/Neg insp force ("RespPressure")
    221,1,1211,1655,2000,226873,224738,224419,224750,227187, # Insp pressure
    543, # PlateauPressure
    5865,5866,224707,224709,224705,224706, # APRV pressure
    60,437,505,506,686,220339,224700, # PEEP
    3459, # high pressure relief
    501,502,503,224702, # PCV
    223,667,668,669,670,671,672, # TCPCV
    157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810, # ETT
    224701, # PSVlevel
)

MECHVENT_CODES = (
    640, # extubated
    720, # vent type
    467, # O2 delivery device
) + MECHVENT_MEASUREMENT_CODES

PREADM_FLUID_CODES = (
    30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
    226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
    227071,227072
)

UO_CODES = (
    40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 
    40428, 40096, 40651, 226559, 226560, 227510, 226561, 227489, 226584,
    226563, 226564, 226565, 226557, 226558
)

VASO_CODES = (
    30128, 30120, 30051, 221749, 221906, 30119, 30047, 30127, 221289,
    222315, 221662, 30043, 30307
)

PROCEDURE_EVENTS_CODES = ( # TODO
    225752, 224270
)

DATE_TIME_EVENTS_CODES = ( # TODO
    229352
)

def abx():
    """
    Antibiotics administration. This is gathered from mimiciv_hosp.prescriptions
    (mimiciv_icu.icustays is used to get the stay ID), where prescriptions are 
    filtered to a set of Generic Sequence Numbers (GSN) corresponding to 
    antibiotics.
    """
    base = """
        SELECT 
            p.hadm_id, 
            i.stay_id AS icustay_id, 
            EXTRACT(EPOCH FROM p.starttime::timestamp) AS startdate, 
            EXTRACT(EPOCH FROM p.stoptime::timestamp) AS enddate, 
            gsn, ndc, dose_val_rx, dose_unit_rx, route
        FROM physionet_data.mimiciv_hosp.prescriptions AS p
            LEFT OUTER JOIN physionet_data.mimiciv_icu.icustays AS i 
            ON p.hadm_id = i.hadm_id
        WHERE gsn IN {gsn}
        ORDER BY p.hadm_id, i.stay_id
    """
    
    return base.format(gsn=repr(ANTIBIOTIC_GSN_CODES))

def ce(min_stay, max_stay):
    """
    Chart events - the bulk of information about a patient's stay, including 
    vital signs, ventilator settings, lab values, code status, mental status, 
    etc. (see MIMIC documentation for table mimiciv_icu.chartevents).
    """
    query = """
    SELECT DISTINCT {stay_id_field} AS icustay_id, 
        EXTRACT(EPOCH FROM charttime::timestamp) AS charttime,  -- Cast charttime to timestamp
        itemid, 
        CASE 
            WHEN LOWER(value) = 'none' THEN 0 
            WHEN LOWER(value) = 'ventilator' THEN 1 
            WHEN LOWER(value) IN ('cannula', 'nasal cannula', 'high flow nasal cannula') THEN 2 
            WHEN LOWER(value) = 'face tent' THEN 3 
            WHEN LOWER(value) = 'aerosol-cool' THEN 4 
            WHEN LOWER(value) = 'trach mask' THEN 5 
            WHEN LOWER(value) = 'hi flow neb' THEN 6 
            WHEN LOWER(value) = 'non-rebreather' THEN 7 
            WHEN LOWER(value) = '' THEN 8  
            WHEN LOWER(value) = 'venti mask' THEN 9 
            WHEN LOWER(value) = 'medium conc mask' THEN 10 
            ELSE valuenum 
        END AS valuenum 
    FROM {table} 
    WHERE {stay_id_field} >= {min_stay} AND {stay_id_field} < {max_stay} 
        AND value IS NOT NULL AND itemid IN {codes}  
    ORDER BY icustay_id, charttime
    """
    
    kwargs = {
        'min_stay': min_stay, 
        'max_stay': max_stay, 
        'codes': repr(CHARTEVENT_CODES)
    }
    
    
    kwargs['table'] = 'physionet_data.mimiciv_icu.chartevents'
    kwargs['stay_id_field'] = 'stay_id'
        
    return query.format(**kwargs)

# def comorbidities(elixhauser_table, mimiciii=False):
#     """
#     Table from which the Elixhauser-Quan score is calculated. This table is
#     extracted not to provide explicit features for the MIMIC calculation, but
#     rather to inform clinicians about patients' specific comorbidities.
#     """
#     if mimiciii:
#         query = """
#             select ad.subject_id, ad.hadm_id, i.icustay_id as icustay_id, {fields}
#             from `physionet_data.mimiciii_clinical.admissions` as ad, `physionet_data.mimiciii_clinical.icustays` as i, `{elix}` as elix
#             where ad.hadm_id=i.hadm_id and elix.hadm_id=ad.hadm_id
#             order by subject_id asc, intime asc
#         """
#     else:
#         query = """
#             select ad.subject_id, ad.hadm_id, i.stay_id as icustay_id, {fields}
#             from `physionet_data.mimiciv_hosp.admissions` as ad, `physionet_data.mimiciv_icu.icustays` as i, `{elix}` as elix
#             where ad.hadm_id=i.hadm_id and elix.hadm_id=ad.hadm_id
#             order by subject_id asc, intime asc
#         """
#     return query.format(elix=elixhauser_table, fields=', '.join(COMORBIDITY_FIELDS))

def culture():
    """
    According to Komorowski, these "correspond to blood/urine/CSF/sputum
    cultures etc". This is extracted from mimiciv_icu.chartevents, where the item
    ID is within a set of particular measurement types (see the table
    derived_data.culture_itemids).
    """
    query = """
        SELECT subject_id, hadm_id, {stay_id_field} AS icustay_id, 
               EXTRACT(EPOCH FROM charttime::timestamp) AS charttime, 
               itemid
        FROM {table}
        WHERE itemid IN {codes}
        ORDER BY subject_id, hadm_id, charttime
    """
    kwargs = {'codes': repr(CULTURE_CODES)}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['table'] = 'physionet_data.mimiciv_icu.chartevents'
    
    return query.format(**kwargs)

# def demog(elixhauser_table, mimiciii=False):
def demog():
    """
    Demographic information, including dates of admission, discharge, and death,
    as well as comorbidities. This is extracted from several tables, including 
    mimiciv_hosp.admissions (admission and discharge times), mimiciv_icu.icustays 
    (ICU type and timing), mimiciv_hosp.patients (gender, age, and date of death),
    and the derived Elixhauser-Quan comorbidities measure.
    """
    query = """
        SELECT 
            ad.subject_id,
            ad.hadm_id, 
            i.{stay_id_field} AS icustay_id,
            EXTRACT(EPOCH FROM ad.admittime::timestamp) AS admittime,
            EXTRACT(EPOCH FROM ad.dischtime::timestamp) AS dischtime, 
            ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
            CASE
                WHEN i.first_careunit = 'NICU' THEN 5
                WHEN i.first_careunit = 'SICU' THEN 2
                WHEN i.first_careunit = 'CSRU' THEN 4
                WHEN i.first_careunit = 'CCU' THEN 6
                WHEN i.first_careunit = 'MICU' THEN 1
                WHEN i.first_careunit = 'TSICU' THEN 3
            END AS unit,
            EXTRACT(EPOCH FROM i.intime::timestamp) AS intime,
            EXTRACT(EPOCH FROM i.outtime::timestamp) AS outtime, 
            i.los,
            {age} AS age,
            {dob} AS dob,
            EXTRACT(EPOCH FROM p.dod::timestamp) AS dod,
            p.dod IS NOT NULL AS expire_flag,
            CASE
                WHEN p.gender = 'M' THEN 1
                WHEN p.gender = 'F' THEN 2
            END AS gender,
            -- Properly casting p.dod and ad.dischtime to timestamp
            CAST((p.dod::timestamp - ad.dischtime::timestamp <= INTERVAL '24 hours' AND p.dod IS NOT NULL) AS INT) AS morta_hosp,  
            CAST((p.dod::timestamp - i.intime::timestamp <= INTERVAL '90 days' AND p.dod IS NOT NULL) AS INT) AS morta_90
        FROM {admissions} AS ad
        JOIN {icustays} AS i ON ad.hadm_id = i.hadm_id
        JOIN {patients} AS p ON p.subject_id = i.subject_id
        ORDER BY ad.subject_id ASC, i.intime ASC
    """
    
    kwargs = {}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['admissions'] = 'physionet_data.mimiciv_hosp.admissions'
    kwargs['icustays'] = 'physionet_data.mimiciv_icu.icustays'
    kwargs['patients'] = 'physionet_data.mimiciv_hosp.patients'
    kwargs['dob'] = 'p.anchor_year - p.anchor_age'
    kwargs['age'] = 'EXTRACT(year FROM i.intime::timestamp) - p.anchor_year + p.anchor_age'
        
    return query.format(**kwargs)

def fluid_mv():
    """
    Real-time fluid input from Metavision. 
    Adjustments include handling different item IDs and calculating the total equivalent volume.
    """
    query = """
        WITH t1 AS
        (
            SELECT
                {stay_id_field} AS icustay_id,
                EXTRACT(EPOCH FROM starttime::timestamp) AS starttime,
                EXTRACT(EPOCH FROM endtime::timestamp) AS endtime,
                itemid, 
                amount, 
                rate,
                CASE
                    WHEN itemid IN (30176,30315) THEN amount * 0.25
                    WHEN itemid IN (30161) THEN amount * 0.3
                    WHEN itemid IN (30020,30015,225823,30321,30186,30211,30353,42742,42244,225159) THEN amount * 0.5
                    WHEN itemid IN (227531) THEN amount * 2.75
                    WHEN itemid IN (30143,225161) THEN amount * 3
                    WHEN itemid IN (30009,220862) THEN amount * 5
                    WHEN itemid IN (30030,220995,227533) THEN amount * 6.66
                    WHEN itemid IN (228341) THEN amount * 8
                    ELSE amount
                END AS tev -- total equivalent volume
            FROM {table}
            WHERE {stay_id_field} IS NOT NULL 
            AND amount IS NOT NULL 
            AND itemid IN {items}
        )
        SELECT
            icustay_id,
            starttime, 
            endtime,
            itemid, 
            ROUND(CAST(amount AS numeric), 3) AS amount,
            ROUND(CAST(rate AS numeric), 3) AS rate,
            ROUND(CAST(tev AS numeric), 3) AS tev -- total equivalent volume
        FROM t1
        ORDER BY icustay_id, starttime, itemid
    """
    
    # Define the input event codes for PostgreSQL query
    kwargs = {'items': repr(INPUTEVENT_CODES)}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['table'] = 'physionet_data.mimiciv_icu.inputevents'
    
    return query.format(**kwargs)
    
def labs_ce():
    """Lab events extracted from the chartevents table."""
    query = """
        SELECT
            {stay_id_field} AS icustay_id,
            EXTRACT(EPOCH FROM charttime::timestamp) AS charttime,
            itemid,
            valuenum
        FROM {table}
        WHERE valuenum IS NOT NULL 
        AND {stay_id_field} IS NOT NULL 
        AND itemid IN {codes}
        ORDER BY icustay_id, charttime, itemid
    """
    kwargs = {'codes': repr(LABS_CE_CODES)}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['table'] = 'physionet_data.mimiciv_icu.chartevents'
    
    return query.format(**kwargs)

def labs_le():
    """Lab events extracted from the labevents table."""
    query = """
        SELECT
            xx.icustay_id,
            EXTRACT(EPOCH FROM f.charttime::timestamp) AS timestp, 
            f.itemid, 
            f.valuenum
        FROM (
            SELECT subject_id, hadm_id, {stay_id_field} AS icustay_id, intime, outtime
            FROM {stays}
            GROUP BY subject_id, hadm_id, icustay_id, intime, outtime
        ) AS xx
        INNER JOIN {events} AS f
        ON f.hadm_id = xx.hadm_id
        AND EXTRACT(EPOCH FROM f.charttime::timestamp) - EXTRACT(EPOCH FROM xx.intime::timestamp) >= 24*3600
        AND EXTRACT(EPOCH FROM xx.outtime::timestamp) - EXTRACT(EPOCH FROM f.charttime::timestamp) >= 24*3600
        AND f.itemid IN {codes}
        AND f.valuenum IS NOT NULL
        ORDER BY f.hadm_id, timestp, f.itemid
    """
    kwargs = {'codes': repr(LABS_LE_CODES)}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['stays'] = 'physionet_data.mimiciv_icu.icustays'
    kwargs['events'] = 'physionet_data.mimiciv_hosp.labevents'
    
    return query.format(**kwargs)

def mechvent_pe():
    """
    Mechanical ventilation information, extracted from the procedureevents table (MIMIC-IV only).
    """
    return """
        SELECT subject_id,
               hadm_id,
               stay_id,
               EXTRACT(EPOCH FROM starttime::timestamp) AS starttime,
               EXTRACT(EPOCH FROM endtime::timestamp) AS endtime,
               CASE 
                   WHEN itemid IN (225792, 225794, 224385, 225433) THEN 1 
                   ELSE 0 
               END AS mechvent,
               CASE 
                   WHEN itemid IN (227194, 227712, 225477, 225468) THEN 1 
                   ELSE 0 
               END AS extubated,
               CASE 
                   WHEN itemid = 225468 THEN 1 
                   ELSE 0 
               END AS selfextubated,
               itemid,
               CASE 
                   WHEN valueuom = 'hour' THEN value * 60
                   WHEN valueuom = 'min' THEN value
                   WHEN valueuom = 'day' THEN value * 60 * 24
                   ELSE value 
               END AS value
        FROM physionet_data.mimiciv_icu.procedureevents
        WHERE itemid IN (225792, 225794, 227194, 227712, 224385, 225433, 225468, 225477)
    """

def mechvent():
    """
    Default mechanical ventilation information, extracted from chartevents. This
    is supplemented by data from the procedureevents table in MIMIC-IV.
    """
    query = """
        SELECT
            {stay_id_field} AS icustay_id,
            EXTRACT(EPOCH FROM charttime::timestamp) AS charttime,  -- case statement determining whether it is an instance of mech vent
            MAX(
                CASE
                    WHEN itemid IS NULL OR value IS NULL THEN 0  -- can't have null values
                    WHEN itemid = 720 AND value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
                    WHEN itemid = 467 AND value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
                    WHEN itemid IN {measurement_codes} THEN 1
                    ELSE 0
                END
            ) AS MechVent,
            MAX(
                CASE
                    WHEN itemid IS NULL OR value IS NULL THEN 0
                    WHEN itemid = 640 AND value = 'Extubated' THEN 1
                    WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
                    ELSE 0
                END
            ) AS Extubated,
            MAX(
                CASE
                    WHEN itemid IS NULL OR value IS NULL THEN 0
                    WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
                    ELSE 0
                END
            ) AS SelfExtubated
        FROM {events} ce
        WHERE value IS NOT NULL
        AND itemid IN {codes}
        GROUP BY icustay_id, charttime
    """
    kwargs = {'codes': repr(MECHVENT_CODES), 'measurement_codes': repr(MECHVENT_MEASUREMENT_CODES)}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['events'] = 'physionet_data.mimiciv_icu.chartevents'
    
    return query.format(**kwargs)

def microbio():
    """
    Date and time of all microbiology events (whether they are positive or
    negative). According to the MIMIC documentation:

        Microbiology tests are a common procedure to check for infectious growth 
        and to assess which antibiotic treatments are most effective. If a blood
        culture is requested for a patient, then a blood sample will be taken 
        and sent to the microbiology lab. The time at which this blood sample is 
        taken is the charttime. The spec_type_desc will indicate that this is a 
        blood sample. Bacteria will be cultured on the blood sample, and the 
        remaining columns depend on the outcome of this growth:

        - If no growth is found, the remaining columns will be NULL
        - If bacteria is found, then each organism of bacteria will be present 
        in org_name, resulting in multiple rows for the single specimen (i.e. 
        multiple rows for the given spec_type_desc).
        - If antibiotics are tested on a given bacterial organism, then each 
        antibiotic tested will be present in the ab_name column (i.e. multiple 
        rows for the given org_name associated with the given spec_type_desc). 
        Antibiotic parameters and sensitivities are present in the remaining 
        columns (dilution_text, dilution_comparison, dilution_value, 
        interpretation).
    """
    query = """
        SELECT
            m.subject_id, 
            m.hadm_id, 
            i.{stay_id_field} AS icustay_id, 
            EXTRACT(EPOCH FROM m.charttime::timestamp) AS charttime, 
            EXTRACT(EPOCH FROM m.chartdate::timestamp) AS chartdate, 
            org_itemid, 
            spec_itemid, 
            ab_itemid, 
            interpretation
        FROM {events} m 
        LEFT OUTER JOIN {stays} i 
        ON m.subject_id = i.subject_id AND m.hadm_id = i.hadm_id
    """
    kwargs = {}
    
    kwargs['stay_id_field'] = 'stay_id'
    kwargs['stays'] = 'physionet_data.mimiciv_icu.icustays'
    kwargs['events'] = 'physionet_data.mimiciv_hosp.microbiologyevents'
    
    return query.format(**kwargs)

def preadm_fluid():
    """
    Pre-admission fluid intake, as measured from the
    physionet_data.mimiciv_icu.inputevents table.
    """ 
    # Metavision only
    query = """
        WITH mv AS
        (
            SELECT ie.stay_id AS icustay_id, SUM(ie.amount) AS sum
            FROM physionet_data.mimiciv_icu.inputevents ie
            JOIN physionet_data.mimiciv_icu.d_items ci
            ON ie.itemid = ci.itemid
            WHERE ie.itemid IN {codes}
            GROUP BY icustay_id
        )
        SELECT pt.stay_id AS icustay_id,
            CASE WHEN mv.sum IS NOT NULL THEN mv.sum
            ELSE NULL END AS inputpreadm
        FROM physionet_data.mimiciv_icu.icustays pt
        LEFT OUTER JOIN mv
        ON mv.icustay_id = pt.stay_id
        ORDER BY icustay_id
    """
    kwargs = {'codes': repr(PREADM_FLUID_CODES)}
    return query.format(**kwargs)

def preadm_uo():
    """
    Pre-admission output events - information regarding patient outputs
    including urine, drainage, and so on (MIMIC documentation). There is only
    one item ID selected here, which is "Pre-Admission" (all lumped into one
    value).
    """
    query = """
        SELECT DISTINCT oe.{stay_id_field} AS icustay_id,
            EXTRACT(EPOCH FROM CAST(oe.charttime AS timestamp)) AS charttime,  -- Cast charttime to timestamp
            oe.itemid,
            oe.value,
            EXTRACT(EPOCH FROM (CAST(ic.intime AS timestamp) - CAST(oe.charttime AS timestamp))) / 60 AS datediff_minutes  -- Cast both intime and charttime
        FROM physionet_data.mimiciv_icu.outputevents oe
        JOIN physionet_data.mimiciv_icu.icustays ic
        ON oe.{stay_id_field} = ic.{stay_id_field}
        WHERE itemid IN (40060, 226633)    
        ORDER BY icustay_id, charttime, itemid
    """
    kwargs = {}
    kwargs['stay_id_field'] = 'stay_id'
    return query.format(**kwargs)

def uo():
    """
    Real-time urine output events from mimiciv_icu.outputevents.
    """
    query = """
        SELECT {stay_id_field} AS icustay_id,
            EXTRACT(EPOCH FROM CAST(charttime AS timestamp)) AS charttime,  -- Cast charttime to timestamp
            itemid,
            value
        FROM physionet_data.mimiciv_icu.outputevents
        WHERE {stay_id_field} IS NOT NULL AND value IS NOT NULL AND itemid IN {codes}
        ORDER BY icustay_id, charttime, itemid
    """
    kwargs = {'codes': repr(UO_CODES)}
    kwargs['stay_id_field'] = 'stay_id'
    return query.format(**kwargs)

def vaso_base(mv):
    """
    Real-time vasopressor input from Metavision. From the original Komorowski
    data extraction code:
    * Drugs converted in noradrenaline-equivalent
    * Body weight assumed 80 kg when missing

    Drugs selected are epinephrine, dopamine, phenylephrine, norepinephrine,
    vasopressin. CareVue also contains Levophed and Neosynephrine (extracted in
    MIMIC-III only).
    """
    query = """
    SELECT {stay_id_field} AS icustay_id,
        itemid, 
        {times},
        CASE 
            WHEN itemid IN (30120, 221906, 30047) AND rateuom='mcg/kg/min' THEN ROUND(CAST(rate AS NUMERIC), 3)  -- norad
            WHEN itemid IN (30120, 221906, 30047) AND rateuom='mcg/min' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3)  -- norad
            WHEN itemid IN (30119, 221289) AND rateuom='mcg/kg/min' THEN ROUND(CAST(rate AS NUMERIC), 3) -- epi
            WHEN itemid IN (30119, 221289) AND rateuom='mcg/min' THEN ROUND(CAST(rate / 80 AS NUMERIC), 3) -- epi
            WHEN itemid IN (30051, 222315) AND rate > 0.2 THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin, in U/h
            WHEN itemid IN (30051, 222315) AND rateuom='units/min' THEN ROUND(CAST(rate * 5 AS NUMERIC), 3) -- vasopressin
            WHEN itemid IN (30051, 222315) AND rateuom='units/hour' THEN ROUND(CAST(rate * 5 / 60 AS NUMERIC), 3) -- vasopressin
            WHEN itemid IN (30128, 221749, 30127) AND rateuom='mcg/kg/min' THEN ROUND(CAST(rate * 0.45 AS NUMERIC), 3) -- phenyl
            WHEN itemid IN (30128, 221749, 30127) AND rateuom='mcg/min' THEN ROUND(CAST(rate * 0.45 / 80 AS NUMERIC), 3) -- phenyl
            WHEN itemid IN (221662, 30043, 30307) AND rateuom='mcg/kg/min' THEN ROUND(CAST(rate * 0.01 AS NUMERIC), 3)  -- dopa
            WHEN itemid IN (221662, 30043, 30307) AND rateuom='mcg/min' THEN ROUND(CAST(rate * 0.01 / 80 AS NUMERIC), 3) 
            ELSE NULL 
        END AS rate_std -- dopa
    FROM {events}
    WHERE itemid IN {codes} AND rate IS NOT NULL {mv_conditions}
    ORDER BY icustay_id, itemid, {sort};
    """

    kwargs = {
        'codes': repr(VASO_CODES)
    }

    if mv:
        kwargs['mv_conditions'] = "AND statusdescription <> 'Rewritten'"
        kwargs['times'] = "EXTRACT(EPOCH FROM starttime::timestamp) AS starttime, EXTRACT(EPOCH FROM endtime::timestamp) AS endtime"
        kwargs['sort'] = "starttime"
    else:
        kwargs['mv_conditions'] = ""
        kwargs['times'] = "EXTRACT(EPOCH FROM charttime::timestamp) AS charttime"
        kwargs['sort'] = "charttime"

    kwargs['stay_id_field'] = 'stay_id'
    kwargs['stays'] = 'physionet_data.mimiciv_icu.icustays'
    kwargs['events'] = 'physionet_data.mimiciv_icu.inputevents'

    return query.format(**kwargs)

def vaso_mv():
    return vaso_base(True)

SQL_QUERY_FUNCTIONS = {
    # "abx": abx,
    # "ce": ce,
    # "culture": culture,
    # "demog": demog,
    # "fluid_mv": fluid_mv,
    # "labs_ce": labs_ce,
    # "labs_le": labs_le,
    # "mechvent_pe": mechvent_pe,
    # "mechvent": mechvent,
    # "microbio": microbio,
    # "preadm_fluid": preadm_fluid,
    # "preadm_uo": preadm_uo,
    # "uo": uo,
    "vaso_mv": vaso_mv
}