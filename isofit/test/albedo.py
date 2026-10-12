# Assumed error for net radiometer derived SW albedo
# Ref example: "SURFRAD-A National SurfaceRadiation Budget Network for Atmospheric Research"
ASSUMED_ERROR=0.03


# DOZIER
#    'lat': 37.64315005500472,
#    'lon': -119.02909564511828,
# 10m 3DEP aspect: 247
# 10m 3DEP slope: 11
dozier_obs_data = {
    '20230216': 0.764818862,
    '20230220': 0.736560416,
    '20230423': 0.643165473,
    '20230626': 0.540829255,
    '20240416': 0.673550013, 
    '20250327': 0.706681957,
    '20250520': 0.558824517,
    '20260403': 0.613007784,
    '20260407': 0.564351092,
    '20260414': 0.738222058,
    '20260418': 0.671119109,
}


# SBSP
#    'lat': 37.906883,
#    'lon': -107.726265,
#aspect: 70
#slope: 7
sbsp_obs_data = {
    '20230417': 0.633429672,
    '20240224': 0.860556941,
    '20240414': 0.612840044,
}

# SASP
#    'lat': 37.906914,
#    'lon': -107.711322,
#aspect: assumed flat
#slope: assumed flat
sasp_obs_data = {
    '20230417': 0.578817,
    '20240224': 0.803019,
    '20240414': 0.60237,
}


# TABLE ROCK
# 'lat': 40.12498,
# 'lon': -105.2368,
#aspect: assumed flat
#slope: assumed flat
tablerock_obs_data = {
    '20230223': 0.792568352,
                      }


# GRAND MESA
#'lat': 39.050802,
#'lon': -108.061435,
#aspect: assumed flat
#slope: assumed flat
grandmesa_obs_data = {
    '20230402': 0.797699,
    '20230529': 0.522550053,
    '20250130': 0.765266022,
}


# SNOTEL 312 - Banner Summit
#    'lat': 44.30342,
#    'lon': -115.23447,
#aspect: 60.9
#slope: 4.5
snotel312_obs_data = {
    '20240408': 0.723129,
    '20250211': 0.709935897,
}


# SNOTEL 335
#'lat': 39.80364,
#'lon': -105.77786,
# aspect: 57
#slope: 6
snotel335_obs_data = {
    '20240407': 0.818,
    '20260130': 0.772,
}


# SNOTEL 365
#'lat': 45.89107,
#'lon': -110.93851,
#aspect: assumed flat
#slope: assumed flat
snotel365_obs_data = {
    '20260418': 0.683461117,
}


# SNOTEL 737
#'lat': 39.01467,
#'lon': -107.04933,
# aspect: 213
#slope: 8
snotel737_obs_data = {
    '20260130': 0.751235585,
}


# SNOTEL 825
#'lat': 40.53740,
#'lon': -106.67655,
#aspect: assumed flat
#slope: assumed flat
snotel825_obs_data = {
    '20250405': 0.770833333,
    '20260327': 0.55286344,
    '20260421': 0.62831792,
}


# NIWOT
#'lat': 40.0543,
#'lon': -105.5824,
#aspect: assumed flat
#slope: assumed flat
niwot_obs_data = {
    '20230203': 0.798175277,
}


# CPER
#'lat': 40.81550,
#'lon': -104.74560,
# aspect: 83
#slope: 1
cper_obs_data = {
    '20230203': 0.711876592,
}


# NEON-NG
#'lat': 46.76970,
#'lon': -100.91540,
#aspect: assumed flat
#slope: assumed flat
neonng_obs_data = {
    '20230218': 0.645,
}
