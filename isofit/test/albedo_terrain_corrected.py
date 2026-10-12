# Assumed error for net radiometer derived SW albedo
# Ref example: "SURFRAD-A National SurfaceRadiation Budget Network for Atmospheric Research"
ASSUMED_ERROR=0.03


# DOZIER
#    'lat': 37.64315005500472,
#    'lon': -119.02909564511828,
# 10m 3DEP aspect: 250
# 10m 3DEP slope: 8
dozier_obs_data = {
    '20230216': 0.708602,
    '20230220': 0.725934,
    '20230423': 0.650216,
    '20230626': 0.570874,
    '20240416': 0.646818,
    '20250327': 0.667344,
    '20250520': 0.493705,
    '20260403': 0.641354,
    '20260407': 0.638707,
    '20260414': 0.712538,
    '20260418': 0.676168,
}


# SBSP
#    'lat': 37.906883,
#    'lon': -107.726265,
#aspect: 70
#slope: 7
sbsp_obs_data = {
    '20230417':0.675501,
    '20240224':0.816607,
    '20240414': 0.658957,
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
    '20230223': 0.7926,
                      }


# GRAND MESA
#'lat': 39.050802,
#'lon': -108.061435,
#aspect: assumed flat
#slope: assumed flat
grandmesa_obs_data = {
    '20230402': 0.7977,
    '20230529': 0.5226,
    '20250130': 0.7653,
}

# SNOTEL 312 - Banner Summit
#    'lat': 44.30342,
#    'lon': -115.23447,
#aspect: 60.9
#slope: 4.5
snotel312_obs_data = {
    '20240408': 0.730103,
    '20250211': 0.785167,
}

# SNOTEL 335
#'lat': 39.80364,
#'lon': -105.77786,
# aspect: 57
#slope: 6
snotel335_obs_data = {
    '20240407': 0.805522,
    '20260130': 0.807328,
}
     

# SNOTEL 365
#'lat': 45.89107,
#'lon': -110.93851,
#aspect: 64
#slope: 7
snotel365_obs_data = {
    '20260418': 0.674239,
}


# SNOTEL 737
#'lat': 39.01467,
#'lon': -107.04933,
# aspect: 261
#slope: 10
snotel737_obs_data = {
    '20260130': 0.778794,
}


# SNOTEL 825
#'lat': 40.53740,
#'lon': -106.67655,
#aspect: assumed flat
#slope: assumed flat
snotel825_obs_data = {
    '20250405': 0.7708,
    '20260327': 0.5529,
    '20260421': 0.6283,
}


# NIWOT
#'lat': 40.0543,
#'lon': -105.5824,
#aspect: assumed flat
#slope: assumed flat
niwot_obs_data = {
    '20230203': 0.7982,
}


# CPER
#'lat': 40.81550,
#'lon': -104.74560,
# aspect: 83
#slope: 1
cper_obs_data = {
    '20230203': 0.7253,
}


# NEON-NG
#'lat': 46.76970,
#'lon': -100.91540,
#aspect: assumed flat
#slope: assumed flat
neonng_obs_data = {
    '20230218': 0.6450,
}
