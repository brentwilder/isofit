# Assumed error for net radiometer derived SW albedo
# Ref example: "SURFRAD-A National SurfaceRadiation Budget Network for Atmospheric Research"
ASSUMED_ERROR=0.03


# DOZIER
#    'lat': 37.64315005500472,
#    'lon': -119.02909564511828,
# 10m 3DEP aspect: 226
# 10m 3DEP slope: 1
dozier_obs_data = {
    '20230216': 0.7533,
    '20230220': 0.7259,
    '20230423': 0.6401,
    '20230626': 0.5379,
    '20240416': 0.6684, 
    '20250327': 0.6979,
    '20250520': 0.5468,
    '20260403': 0.6075,
    '20260407': 0.5562,
    '20260414': 0.7334,
    '20260418': 0.6682,
}


# SBSP
#    'lat': 37.906883,
#    'lon': -107.726265,
#aspect: 236
#slope: 0
sbsp_obs_data = {
    '20230417': 0.7254,
    '20240224': 0.8114,
    '20240414': 0.6384,
}


# TABLE ROCK
# 'lat': 40.12498,
# 'lon': -105.2368,
#aspect: 81
#slope: 0
tablerock_obs_data = {
    '20230223': 0.7926,
                      }


# GRAND MESA
#'lat': 39.050802,
#'lon': -108.061435,
# aspect: 21
#slope: 0
grandmesa_obs_data = {
    '20230402': 0.7977,
    '20230529': 0.5226,
    '20250130': 0.7653,
}


# SNOTEL 335
#'lat': 39.80364,
#'lon': -105.77786,
# aspect: 104
#slope: 1
snotel335_obs_data = {
    '20240407': 0.8223,
    '20260130': 0.7831,
}


# SNOTEL 365
#'lat': 45.89107,
#'lon': -110.93851,
#aspect: 60
#slope: 0
snotel365_obs_data = {
    '20260418': 0.6835,
}


# SNOTEL 737
#'lat': 39.01467,
#'lon': -107.04933,
# aspect: 300
#slope: 1
snotel737_obs_data = {
    '20260130': 0.7438,
}


# SNOTEL 825
#'lat': 40.53740,
#'lon': -106.67655,
# aspect: 306
#slope: 0
snotel825_obs_data = {
    '20250405': 0.7708,
    '20260327': 0.5529,
    '20260421': 0.6283,
}


# NIWOT
#'lat': 40.0543,
#'lon': -105.5824,
# aspect: 81
#slope: 0
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
# aspect: 100
#slope: 0
neonng_obs_data = {
    '20230218': 0.6450,
}
