"""Project configuration for concrete performance prediction."""

FEATURE_COLUMNS = [
    "cement_kg_m3",
    "water_kg_m3",
    "fine_aggregate_kg_m3",
    "coarse_aggregate_kg_m3",
    "recycled_coarse_aggregate_pct",
    "silica_fume_pct",
    "fly_ash_pct",
    "superplasticizer_pct",
    "curing_days",
    "saturation_ratio_pct",
    "binder_kg_m3",
    "water_binder_ratio",
]

TARGET_COLUMNS = [
    "compressive_strength_mpa",
    "splitting_tensile_strength_mpa",
    "porosity_pct",
    "electrical_resistivity_kohm_cm",
    "water_absorption_pct",
    "co2_emission_kg_m3",
]

RANDOM_STATE = 42
