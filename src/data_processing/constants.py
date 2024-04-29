"""Constants for tfrecord_reader."""

INPUT_FEATURES = [
    "elevation",  # Elevation
    "th",  # Wind Direction
    "vs",  # Wind Speed
    "tmmn",  # Min Temperature
    "tmmx",  # Max Temperature
    "sph",  # Humidity
    "pr",  # Precipitation
    "pdsi",  # Pressure
    "NDVI",  # Normalized Difference Vegetation Index
    "population",  # Population Density
    "erc",  # NFDRS fire danger index energy release component
    "PrevFireMask",  # D-1 Fire Mask
]

OUTPUT_FEATURES = [
    "FireMask",
]

# For each variable, the statistics are ordered in the form: (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    # Elevation in m; 0.1 percentile, 99.9 percentile
    "elevation": (0.0, 3141.0, 657.3003, 649.0147),
    # Wind direction in degrees clockwise from north. Thus min set to 0 and max set to 360.
    "th": (0.0, 360.0, 190.32976, 72.59854),
    # Wind speed in m/s; 0., 99.9 percentile
    "vs": (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    "tmmn": (253.15, 298.94891357421875, 281.08768, 8.982386),
    # -20 degree C, 99.9 percentile
    "tmmx": (253.15, 315.09228515625, 295.17383, 9.815496),
    # Specific humidity. The range of specific humidity is up to 100% so max is 1.
    "sph": (0.0, 1.0, 0.0071658953, 0.0042835088),
    # Precipitation in mm; 0., 99.9 percentile
    "pr": (0.0, 44.53038024902344, 1.7398051, 4.482833),
    # Pressure; 0.1 percentile, 99.9 percentile
    "pdsi": (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    # Normalized Difference vegetation Index
    "NDVI": (-9821.0, 9996.0, 5157.625, 2466.6677),  # min, max
    # Population density; min, 99.9 percentile
    "population": (0.0, 2534.06298828125, 25.531384, 154.72331),
    # NFDRS fire danger index energy release component expressed in BTU's per square foot; 0., 99.9 percentile
    "erc": (0.0, 106.24891662597656, 37.326267, 20.846027),
    # We don't want to normalize the FireMasks; 1 indicates fire, 0 no fire, -1 unlabeled data
    "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
    "FireMask": (-1.0, 1.0, 0.0, 1.0),
}
