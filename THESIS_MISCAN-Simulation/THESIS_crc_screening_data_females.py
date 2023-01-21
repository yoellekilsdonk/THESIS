"""
Data of the CRC_screening process calibrated on NL observations.
With the exception of the parameter ``fatal`` for ``colonoscopy`` within ``tests``, which is taken
from the US model.

"""


def tests_per_cutoff(cutoff):
    sys_naa_factor = 1.05
    sys_aa_factor = 1.15
    naa_factor = 0.93
    aa_factor = 0.55
    crc_factor = 0.970868

    spec_factor = 0.85

    tests = {
        "15FIT55": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.2102,
                "npl_ad6_9": 0.2102,
                "p_ad6_9": 0.2102,
                "pl_ad6_9": 0.2102,
                "np_ad10": 0.1805,
                "p_ad10": 0.1805,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.046763,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.1072,
                "npl_ad6_9": 0.1072,
                "p_ad6_9": 0.1072,
                "pl_ad6_9": 0.1072,
                "np_ad10": 0.4706,
                "p_ad10": 0.4706,
                "pcl1a": 0.7132,
                "pcl1b": 0.3484,
                "pcl2a": 0.7132,
                "pcl2b": 0.3484,
                "pcl3a": 0.7132,
                "pcl3b": 0.3484,
                "pcl4": 0.7132,
            },
        },
        "15FIT75": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.2304,
                "npl_ad6_9": 0.2304,
                "p_ad6_9": 0.2304,
                "pl_ad6_9": 0.2304,
                "np_ad10": 0.1979,
                "p_ad10": 0.1979,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0762874,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.1265,
                "npl_ad6_9": 0.1265,
                "p_ad6_9": 0.1265,
                "pl_ad6_9": 0.1265,
                "np_ad10": 0.3988,
                "p_ad10": 0.3988,
                "pcl1a": 0.6179,
                "pcl1b": 0.2581,
                "pcl2a": 0.6179,
                "pcl2b": 0.2581,
                "pcl3a": 0.6179,
                "pcl3b": 0.2581,
                "pcl4": 0.6179,
            },
        },
        "47FIT55": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.7221 * sys_naa_factor,
                "npl_ad6_9": 0.7221 * sys_naa_factor,
                "p_ad6_9": 0.7221 * sys_naa_factor,
                "pl_ad6_9": 0.7221 * sys_naa_factor,
                "np_ad10": 0.5581 * sys_aa_factor,
                "p_ad10": 0.5581 * sys_aa_factor,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0168531 * spec_factor,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.1757 * naa_factor,
                "npl_ad6_9": 0.1757 * naa_factor,
                "p_ad6_9": 0.1757 * naa_factor,
                "pl_ad6_9": 0.1757 * naa_factor,
                "np_ad10": 0.7453 * aa_factor,
                "p_ad10": 0.7453 * aa_factor,
                "pcl1a": 0.9952 * crc_factor,
                "pcl1b": 0.2706 * crc_factor,
                "pcl2a": 0.9952 * crc_factor,
                "pcl2b": 0.2706 * crc_factor,
                "pcl3a": 0.9952 * crc_factor,
                "pcl3b": 0.2706 * crc_factor,
                "pcl4": 0.9952 * crc_factor,
            },
        },
        "47FIT60": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.7854 * sys_naa_factor,
                "npl_ad6_9": 0.7854 * sys_naa_factor,
                "p_ad6_9": 0.7854 * sys_naa_factor,
                "pl_ad6_9": 0.7854 * sys_naa_factor,
                "np_ad10": 0.6071 * sys_aa_factor,
                "p_ad10": 0.6071 * sys_aa_factor,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0211138 * spec_factor,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.2276 * naa_factor,
                "npl_ad6_9": 0.2276 * naa_factor,
                "p_ad6_9": 0.2276 * naa_factor,
                "pl_ad6_9": 0.2276 * naa_factor,
                "np_ad10": 0.739 * aa_factor,
                "p_ad10": 0.739 * aa_factor,
                "pcl1a": 0.8942 * crc_factor,
                "pcl1b": 0.2432 * crc_factor,
                "pcl2a": 0.8942 * crc_factor,
                "pcl2b": 0.2432 * crc_factor,
                "pcl3a": 0.8942 * crc_factor,
                "pcl3b": 0.2432 * crc_factor,
                "pcl4": 0.8942 * crc_factor,
            },
        },
        "47FIT65": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.8487 * sys_naa_factor,
                "npl_ad6_9": 0.8487 * sys_naa_factor,
                "p_ad6_9": 0.8487 * sys_naa_factor,
                "pl_ad6_9": 0.8487 * sys_naa_factor,
                "np_ad10": 0.6539 * sys_aa_factor,
                "p_ad10": 0.6539 * sys_aa_factor,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0245894 * spec_factor,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.3794 * naa_factor,
                "npl_ad6_9": 0.3794 * naa_factor,
                "p_ad6_9": 0.3794 * naa_factor,
                "pl_ad6_9": 0.3794 * naa_factor,
                "np_ad10": 0.7821 * aa_factor,
                "p_ad10": 0.7821 * aa_factor,
                "pcl1a": 0.8562 * crc_factor,
                "pcl1b": 0.2328 * crc_factor,
                "pcl2a": 0.8562 * crc_factor,
                "pcl2b": 0.2328 * crc_factor,
                "pcl3a": 0.8562 * crc_factor,
                "pcl3b": 0.2328 * crc_factor,
                "pcl4": 0.8562 * crc_factor,
            },
        },
        "47FIT70": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.9121 * sys_naa_factor,
                "npl_ad6_9": 0.9121 * sys_naa_factor,
                "p_ad6_9": 0.9121 * sys_naa_factor,
                "pl_ad6_9": 0.9121 * sys_naa_factor,
                "np_ad10": 0.6584 * sys_aa_factor,
                "p_ad10": 0.6584 * sys_aa_factor,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0279114 * spec_factor,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.8252 * naa_factor,
                "npl_ad6_9": 0.8252 * naa_factor,
                "p_ad6_9": 0.8252 * naa_factor,
                "pl_ad6_9": 0.8252 * naa_factor,
                "np_ad10": 0.7779 * aa_factor,
                "p_ad10": 0.7779 * aa_factor,
                "pcl1a": 0.8085 * crc_factor,
                "pcl1b": 0.2199 * crc_factor,
                "pcl2a": 0.8085 * crc_factor,
                "pcl2b": 0.2199 * crc_factor,
                "pcl3a": 0.8085 * crc_factor,
                "pcl3b": 0.2199 * crc_factor,
                "pcl4": 0.8085 * crc_factor,
            },
        },
        "47FIT75": {
            "type": "stool",
            "systematic_lack_of_specificity": 0.,
            "systematic_lack_of_sensitivity_location": [
                (0, 0.00),
            ],
            "systematic_lack_of_sensitivity_state": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.9129 * sys_naa_factor,
                "npl_ad6_9": 0.9129 * sys_naa_factor,
                "p_ad6_9": 0.9129 * sys_naa_factor,
                "pl_ad6_9": 0.9129 * sys_naa_factor,
                "np_ad10": 0.6546 * sys_aa_factor,
                "p_ad10": 0.6546 * sys_aa_factor,
                "pcl1a": 0.0,
                "pcl1b": 0.0,
                "pcl2a": 0.0,
                "pcl2b": 0.0,
                "pcl3a": 0.0,
                "pcl3b": 0.0,
                "pcl4": 0.0,
            },
            "systematic_random_numbers": {
                "lack_of_specificity": "FIT",
                "lack_of_sensitivity_location": "FIT",
                "lack_of_sensitivity_state": "FIT",
            },
            "lack_of_specificity": 0.0295105 * spec_factor,
            "sensitivity": {
                "ad51": 0.0,
                "ad52": 0.0,
                "ad53": 0.0,
                "ad54": 0.0,
                "np_ad6_9": 0.99 * naa_factor,
                "npl_ad6_9": 0.99 * naa_factor,
                "p_ad6_9": 0.99 * naa_factor,
                "pl_ad6_9": 0.99 * naa_factor,
                "np_ad10": 0.7816 * aa_factor,
                "p_ad10": 0.7816 * aa_factor,
                "pcl1a": 0.7511 * crc_factor,
                "pcl1b": 0.2042 * crc_factor,
                "pcl2a": 0.7511 * crc_factor,
                "pcl2b": 0.2042 * crc_factor,
                "pcl3a": 0.7511 * crc_factor,
                "pcl3b": 0.2042 * crc_factor,
                "pcl4": 0.7511 * crc_factor,
            },
        },
        "COL": {
            "type": "colonoscopy",
            "reach": [
                (0.00, 1),
                (0.04, 7),
            ],
            "sensitivity": {
                "ad51": 0.75,
                "ad52": 0.75,
                "ad53": 0.75,
                "ad54": 0.75,
                "np_ad6_9": 0.85,
                "npl_ad6_9": 0.85,
                "p_ad6_9": 0.85,
                "pl_ad6_9": 0.85,
                "np_ad10": 0.95,
                "p_ad10": 0.95,
                "pcl1a": 0.95,
                "pcl2a": 0.95,
                "pcl3a": 0.95,
                "pcl1b": 0.95,
                "pcl2b": 0.95,
                "pcl3b": 0.95,
                "pcl4": 0.95,
            },
            "fatal": {
                "ad51": 0.0000142391,
                "ad52": 0.0000142391,
                "ad53": 0.0000142391,
                "ad54": 0.0000142391,
                "np_ad6_9": 0.0000142391,
                "npl_ad6_9": 0.0000142391,
                "p_ad6_9": 0.0000142391,
                "pl_ad6_9": 0.0000142391,
                "np_ad10": 0.0000142391,
                "p_ad10": 0.0000142391,
            },
            "fatal_size": {
                "small": 0.0,
                "medium": 0.0,
                "large": 0.0008,
            },
        },
        "triage": {
            "type": "triage",
            "reset_found": True,
            "rules": [
                {
                    "states": ["np_ad10", "p_ad10"],
                    "min": 1,
                    "message": "short_interval",
                },
                {
                    "states": [
                        "ad51", "ad52", "ad53", "ad54",
                        "np_ad6_9", "npl_ad6_9", "p_ad6_9", "pl_ad6_9",
                    ],
                    "min": 3,
                    "message": "short_interval",
                },
                {
                    "states": [
                        "ad51", "ad52", "ad53", "ad54",
                        "np_ad6_9", "npl_ad6_9", "p_ad6_9", "pl_ad6_9",
                    ],
                    "min": 2,
                    "message": "long_interval",
                },
            ],
        },
        "pause": {
            "type": "dummy",
            "message": "end",
        },
    }

    return tests
