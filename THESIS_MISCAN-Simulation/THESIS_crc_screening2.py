"""Colorectal cancer screening module."""
import copy
from typing import Any, Callable, Dict, Sequence

import numpy as np

from panmodel import Process, Simulation
from panmodel.utils import PiecewiseLinear

# All preclinical colorectal cancer states
PRECLINICAL_STATES = {
    "pcl4",
    "pcl3b",
    "pcl3a",
    "pcl2b",
    "pcl2a",
    "pcl1b",
    "pcl1a",
}

# Define other two states ***EDITED***
ADVANCED_ADENOMA_STATES = {
    "np_ad10",
    "p_ad10"
}

NONADVANCED_ADENOMA_STATES = {
    "np_ad6_9",
    "npl_ad6_9",
    "p_ad6_9",
    "pl_ad6_9"
}

STATE_PRIORITY = {
    "ad51": 1,
    "ad52": 1,
    "ad53": 1,
    "ad54": 1,
    "np_ad6_9": 2,
    "npl_ad6_9": 2,
    "p_ad6_9": 2,
    "pl_ad6_9": 2,
    "np_ad10": 3,
    "p_ad10": 3,
}

PRIORITY_SIZE = {
    1: "small",
    2: "medium",
    3: "large",
}

SURVEILLANCE_DEFAULTS = {
    "participation": 1.,
    "interval": 0.,
    "surveillance": {},
    "max_age": float("inf"),
    "max_year": float("inf"),
    "reset_participation": False,
    "return": True,
}


def set_surveillance_defaults(surveillance: Dict[str, Dict[str, Any]],
                              defaults: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Method to set surveillance defaults."""
    surveillance_ = copy.deepcopy(surveillance)
    for scheme in surveillance_.values():
        for defaults_ in defaults:
            for k, v in defaults_.items():
                if k not in scheme:
                    scheme[k] = copy.deepcopy(v)
    return surveillance_


def _test_factory_dummy(message: str):
    def _test_dummy(*_args):
        return message

    results = [message]

    return _test_dummy, results


def _test_factory_stool(
        name: str,
        lack_of_specificity: float,
        systematic_lack_of_sensitivity_state: Dict[str, float],
        r_systematic_lack_of_sensitivity_state: str,
        systematic_lack_of_sensitivity_location: PiecewiseLinear.Data,
        r_systematic_lack_of_sensitivity_location: str,
        sensitivity: Dict[str, float]
):
    sys_lack_of_spec = f"crc_screening_systematic_lack_of_specificity_{name}"
    sys_lack_of_sens_location_pwl = PiecewiseLinear(systematic_lack_of_sensitivity_location)

    def _test_stool(simulation: Simulation, r: Callable[[], float]) -> str:
        if simulation.properties(sys_lack_of_spec):
            return "positive"
        if r() < lack_of_specificity:
            return "positive"
        for lesion in simulation.memory["crc_lesions"]:
            if "removed" in lesion:
                continue
            if lesion[r_systematic_lack_of_sensitivity_location] < \
                    sys_lack_of_sens_location_pwl(lesion["localization"]):
                continue
            if lesion[r_systematic_lack_of_sensitivity_state] < \
                    systematic_lack_of_sensitivity_state[lesion["state"]]:
                continue
            if r() < sensitivity[lesion["state"]]:
                return "positive"
        return "negative"

    results = ["negative", "positive"]

    return _test_stool, results


def _test_factory_sigmoidoscopy(
        reach: PiecewiseLinear.Data,
        sensitivity: Dict[str, float]
):
    reach = PiecewiseLinear(reach)

    def _test_sigmoidoscopy(simulation: Simulation, r: Callable[[], float]) -> str:
        cancer_found = False
        lesion_found = False
        reached = reach(r())
        for lesion in simulation.memory["crc_lesions"]:
            if "removed" in lesion:
                continue
            if lesion["localization"] < reached:
                if r() < sensitivity[lesion["state"]]:
                    if lesion["state"] in PRECLINICAL_STATES:
                        cancer_found = True
                    else:
                        lesion_found = True
                        lesion["removed"] = simulation.age

        if cancer_found:
            return "cancer"
        if lesion_found:
            return "adenoma"
        else:
            return "negative"

    results = ["negative", "adenoma", "cancer"]

    return _test_sigmoidoscopy, results


def _test_factory_colonoscopy(
        reach: PiecewiseLinear.Data,
        sensitivity: Dict[str, float],
        fatal: Dict[str, float],
        fatal_size: Dict[str, float]
):
    reach = PiecewiseLinear(reach)

    def _test_colonoscopy(simulation: Simulation, r: Callable[[], float]) -> str:
        cancer_found = False
        reached = reach(r())
        priority = 0
        for lesion in simulation.memory["crc_lesions"]:
            if "removed" in lesion:
                continue
            if lesion["localization"] < reached:
                if r() < sensitivity[lesion["state"]]:
                    if lesion["state"] in PRECLINICAL_STATES:
                        cancer_found = True
                    else:
                        if "crc_screening_found" in simulation.memory:
                            simulation.memory["crc_screening_found"].append(lesion)
                        else:
                            simulation.memory["crc_screening_found"] = [lesion]

                        if STATE_PRIORITY[lesion["state"]] > priority:
                            priority = STATE_PRIORITY[lesion["state"]]

                        if r() < fatal[lesion["state"]]:
                            return "fatal"

                        lesion["removed"] = simulation.age

        if cancer_found:
            return "cancer"
        if priority:
            if r() < fatal_size[PRIORITY_SIZE[priority]]:
                return "fatal"
            return "adenoma"
        return "negative"

    results = ["negative", "adenoma", "cancer", "fatal"]

    return _test_colonoscopy, results


def _test_factory_triage(rules, reset_found: bool):
    def _test_triage(simulation: Simulation, _r: Callable[[], float]) -> str:
        if "crc_screening_found" in simulation.memory:
            lesions = simulation.memory["crc_screening_found"]
            for rule in rules:
                if sum(1 for i in lesions if i["state"] in rule["states"]) >= rule["min"]:
                    message = rule["message"]
                    break
            else:
                message = "negative"
            if reset_found:
                del simulation.memory["crc_screening_found"]
            return message
        return "negative"

    results = ["negative"]
    for rule_ in rules:
        if rule_["message"] not in results:
            results.append(rule_["message"])

    return _test_triage, results


class CRC_screening(Process):
    """This process simulates screening for colorectal cancer. Any
    :class:`~panmodel.core.Universe` with this process should also
    include an instance of the :class:`~panmodel.processes.CRC` process.
    Example data is provided in the
    :mod:`~panmodel.processes.crc_screening.data` module.

    :param tests: The screening tests. Possible types are 'stool',
        'sigmoidoscopy' and 'colonoscopy'.
    :param strategy: The screening interventions.
    :param surveillance: The surveillance programs.
    :param surveillance_defaults: Default values to use in the
        surveillance programs.
    :param name: The name of the process. Defaults to 'crc_screening'.
    """

    name = "crc_screening"

    random_number_generators = [
        "crc_screening",
    ]

    def __init__(self,
                 tests: Dict[str, Dict[str, Any]],
                 strategy: Sequence[Dict[str, Any]],
                 surveillance: Dict[str, Dict[str, Any]],
                 surveillance_defaults: Dict[str, Any] = None,
                 name: str = "crc_screening"):
        self.name = name

        self.tests = tests
        self.strategy = strategy

        surveillance_defaults_ = [SURVEILLANCE_DEFAULTS]
        if surveillance_defaults:
            surveillance_defaults_.append(surveillance_defaults)
        self.surveillance = set_surveillance_defaults(surveillance, surveillance_defaults_)

        self._len_strategy = len(strategy)

        # Determine the age of the first invitation
        if strategy:
            self._first_age = strategy[0]["age"]
        else:
            self._first_age = None

        self._systematic_random_numbers = []
        self._test_fun = {}
        test_results = {}
        for test_name, test in sorted(self.tests.items()):
            if test["type"] == "triage":
                test_fun, results = _test_factory_triage(
                    reset_found=test["reset_found"],
                    rules=test["rules"]
                )
            elif test["type"] == "dummy":
                test_fun, results = _test_factory_dummy(
                    message=test["message"]
                )
            elif test["type"] == "stool":
                r_state = "sys_state_" + test["systematic_random_numbers"][
                    "lack_of_sensitivity_state"
                ]
                if r_state not in self._systematic_random_numbers:
                    self._systematic_random_numbers.append(r_state)
                r_location = "sys_location_" + test["systematic_random_numbers"][
                    "lack_of_sensitivity_location"
                ]
                if r_location not in self._systematic_random_numbers:
                    self._systematic_random_numbers.append(r_location)
                test_fun, results = _test_factory_stool(
                    name=test_name,
                    systematic_lack_of_sensitivity_state=test[
                        "systematic_lack_of_sensitivity_state"
                    ],
                    r_systematic_lack_of_sensitivity_state=r_state,
                    systematic_lack_of_sensitivity_location=test[
                        "systematic_lack_of_sensitivity_location"
                    ],
                    r_systematic_lack_of_sensitivity_location=r_location,
                    lack_of_specificity=test["lack_of_specificity"],
                    sensitivity=test["sensitivity"]
                )
            elif test["type"] == "sigmoidoscopy":
                test_fun, results = _test_factory_sigmoidoscopy(
                    reach=test["reach"],
                    sensitivity=test["sensitivity"],
                )
            elif test["type"] == "colonoscopy":
                test_fun, results = _test_factory_colonoscopy(
                    reach=test["reach"],
                    sensitivity=test["sensitivity"],
                    fatal=test["fatal"],
                    fatal_size=test["fatal_size"]
                )
            else:
                raise ValueError(
                    f"Test type '{test['type']}' does not exist. Possible "
                    f"types are 'dummy', 'stool', 'sigmoidoscopy', "
                    f"'colonoscopy' and 'triage'."
                )
            self._test_fun[test_name] = test_fun
            test_results[test_name] = results

        # Determine the possible tags of logged events
        self.event_tags = [
            "crc_screening_first_invitation",
            "crc_screening_invitation",
        ]

        # Determine the possible tags of logged state ***EDITED***
        self.event_tags += [
            f"state_{cancer_yes_no}"
            for cancer_yes_no in range(1,5)
        ]

        for test_name in set(invitation["test"] for invitation in self.strategy):
            if test_name not in self.tests:
                raise ValueError(
                    f"The test '{test_name}' in 'strategy' is not included "
                    f"in 'tests'."
                )

            self.event_tags += [
                f"crc_screening_{test_name}_{result}"
                for result in test_results[test_name]
            ]

        for surveillance_name, surveillance_ in self.surveillance.items():
            test_name = surveillance_["test"]

            if test_name not in self.tests:
                raise ValueError(
                    f"The test '{test_name}' in surveillance "
                    f"'{surveillance_name}' is not included in 'tests'."
                )

            self.event_tags += [
                f"crc_screening_{surveillance_name}_{result}"
                for result in test_results[test_name]
            ]

        self.callbacks = {
            "__start__": self.schedule_first_invitation,
            "crc_screening_invitation": self.invitation,
            "crc_screening_surveillance": self.surveillance_appointment,
            "crc_onset": self.onset,
            "crc_clinical_now": self.stop_screening,
        }

        # Do not schedule a first invitation if the strategy is empty
        if self._first_age is None:
            del self.callbacks["__start__"]

    def __getstate__(self):
        """Method to retrieve the current state."""
        return {
            "tests": self.tests,
            "strategy": self.strategy,
            "surveillance": self.surveillance,
            "name": self.name,
        }

    def __setstate__(self, state):
        """Method to set the state."""
        self.__init__(**state)

    def properties(self, rng: np.random.Generator, n: int,
                   properties: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Method to define properties for each individual."""
        # Random numbers
        r = {}
        for name, test in sorted(self.tests.items()):
            if test["type"] == "stool":
                x = test["systematic_random_numbers"]["lack_of_specificity"]
                if x not in r:
                    r[x] = rng.random(n)

        # True/False
        systematic_lack_of_specificity = {}
        for name, test in sorted(self.tests.items()):
            if test["type"] == "stool":
                x = test["systematic_random_numbers"]["lack_of_specificity"]
                y = r[x] < test["systematic_lack_of_specificity"]
                z = f"crc_screening_systematic_lack_of_specificity_{name}"
                systematic_lack_of_specificity[z] = y

        return systematic_lack_of_specificity

    @staticmethod
    def stop_screening(simulation: Simulation, *_args, **_kwargs):
        """Method to add flag to signal screening was stopped."""
        simulation.memory["crc_screening_stop"] = True

    def onset(self, simulation: Simulation, *_args, **_kwargs):
        """Method to schedule disease onset."""
        if "crc_clinical" in simulation.memory:
            return

        lesion = simulation.memory["crc_lesions"][-1]
        r = simulation.random["crc_screening"].random

        for tag in self._systematic_random_numbers:
            lesion[tag] = r()

    def schedule_first_invitation(self, simulation: Simulation):
        """Method to schedule the first screening invitation."""
        # Schedule first invitation
        simulation.add_event_age(
            self._first_age, "crc_screening_invitation", invitation_idx=0
        )

    def invitation(self, simulation: Simulation, invitation_idx: int):
        """Method to process participation and outcomes of invited screening round."""
        # Do nothing if screening was stopped
        if "crc_screening_stop" in simulation.memory:
            return

        # Retrieve random number generator
        r = simulation.random["crc_screening"].random

        # Retrieve invitation data
        invitation = self.strategy[invitation_idx]

        # Participation
        if "crc_screening_participator" in simulation.memory:
            # Log invitation
            simulation.log_event("crc_screening_invitation")

            if simulation.memory["crc_screening_participator"]:
                if r() > invitation["participation_participator"]:
                    # Update participator state
                    simulation.memory["crc_screening_participator"] = False

                    # Surveillance
                    if "__not_participated__" in invitation["surveillance"]:
                        self._schedule_surveillance(
                            simulation,
                            invitation["surveillance"]["__not_participated__"]
                        )
                        return

                    # Schedule the next invitation
                    invitation_idx += 1
                    if invitation_idx < len(self.strategy):
                        simulation.add_event_age(
                            self.strategy[invitation_idx]["age"],
                            "crc_screening_invitation",
                            invitation_idx=invitation_idx
                        )

                    return
            else:
                if r() < invitation["participation_non_participator"]:
                    # Update participator state
                    simulation.memory["crc_screening_participator"] = True
                else:
                    # Surveillance
                    if "__not_participated__" in invitation["surveillance"]:
                        self._schedule_surveillance(
                            simulation,
                            invitation["surveillance"]["__not_participated__"]
                        )
                        return

                    # Schedule the next invitation
                    invitation_idx += 1
                    if invitation_idx < len(self.strategy):
                        simulation.add_event_age(
                            self.strategy[invitation_idx]["age"],
                            "crc_screening_invitation",
                            invitation_idx=invitation_idx
                        )

                    return

        else:
            # Log invitation
            simulation.log_event("crc_screening_first_invitation")

            if r() < invitation["participation_first"]:
                # Update participator state
                simulation.memory["crc_screening_participator"] = True
            else:
                # Update participator state
                simulation.memory["crc_screening_participator"] = False

                # Surveillance
                if "__not_participated__" in invitation["surveillance"]:
                    self._schedule_surveillance(
                        simulation,
                        invitation["surveillance"]["__not_participated__"]
                    )
                    return

                # Schedule the next invitation
                invitation_idx += 1
                if invitation_idx < len(self.strategy):
                    simulation.add_event_age(
                        self.strategy[invitation_idx]["age"],
                        "crc_screening_invitation",
                        invitation_idx=invitation_idx
                    )

                return

        # Retrieve test data
        test = invitation["test"]

        # Perform test
        result = self._test_fun[test](simulation, r)

        # Process result
        if result == "cancer":
            simulation.log_event(f"crc_screening_{test}_cancer")
            simulation.add_event(0., "crc_clinical", screen_detected=True)
            return
        elif result == "fatal":
            simulation.add_event(0., f"crc_screening_{test}_fatal", terminate=True)
            return
        else:
            simulation.log_event(f"crc_screening_{test}_{result}")

        # Process current state of cancer ***EDITED***
        cancer_yes_no = 1
        for lesion in simulation.memory["crc_lesions"]:
            if "removed" in lesion:
                continue
            elif lesion['state'] in NONADVANCED_ADENOMA_STATES:
                cancer_yes_no = max(cancer_yes_no, 2)
            elif lesion['state'] in ADVANCED_ADENOMA_STATES:
                cancer_yes_no = max(cancer_yes_no, 3)
            elif lesion['state'] in PRECLINICAL_STATES:
                cancer_yes_no = max(cancer_yes_no, 4)
                break

        # Log current state of cancer ***EDITED***
        simulation.log_event(f"state_{cancer_yes_no}")

        # Surveillance
        if result in invitation["surveillance"]:
            simulation.memory["diagnostic_colo_participation"] = invitation["participation_diag"] # Fix participation to diagnostic colonoscopy ***EDITED***
            self._schedule_surveillance(simulation, invitation["surveillance"][result])
            return

        # Schedule the next invitation
        invitation_idx += 1
        if invitation_idx < self._len_strategy:
            simulation.add_event_age(
                self.strategy[invitation_idx]["age"], "crc_screening_invitation",
                invitation_idx=invitation_idx
            )

    def help(self, simulation: Simulation):
        # Process result
        pass

    def surveillance_appointment(self, simulation: Simulation, surveillance_name: str):
        """Method to process participation and outcomes of surveillance appointment."""
        # Do nothing if screening was stopped
        if "crc_screening_stop" in simulation.memory:
            return

        # Retrieve surveillance scheme
        surveillance = self.surveillance[surveillance_name]

        # Maximum age
        if simulation.age >= surveillance["max_age"]:
            if "__max_age__" in surveillance["surveillance"]:
                self._schedule_surveillance(
                    simulation, surveillance["surveillance"]["__max_age__"]
                )
            elif surveillance["return"] is True or "__max_age__" in surveillance["return"]:
                self._schedule_invitation(simulation)
            return

        # Maximum year
        if simulation.year >= surveillance["max_year"]:
            if surveillance["return"] is True or "__max_year__" in surveillance["return"]:
                self._schedule_invitation(simulation)
            elif "__max_year__" in surveillance["surveillance"]:
                self._schedule_surveillance(
                    simulation, surveillance["surveillance"]["__max_year__"]
                )
            return

        # Retrieve random number generator
        r = simulation.random["crc_screening"].random

        # Participation
        #if r() > surveillance["participation"]:
        #    if "__not_participated__" in surveillance["surveillance"]:
        #        self._schedule_surveillance(
        #            simulation, surveillance["surveillance"]["__not_participated__"]
        #        )
        #    elif surveillance["return"] is True \
        #            or "__not_participated__" in surveillance["return"]:
        #        self._schedule_invitation(simulation)
        #    return

        # Fix participation to diagnostic colonoscopy ***EDITED***
        if surveillance == "diagnostic_colonoscopy":
            if r() > simulation.memory["diagnostic_colo_participation"]:
                if "__not_participated__" in surveillance["surveillance"]:
                    self._schedule_surveillance(
                        simulation, surveillance["surveillance"]["__not_participated__"]
                    )
                elif surveillance["return"] is True \
                        or "__not_participated__" in surveillance["return"]:
                    self._schedule_invitation(simulation)
                return
        else:
            if r() > surveillance["participation"]:
                if "__not_participated__" in surveillance["surveillance"]:
                    self._schedule_surveillance(
                        simulation, surveillance["surveillance"]["__not_participated__"]
                    )
                elif surveillance["return"] is True \
                        or "__not_participated__" in surveillance["return"]:
                    self._schedule_invitation(simulation)
                return

        # Retrieve test data
        test = surveillance["test"]

        # Perform test
        result = self._test_fun[test](simulation, r)

        # Process result
        if result == "cancer":
            simulation.log_event(f"crc_screening_{test}_cancer")
            simulation.add_event(0., "crc_clinical", screen_detected=True)
            return
        elif result == "fatal":
            simulation.add_event(0., f"crc_screening_{test}_fatal", terminate=True)
            return
        else:
            simulation.log_event(f"crc_screening_{test}_{result}")

        # Surveillance
        if result in surveillance["surveillance"]:
            self._schedule_surveillance(
                simulation, surveillance["surveillance"][result]
            )
            return

        # Return to screening
        if surveillance["return"] is True or result in surveillance["return"]:
            self._schedule_invitation(simulation)

    def _schedule_invitation(self, simulation: Simulation):
        try:
            invitation_idx = next(
                idx for idx, invitation in enumerate(self.strategy)
                if invitation["age"] > simulation.age
            )
        except StopIteration:
            return
        simulation.add_event_age(
            self.strategy[invitation_idx]["age"], "crc_screening_invitation",
            invitation_idx=invitation_idx
        )

    def _schedule_surveillance(self, simulation: Simulation, surveillance_name: str):
        surveillance = self.surveillance[surveillance_name]
        if surveillance["reset_participation"]:
            if "crc_screening_participator" in simulation.memory:
                del simulation.memory["crc_screening_participator"]
        simulation.add_event(
            surveillance["interval"], "crc_screening_surveillance",
            surveillance_name=surveillance_name
        )