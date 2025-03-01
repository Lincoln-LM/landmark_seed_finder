"""Global options"""

import importlib.util
import logging
import os

USE_NUMBA = os.environ.get("NPP_USE_NUMBA", "TRUE") == "TRUE"

if USE_NUMBA and importlib.util.find_spec("numba") is None:
    logging.warning(
        "Numba not detected but environment variable 'NPP_USE_NUMBA' is either set to 'TRUE' "
        "or is missing. Defaulting to not using numba."
    )
    USE_NUMBA = False
