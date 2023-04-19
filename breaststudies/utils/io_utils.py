import json
import numpy as np
from pathlib import PurePath

class AdvJsonEncoder(json.JSONEncoder):
    """ Advanced Json Encoder to handle a wider range of data formats"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
        elif isinstance(obj, PurePath):
            return str(obj)
        return json.JSONEncoder.default(self, obj)