"""Initialises paths to data loading scripts."""

import os
import sys

pp_path = os.path.expanduser('~/repos/pose-prediction/keras')
paths = {pp_path}
for path in paths:
    assert os.path.isdir(path), 'code at %s must exist' % path
    if path not in sys.path:
        sys.path.append(path)
