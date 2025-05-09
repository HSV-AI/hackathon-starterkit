import sys
import os

# Add project root directory to sys.path so that 'src' package can be imported
test_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(test_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
