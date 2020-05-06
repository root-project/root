import sys, os
from doctest import testmod

try:
   import IPython
except:
   raise ImportError("Cannot import IPython")

if len(sys.argv) < 2:
    raise RuntimeError('Please specify tutorial file path as only argument of this script')

filename = sys.argv[1]

# Replicate what doctest does: insert the module's
# dir into sys.path and try to import it
dirname, filename = os.path.split(filename)
sys.path.insert(0, dirname)
m = __import__(filename[:-3])
del sys.path[0]
failures, _ = testmod(m)
if failures:
    sys.exit(1)
else:
    sys.exit(0)
