import os
cbl = os.environ.get('CPPYY_BACKEND_LIBRARY')
if cbl is not None and not os.path.isfile(cbl):
    # CPPYY_BACKEND_LIBRARY does not point to an existing file
    # Let cppyy find it on its own
    del os.environ['CPPYY_BACKEND_LIBRARY']

import cppyy
cppyy.gbl.std.map('string','string')()
