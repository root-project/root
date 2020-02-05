from JupyROOT.helpers import cppcompleter, utils

if '__IPYTHON__' in __builtins__ and __IPYTHON__:
    cppcompleter.load_ipython_extension(get_ipython())
    utils.iPythonize()
