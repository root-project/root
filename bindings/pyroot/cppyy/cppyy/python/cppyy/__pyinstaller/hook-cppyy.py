# PyInstaller hooks to declare the "data files" (libraries, headers, etc.) of
# cppyy-backend. Placed here rather then in cppyy-backend to guarantee that any
# packaging of top-level "cppyy" picks up the backend as well.
#
# See also setup.cfg.

__all__ = ['datas']


def _backend_files():
    import cppyy_backend, glob, os

    all_files = glob.glob(os.path.join(
        os.path.dirname(cppyy_backend.__file__), '*'))

    def datafile(path):
        return path, os.path.join('cppyy_backend', os.path.basename(path))

    return [datafile(filename) for filename in all_files if os.path.isdir(filename)]

def _api_files():
    import cppyy, os

    paths = str(cppyy.gbl.gInterpreter.GetIncludePath()).split('-I')
    for p in paths:
        if not p: continue

        apipath = os.path.join(p.strip()[1:-1], 'CPyCppyy')
        if os.path.exists(apipath):
            return [(apipath, os.path.join('include', 'CPyCppyy'))]

    return []

datas = _backend_files()+_api_files()
