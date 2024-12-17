# monkey patch to be able to select a specific backend based on PyPy's version,
# which is not possible in the pyproject.toml file as there is currently no
# marker for it (this may change, after which this file can be removed)

try:
  # _BACKEND is the primary, __legacy__ the backwards compatible backend
    from setuptools.build_meta import _BACKEND
    main = _BACKEND
except (NameError, ImportError):
  # fallback as the name __legacy__ is actually documented (and part of __all__)
    main = __legacy__

# the following ensures proper build/installation order, after which the normal
# install through setup.py picks up their wheels from the cache (TODO: note the
# duplication here with setup.py; find a better way)
_get_requires_for_build_wheel = main.get_requires_for_build_wheel
def get_requires_for_build_wheel(*args, **kwds):
    try:
        import __pypy__, sys
        version = sys.pypy_version_info
        requirements = ['cppyy-cling==6.32.8']
        if version[0] == 5:
            if version[1] <= 9:
                requirements = ['cppyy-cling<6.12']
            elif version[1] <= 10:
                requirements = ['cppyy-cling<=6.15']
        elif version[0] == 6:
            if version[1] <= 0:
                requirements = ['cppyy-cling<=6.15']
        elif version[0] == 7:
            if version[1] <= 3 and version[2] <= 3:
                requirements = ['cppyy-cling<=6.18.2.3']
    except ImportError:
        # CPython
        requirements = ['cppyy-backend==1.15.3', 'cppyy-cling==6.32.8']

    return requirements + _get_requires_for_build_wheel(*args, **kwds)

main.get_requires_for_build_wheel = get_requires_for_build_wheel
