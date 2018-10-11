import py, pytest, os, sys
from support import setup_make


setup_make("functioncallsDict.so")

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("functioncallsDict.so"))

import cppyy
cppyy.load_reflection_info(test_dct)


#- group: empty --------------------------------------------------------------
def py_empty_call():
    pass

group = 'empty'
@pytest.mark.benchmark(group=group, warmup=True)
def test_py_empty_call(benchmark):
    benchmark(py_empty_call)

@pytest.mark.benchmark(group=group, warmup=True)
def test_empty_call(benchmark):
    benchmark(cppyy.gbl.empty_call)


#- group: builtin-args -------------------------------------------------------
def py_take_a_value(val):
    pass

group = 'builtin-args'
@pytest.mark.benchmark(group=group, warmup=True)
def test_py_take_a_value(benchmark):
    benchmark(py_take_a_value, 1)

@pytest.mark.benchmark(group=group, warmup=True)
def test_take_an_int(benchmark):
    benchmark(cppyy.gbl.take_an_int, 1)
