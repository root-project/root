# UHI tests
ROOT_ADD_PYUNITTEST(uhi_plotting uhi_plotting.py GENERIC PYTHON_DEPS uhi numpy)
ROOT_ADD_PYUNITTEST(uhi_indexing uhi_indexing.py GENERIC PYTHON_DEPS uhi numpy)
ROOT_ADD_PYUNITTEST(uhi_serialization_json uhi_serialization_json.py GENERIC PYTHON_DEPS uhi numpy)
ROOT_ADD_PYUNITTEST(uhi_test_suite uhi_test_suite.py GENERIC PYTHON_DEPS uhi numpy)
