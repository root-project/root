# C++
ROOT_ADD_GTEST(vecops_rvec vecops_rvec.cxx LIBRARIES ROOTVecOps RIO Tree)
ROOT_ADD_GTEST(vecops_radoptallocator vecops_radoptallocator.cxx LIBRARIES Core)

# Python
ROOT_ADD_PYUNITTEST(vecops_rvec vecops_rvec.py)
