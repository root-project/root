# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

# Tests using google test
ROOT_ADD_GTEST(TestRandomGenerator
               TestRandomGenerator.cxx
               LIBRARIES TMVA)
ROOT_ADD_GTEST(TestOptimizeConfigParameters
               TestOptimizeConfigParameters.cxx
               LIBRARIES TMVA)

if(dataframe)
    # RTensor
    ROOT_ADD_GTEST(rtensor rtensor.cxx LIBRARIES ROOTVecOps TMVA)
    ROOT_ADD_GTEST(rtensor-iterator rtensor_iterator.cxx LIBRARIES ROOTVecOps TMVA)
    ROOT_ADD_GTEST(rtensor-utils rtensor_utils.cxx LIBRARIES ROOTVecOps TMVA ROOTDataFrame)
    # RStandardScaler
    ROOT_ADD_GTEST(rstandardscaler rstandardscaler.cxx LIBRARIES ROOTVecOps TMVA ROOTDataFrame)
    # RReader
    ROOT_ADD_GTEST(rreader rreader.cxx LIBRARIES ROOTVecOps TMVA ROOTDataFrame)
    # Tree inference system and user interface
    # Commented out right now because RBDT doesn't provide low-level interfaces
    # since the sync with FastForest. Only the construction from XGBoost models
    # is supported. If this feature becomes necessary to implement, one can
    # resurrect these tests.
    # ROOT_ADD_GTEST(branchlessForest branchlessForest.cxx LIBRARIES TMVA)
    # ROOT_ADD_GTEST(rbdt rbdt.cxx LIBRARIES ROOTVecOps TMVAUtils)
endif()

if(dataframe)
  ROOT_FIND_PYTHON_MODULE(xgboost QUIET)
  if (ROOT_XGBOOST_FOUND)
    ROOT_ADD_PYUNITTEST(rbdt_xgboost rbdt_xgboost.py)
  endif()
endif()

#--stressTMVA--------------------------------------------------------------------------------------

ROOT_EXECUTABLE(stressTMVA stressTMVA.cxx LIBRARIES TMVA)

ROOT_ADD_TEST(test-stresstmva COMMAND stressTMVA -b LABELS longtest TIMEOUT 1800)

if(tmva-cpu)
  target_compile_definitions(stressTMVA PRIVATE DNNCPU)
endif()

if(tmva-gpu)
  target_compile_definitions(stressTMVA PRIVATE DNNCUDA)
  set_property(TEST test-stresstmva PROPERTY RESOURCE_LOCK GPU)
endif()
