HIST_ADD_GTEST(hist_axes hist_axes.cxx)
HIST_ADD_GTEST(hist_engine hist_engine.cxx)
HIST_ADD_GTEST(hist_index hist_index.cxx)
HIST_ADD_GTEST(hist_regular hist_regular.cxx)
HIST_ADD_GTEST(hist_variable hist_variable.cxx)

if(NOT DEFINED histv7_standalone)
  ROOT_ADD_GTEST(hist_io hist_io.cxx LIBRARIES RIO ROOTHist)
endif()
