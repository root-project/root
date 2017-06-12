ROOT_ADD_GTEST(testTProfile2Poly test_tprofile2poly.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTHn THn.cxx LIBRARIES Hist Matrix MathCore RIO)
if(fftw3)
  ROOT_ADD_GTEST(testTF1 test_tf1.cxx LIBRARIES Hist)
endif()

ROOT_ADD_GTEST(stressHistogramUnit
        stress/StressHistogramGlobal.cxx
        stress/test*.cxx
        stress/add/test*.cxx
        stress/mul/test*.cxx
        stress/merge/test*.cxx
        LIBRARIES Hist RIO)

