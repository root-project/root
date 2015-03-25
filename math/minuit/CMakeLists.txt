############################################################################
# CMakeLists.txt file for building ROOT math/minuit package
############################################################################

ROOT_USE_PACKAGE(math/matrix)
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

if (CMAKE_BUILD_TYPE STREQUAL Optimized)
  set(MINUIT_EXTRA_CONSERVATIVE_FLAGS "-fno-fast-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MINUIT_EXTRA_CONSERVATIVE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MINUIT_EXTRA_CONSERVATIVE_FLAGS}")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Minuit DEPENDENCIES Graf Hist Matrix MathCore DICTIONARY_OPTIONS "-writeEmptyRootPCM")
