############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

include_directories(${FFTW_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(FFTW
                              LIBRARIES ${FFTW_LIBRARIES}
                              DEPENDENCIES Core
                              BUILTINS FFTW3)
