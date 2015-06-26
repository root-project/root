############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
include_directories(${FFTW_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__FFTW *.h MODULE FFTW LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(FFTW *.cxx G__FFTW.cxx LIBRARIES Core ${FFTW_LIBRARIES} DEPENDENCIES )
ROOT_INSTALL_HEADERS()

if(builtin_fftw3)
  # ROOT_ADD_BUILTIN_DEPENDENCIES(FFTW FFTW)
  add_custom_command(OUTPUT ${FFTW_LIBRARIES} DEPENDS FFTW3)
  if(NOT TARGET FFTW3LIBS)
    add_custom_target(FFTW3LIBS DEPENDS ${FFTW_LIBRARIES})
  endif()
  add_dependencies(FFTW FFTW3LIBS)
endif()
