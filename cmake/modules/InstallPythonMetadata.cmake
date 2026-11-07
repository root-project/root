# Installs Python METADATA and INSTALLER files for compatibility with importlib.metadata
# The presence of INSTALLER (along with intentionally neglecting RECORD) prevents 
# package managers from uninstalling or otherwise touching the ROOT import package if 
# it wasn't installed via a wheel.
# See: https://packaging.python.org/en/latest/specifications/recording-installed-packages/

# scikit-build-core handles metadata so only do this for non-wheel builds to avoid conflict
if(NOT _wheel_build AND pyroot AND Python3_FOUND)
  if(MSVC)
    set(DISTINFO_INSTALL_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  else()
    set(DISTINFO_INSTALL_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endif()

  configure_file("${CMAKE_SOURCE_DIR}/config/METADATA.in" "${DISTINFO_INSTALL_DIR}/root-${ROOT_VERSION}.dist-info/METADATA" @ONLY NEWLINE_STYLE UNIX)
  file(WRITE "${DISTINFO_INSTALL_DIR}/root-${ROOT_VERSION}.dist-info/INSTALLER" "CMake")
  install(DIRECTORY "${DISTINFO_INSTALL_DIR}/root-${ROOT_VERSION}.dist-info" DESTINATION "${CMAKE_INSTALL_PYTHONDIR}")
endif()
