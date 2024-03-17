# Try to enable everything, without failing on missing deps
set(all ON CACHE BOOL "")
set(fail-on-missing OFF CACHE BOOL "")

# Try to enable testing options
set(clingtest ON CACHE BOOL "")
set(roottest ON CACHE BOOL "")
set(testing ON CACHE BOOL "")

# Disable options likely to cause configuration failures
foreach(option clad cuda r roofit_multiprocess tmva-gpu vc veccore vecgeom)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()

# Optionally include any local customizations
include(${CMAKE_CURRENT_LIST_DIR}/custom.cmake OPTIONAL)
