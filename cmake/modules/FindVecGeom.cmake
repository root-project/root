# Locate the VecGeom library. 
#
# This file is meant to be copied into projects that want to use VecGeom. It will
# search for VecGeomConfig.cmake, which ships with VecGeom and which will provide 
# up-to-date buildsystem changes. 
#
# This module defines the following variables:
# VECGEOM_FOUND
# VECGEOM_INCLUDE_DIR
# VECGEOM_LIBRARIES
# VECGEOM_DEFINITIONS
# VECGEOM_VERSION_MAJOR # not yet
# VECGEOM_VERSION_MINOR # not yet
# VECGEOM_VERSION_PATCH # not yet
# VECGEOM_VERSION # not yet
# VECGEOM_VERSION_STRING # not yet
# VECGEOM_INSTALL_DIR
# VECGEOM_LIB_DIR
# VECGEOM_CMAKE_MODULES_DIR
#

find_package(VecGeom ${VecGeom_FIND_VERSION} NO_MODULE PATHS $ENV{HOME} $ENV{VECGEOMROOT} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecGeom CONFIG_MODE)
