# Copyright (C) 2023, Stephan Lachnit.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# - Locate libbsd library via PkgConfig
# Defines LIBBSD::libbsd target if found via PkgConfig
# Additionally variables from https://cmake.org/cmake/help/latest/module/FindPkgConfig.html#command:pkg_check_modules

find_package(PkgConfig)

if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBBSD QUIET libbsd IMPORTED_TARGET GLOBAL)

    if(LIBBSD_FOUND)
        add_library(LIBBSD::libbsd ALIAS PkgConfig::LIBBSD)
    endif()
endif()
