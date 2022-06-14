# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(mysql)
  add_subdirectory(mysql)
endif()
if(oracle)
  add_subdirectory(oracle)
endif()
if(odbc)
  add_subdirectory(odbc)
endif()
if(pgsql)
  add_subdirectory(pgsql)
endif()
if(sqlite)
  add_subdirectory(sqlite)
endif()
