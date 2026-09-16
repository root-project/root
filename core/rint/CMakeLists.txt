# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Rint
  STAGE1
  HEADERS
    TRint.h
    TTabCom.h
  SOURCES
    src/TRint.cxx
    src/TTabCom.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
)
target_include_directories(Core PRIVATE inc)

ROOT_ADD_TEST_SUBDIRECTORY(test)

set_target_properties(Rint PROPERTIES
    OUTPUT_NAME "ROOTRint"
    EXPORT_NAME "Rint"
)
if (symlink_libs)
  install(CODE "
    set(LIB_DIR \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
    set(NEW_NAME \"\${CMAKE_SHARED_LIBRARY_PREFIX}ROOTRint\${CMAKE_SHARED_LIBRARY_SUFFIX}\")
    set(OLD_NAME \"\${CMAKE_SHARED_LIBRARY_PREFIX}Rint\${CMAKE_SHARED_LIBRARY_SUFFIX}\")
    message(STATUS \"Creating symlink: \${OLD_NAME} -> \${NEW_NAME}\")
    execute_process(
      COMMAND \${CMAKE_COMMAND} -E create_symlink \${NEW_NAME} \${OLD_NAME}
      WORKING_DIRECTORY \${LIB_DIR}
    )
  ")
endif()
