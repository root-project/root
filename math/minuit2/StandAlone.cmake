cmake_minimum_required(VERSION 3.1)

# Check to see if we are inside ROOT and set a smart default
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../build/version_number")
    option(minuit2-inroot "The source directory is inside the ROOT source" ON)
else()
    option(minuit2-inroot "The source directory is inside the ROOT source" OFF)
endif()


option(minuit2-standalone "Copy in the files from the main ROOT files" OFF)
if(minuit2-standalone)
    message(STATUS "Copying in files from ROOT sources to make a redistributable source package. You should clean out the new files with make purge or the appropriate git clean command when you are done.")
endif()
if(NOT minuit2-inroot)
    # Hide this option if not inside ROOT
    mark_as_advanced(minuit2-standalone)
endif()

# This file adds copy_standalone
include(copy_standalone.cmake)

# Copy these files in if needed
copy_standalone(SOURCE ../../build DESTINATION . OUTPUT VERSION_FILE
                FILES version_number)

copy_standalone(SOURCE ../.. DESTINATION .
                FILES LGPL2_1.txt)

copy_standalone(SOURCE ../.. DESTINATION . OUTPUT LICENSE_FILE
                FILES LICENSE)

file(READ ${VERSION_FILE} versionstr)
string(STRIP ${versionstr} versionstr)
string(REGEX REPLACE "([0-9]+[.][0-9]+)[/]([0-9]+)" "\\1.\\2" versionstr ${versionstr})

project(Minuit2
    VERSION ${versionstr}
    LANGUAGES CXX)


# Inherit default from parent project if not main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "Minuit2 ${PROJECT_VERSION} standalone")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# Common features to all packages (Math and Minuit2)
# If using this with add_subdirectory, the Minuit2
# namespace does not get automatically prepended,
# so including an alias for that.
add_library(Common INTERFACE)
add_library(Minuit2::Common ALIAS Common)

# OpenMP support
if(minuit2-omp)
    target_link_libraries(Common INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "Building Minuit2 with OpenMP support")
endif()

# MPI support
# Uses the old CXX bindings (deprecated), probably do not activate
if(minuit2-mpi)
    message(STATUS "Building Minuit2 with MPI support")
    message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
    target_compile_definitions(Common INTERFACE "-DMPIPROC")
    target_link_libraries(Common INTERFACE MPI::MPI_CXX)
endif()


add_subdirectory(src)

# Exporting targets to allow find_package(Minuit2) to work properly

# Make a config file to make this usable as a CMake Package
# Start by adding the version in a CMake understandable way
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    Minuit2ConfigVersion.cmake
    VERSION ${Minuit2_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

# Now, install the Interface targets
install(TARGETS Common
        EXPORT Minuit2Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

install(EXPORT Minuit2Targets
        FILE Minuit2Targets.cmake
        NAMESPACE Minuit2::
        DESTINATION lib/cmake/Minuit2
        )

# Adding the Minuit2Config file
configure_file(Minuit2Config.cmake.in Minuit2Config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Minuit2Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/Minuit2ConfigVersion.cmake"
        DESTINATION lib/cmake/Minuit2
        )

# Allow build directory to work for CMake import
export(TARGETS Common Math Minuit2 NAMESPACE Minuit2:: FILE Minuit2Targets.cmake)
export(PACKAGE Minuit2)

# Only add tests if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(test/MnSim)
    add_subdirectory(test/MnTutorial)
endif()

# Add purge target
get_property(COPY_STANDALONE_LISTING GLOBAL PROPERTY COPY_STANDALONE_LISTING)
add_custom_target(purge
    COMMAND ${CMAKE_COMMAND} -E remove ${COPY_STANDALONE_LISTING})


# Packaging support
set(CPACK_PACKAGE_VENDOR "root.cern.ch")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Minuit2 standalone fitting tool")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${LICENSE_FILE}")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
# CPack collects *everything* except what's listed here.
set(CPACK_SOURCE_IGNORE_FILES
    /test/CMakeLists.txt
    /test/Makefile
    /test/testMinimizer.cxx
    /test/testNdimFit.cxx
    /test/testUnbinGausFit.cxx
    /test/testUserFunc.cxx
    /Module.mk
    /.git
    /dist
    /.*build.*
    /\\\\.DS_Store
    /.*\\\\.egg-info
    /var
    /Pipfile.*$
)
include(CPack)

