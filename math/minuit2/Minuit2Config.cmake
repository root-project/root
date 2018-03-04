include(CMakeFindDependencyMacro)

find_dependency(OpenMP)

if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
# For CMake < 3.9, we need to make the target ourselves
    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS})

        find_package(Threads REQUIRED)
    endif()
endif()

find_package(MPI)

# For supporting CMake < 3.9:
if(MPI_FOUND OR MPI_CXX_FOUND)
    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX IMPORTED INTERFACE)

        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}") 
        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/Minuit2Targets.cmake")
