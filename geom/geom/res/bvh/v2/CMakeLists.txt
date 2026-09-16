add_library(bvh INTERFACE)
find_package(Threads)
if (Threads_FOUND)
    # Link with the threading library of the system, which may
    # be required by standard header <thread> on some systems
    target_link_libraries(bvh INTERFACE Threads::Threads)
endif()

target_include_directories(bvh INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)

set_target_properties(bvh PROPERTIES CXX_STANDARD 20)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/bvh/v2
    FILES_MATCHING PATTERN "*.h"
    PATTERN "c_api" EXCLUDE)

if (BVH_BUILD_C_API)
    add_subdirectory(c_api)
endif()
