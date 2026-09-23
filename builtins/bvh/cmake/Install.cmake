include(GNUInstallDirs)
set(bvh_targets bvh)
if (BVH_BUILD_C_API)
    list(APPEND bvh_targets bvh_c)
endif()

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/src/bvh
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN "c_api" EXCLUDE)
install(
    FILES ${PROJECT_SOURCE_DIR}/src/bvh/v2/c_api/bvh.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bvh/v2/c_api/)
install(
    TARGETS ${bvh_targets}
    EXPORT bvh_exports
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT bvh_exports
    FILE bvh-targets.cmake
    NAMESPACE bvh::v2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bvh/v2/)

include(CMakePackageConfigHelpers)
set(CMAKE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/bvh/v2/)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/bvh-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/bvh-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_CMAKEDIR})

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake"
    COMPATIBILITY AnyNewerVersion)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/bvh-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
