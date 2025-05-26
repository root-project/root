message(STATUS "Building AdaptiveCpp for SYCL support.")

include(FetchContent)

if(NOT DEFINED ADAPTIVE_CPP_SOURCE_DIR)
  FetchContent_Declare(
    AdaptiveCpp
    GIT_REPOSITORY https://github.com/root-project/AdaptiveCpp.git
    GIT_TAG ROOT-acpp-v25.02.0-20250615-01)
  FetchContent_GetProperties(AdaptiveCpp)
  if(NOT AdaptiveCpp_POPULATED)
    FetchContent_Populate(AdaptiveCpp)
  endif()
  set(ADAPTIVE_CPP_SOURCE_DIR ${adaptivecpp_SOURCE_DIR})
  message(STATUS "Fetched AdaptiveCpp source to: ${ADAPTIVE_CPP_SOURCE_DIR}")
else()
  message(
    STATUS "ADAPTIVE_CPP_SOURCE_DIR already defined: ${ADAPTIVE_CPP_SOURCE_DIR}"
  )
endif()

set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm)
set(CLANG_EXECUTABLE_PATH ${LLVM_BINARY_DIR}/bin/clang${CMAKE_EXECUTABLE_SUFFIX})

set(ACPP_CLANG
    ${CLANG_EXECUTABLE_PATH}
    CACHE STRING "Clang compiler executable used for compilation." FORCE)

set(ADAPTIVE_CPP_BINARY_DIR ${CMAKE_BINARY_DIR})
message(STATUS "AdaptiveCpp will be built in: ${ADAPTIVE_CPP_BINARY_DIR}")

set(ADAPTIVECPP_INSTALL_CMAKE_DIR
    lib/cmake/AdaptiveCpp
    CACHE PATH "Install path for CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is the
# future install root
set(ADAPTIVECPP_INSTALL_COMPILER_DIR bin)
set(ACPP_CONFIG_FILE_INSTALL_DIR etc/AdaptiveCpp)
set(ADAPTIVECPP_INSTALL_LAUNCHER_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})
set(ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})

install(FILES ${ADAPTIVE_CPP_BINARY_DIR}/lib/libacpp-rt.so DESTINATION lib)
install(DIRECTORY ${ADAPTIVE_CPP_BINARY_DIR}/lib/hipSYCL/bitcode/
        DESTINATION lib/hipSYCL/bitcode)
install(DIRECTORY ${ADAPTIVE_CPP_BINARY_DIR}/include/AdaptiveCpp/
        DESTINATION include/AdaptiveCpp)

file(
  COPY ${ADAPTIVE_CPP_SOURCE_DIR}/cmake/syclcc-launcher
  DESTINATION ${ADAPTIVE_CPP_BINARY_DIR}/${ADAPTIVECPP_INSTALL_LAUNCHER_DIR})
file(
  COPY ${ADAPTIVE_CPP_SOURCE_DIR}/cmake/syclcc-launch.rule.in
  DESTINATION ${ADAPTIVE_CPP_BINARY_DIR}/${ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR})

list(APPEND CMAKE_PREFIX_PATH ${ADAPTIVE_CPP_BINARY_DIR}/${ADAPTIVECPP_INSTALL_CMAKE_DIR})
message(STATUS "Added ${ADAPTIVE_CPP_BINARY_DIR}/${ADAPTIVECPP_INSTALL_CMAKE_DIR} to CMAKE_PREFIX_PATH.")

install(PROGRAMS ${ADAPTIVE_CPP_BINARY_DIR}/bin/acpp
        DESTINATION ${ADAPTIVECPP_INSTALL_COMPILER_DIR})
install(FILES ${ADAPTIVE_CPP_SOURCE_DIR}/cmake/syclcc-launcher
        DESTINATION ${ADAPTIVECPP_INSTALL_LAUNCHER_DIR})
install(FILES ${ADAPTIVE_CPP_SOURCE_DIR}/cmake/syclcc-launch.rule.in
        DESTINATION ${ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR})

file(GLOB CLANG_EXECUTABLES "${LLVM_BINARY_DIR}/bin/clang*")
install(PROGRAMS ${CLANG_EXECUTABLES}
        DESTINATION ${ADAPTIVECPP_INSTALL_COMPILER_DIR})

file(GLOB CONFIG_FILES "${ADAPTIVE_CPP_BINARY_DIR}/etc/AdaptiveCpp/*")
install(FILES ${CONFIG_FILES} DESTINATION ${ACPP_CONFIG_FILE_INSTALL_DIR})

# Create imported target AdaptiveCpp::acpp-common
add_library(AdaptiveCpp::acpp-common STATIC IMPORTED)

set_target_properties(AdaptiveCpp::acpp-common PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ADAPTIVE_CPP_BINARY_DIR}/include;${ADAPTIVE_CPP_BINARY_DIR}/include/AdaptiveCpp"
  INTERFACE_LINK_LIBRARIES "-Wl,-Bsymbolic-functions;\$<LINK_ONLY:dl>"
)

# Create imported target AdaptiveCpp::acpp-rt
add_library(AdaptiveCpp::acpp-rt SHARED IMPORTED)

set_target_properties(AdaptiveCpp::acpp-rt PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ADAPTIVE_CPP_BINARY_DIR}/include;${ADAPTIVE_CPP_BINARY_DIR}/include/AdaptiveCpp"
  INTERFACE_LINK_LIBRARIES "AdaptiveCpp::acpp-common"
)

# Import target "AdaptiveCpp::acpp-common" for configuration "Release"
set_property(TARGET AdaptiveCpp::acpp-common APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AdaptiveCpp::acpp-common PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${ADAPTIVE_CPP_BINARY_DIR}/lib/libacpp-common.a"
  )

# Import target "AdaptiveCpp::acpp-rt" for configuration "Release"
set_property(TARGET AdaptiveCpp::acpp-rt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AdaptiveCpp::acpp-rt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${ADAPTIVE_CPP_BINARY_DIR}/lib/libacpp-rt.so"
  IMPORTED_SONAME_RELEASE "libacpp-rt.so"
  )

# Make a config file to make this usable as a CMake Package
# Start by adding the version in a CMake understandable way
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${ADAPTIVE_CPP_SOURCE_DIR}/cmake/adaptivecpp-config.cmake.in
    ${ADAPTIVE_CPP_BINARY_DIR}/lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake
    INSTALL_DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
    PATH_VARS
    ADAPTIVECPP_INSTALL_COMPILER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR
)
