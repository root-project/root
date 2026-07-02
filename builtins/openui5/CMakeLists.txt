if(builtin_openui5)
  ExternalProject_Add(
    OPENUI5
    URL ${CMAKE_SOURCE_DIR}/builtins/openui5/openui5.tar.gz
    URL_HASH SHA256=b9e6495d8640302d9cf2fe3c99331311335aaab0f48794565ebd69ecc7449e58
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
    TIMEOUT 600
  )
else()
  ExternalProject_Add(
    OPENUI5
    URL https://github.com/SAP/openui5/releases/download/1.135.0/openui5-runtime-1.135.0.zip
    URL_HASH SHA256=13acdb88a7f3f1d4afef6d1d500b53bccc4b593e7acf442721bb4e3da4e2690b
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
    TIMEOUT 600
  )
endif()
install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/distribution/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/distribution/ COMPONENT libraries FILES_MATCHING PATTERN "*")
