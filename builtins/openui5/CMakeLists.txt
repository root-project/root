ExternalProject_Add(
    OPENUI5
    URL https://github.com/UI5/openui5/releases/download/1.152.0/openui5-runtime-1.152.0.zip
    URL_HASH SHA256=4882023d4760ed4c20602e36dd183b02a538fc7e8bbfda607b6e25bd1cbbcdb1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
    TIMEOUT 600
)
install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/distribution/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/distribution/ COMPONENT libraries FILES_MATCHING PATTERN "*")
