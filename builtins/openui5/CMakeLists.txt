set(OPENUI5_VERSION "1.152.0")
set(OPENUI5_HASH "4882023d4760ed4c20602e36dd183b02a538fc7e8bbfda607b6e25bd1cbbcdb1")

ExternalProject_Add(
    BUILTIN_OPENUI5
    URL ${lcgpackages}/openui5-runtime-${OPENUI5_VERSION}.zip
        https://github.com/UI5/openui5/releases/download/${OPENUI5_VERSION}/openui5-runtime-${OPENUI5_VERSION}.zip
    URL_HASH SHA256=${OPENUI5_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
    TIMEOUT 600
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/distribution/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/distribution/ COMPONENT libraries ${DIR_PERMISSIONS})
