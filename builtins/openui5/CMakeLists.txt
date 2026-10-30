set(OPENUI5_VERSION "1.152.0")

if(DEFINED OPENUI5_SOURCE_DIR)
  # Use an already unpacked openui5 runtime distribution instead of
  # downloading it, analogous to CLAD_SOURCE_DIR for clad. This allows to
  # build ROOT without network access.
  if(NOT EXISTS "${OPENUI5_SOURCE_DIR}/resources/sap-ui-core.js")
    message(FATAL_ERROR "OPENUI5_SOURCE_DIR=${OPENUI5_SOURCE_DIR} does not contain an openui5 runtime distribution (resources/sap-ui-core.js not found)")
  endif()
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ui5)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
     ${OPENUI5_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ui5/distribution)
  set(_openui5_distribution_dir ${OPENUI5_SOURCE_DIR})
else()
  ExternalProject_Add(
      OPENUI5
      URL ${lcgpackages}/openui5-runtime-${OPENUI5_VERSION}.zip
      URL_HASH SHA256=4882023d4760ed4c20602e36dd183b02a538fc7e8bbfda607b6e25bd1cbbcdb1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      SOURCE_DIR ${CMAKE_BINARY_DIR}/ui5/distribution
      TIMEOUT 600
  )
  set(_openui5_distribution_dir ${CMAKE_BINARY_DIR}/ui5/distribution)
endif()

install(DIRECTORY ${_openui5_distribution_dir}/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/distribution/ COMPONENT libraries FILES_MATCHING PATTERN "*")
