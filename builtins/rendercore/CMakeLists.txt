# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# RenderCore, the WebGL engine of the REve client, taken as the distribution
# tarball of its release: LICENSE, the bundle REveRenderCore-min.mjs and the
# shaders. The ROOT http server serves it from ui5/eve7/rcore, and the client
# has no web fallback for it, so it is always built.
set(ROOT_RENDERCORE_VERSION 2.3.2)
set(ROOT_RENDERCORE_HASH "a9f786baa49b923f45d09c589ca2e31d927790196968d7ebd1d4fda86e63c4d8")

ExternalProject_Add(RENDERCORE
  DOWNLOAD_DIR      ${ROOT_DOWNLOAD_DIR}
  URL               ${lcgpackages}/RenderCore-${ROOT_RENDERCORE_VERSION}.tar.gz
                    https://github.com/UL-FRI-LGM/RenderCore/releases/download/${ROOT_RENDERCORE_VERSION}/RenderCore-${ROOT_RENDERCORE_VERSION}.tar.gz
  URL_HASH          SHA256=${ROOT_RENDERCORE_HASH}
  SOURCE_DIR        ${CMAKE_BINARY_DIR}/ui5/eve7/rcore
  TIMEOUT           600
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/eve7/rcore/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/eve7/rcore/ COMPONENT libraries FILES_MATCHING PATTERN "*")
