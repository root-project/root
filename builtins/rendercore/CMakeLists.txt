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

# The RenderCore assets, one directory per kind. sdf-fonts/ holds the SDF fonts
# REve serves when a font is not in its writable font directory, so that they
# need not be generated, which needs a display.
set(ROOT_RENDERCORE_ASSETS_VERSION 1.0)
set(ROOT_RENDERCORE_ASSETS_HASH "18a98fea0673ed55552c356c48c70a5fcaf6f792e334b59568427478ee151260")

ExternalProject_Add(RENDERCORE_ASSETS
  DOWNLOAD_DIR      ${ROOT_DOWNLOAD_DIR}
  URL               ${lcgpackages}/RenderCore-assets-${ROOT_RENDERCORE_ASSETS_VERSION}.tar.gz
                    https://github.com/UL-FRI-LGM/RenderCore/releases/download/assets-${ROOT_RENDERCORE_ASSETS_VERSION}/RenderCore-assets-${ROOT_RENDERCORE_ASSETS_VERSION}.tar.gz
  URL_HASH          SHA256=${ROOT_RENDERCORE_ASSETS_HASH}
  SOURCE_DIR        ${CMAKE_BINARY_DIR}/ui5/eve7/assets
  TIMEOUT           600
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/ui5/eve7/assets/ DESTINATION ${CMAKE_INSTALL_OPENUI5DIR}/eve7/assets/ COMPONENT libraries FILES_MATCHING PATTERN "*")
