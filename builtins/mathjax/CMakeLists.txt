set(ROOT_MATHJAX_VERSION 3.2.0)
set(ROOT_MATHJAX_HASH "6847f1cfdd980afbe31b466474da301a56ab859a67d8035442b052dd4ac2fa03")

ExternalProject_Add(
   BUILTIN_MATHJAX
   DOWNLOAD_DIR ${ROOT_DOWNLOAD_DIR}
   URL ${lcgpackages}/mathjax-${ROOT_MATHJAX_VERSION}.tar.gz
       https://github.com/mathjax/MathJax/archive/refs/tags/${ROOT_MATHJAX_VERSION}.tar.gz
   URL_HASH SHA256=${ROOT_MATHJAX_HASH}
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   SOURCE_DIR ${CMAKE_BINARY_DIR}/js/mathjax/
   TIMEOUT 600
   DEPENDS BUILTIN_JSROOT
)

# install no longer necessary - will be performed together with jsroot
# install(DIRECTORY ${CMAKE_BINARY_DIR}/js/mathjax/ DESTINATION ${CMAKE_INSTALL_JSROOTDIR}/mathjax ${DIR_PERMISSIONS})
