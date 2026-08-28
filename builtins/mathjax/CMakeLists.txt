ExternalProject_Add(
   MATHJAX
   URL ${lcgpackages}/mathjax-3.2.0.tar.gz
   URL_HASH SHA256=6847f1cfdd980afbe31b466474da301a56ab859a67d8035442b052dd4ac2fa03
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   SOURCE_DIR ${CMAKE_BINARY_DIR}/js/mathjax/
   TIMEOUT 600
)
