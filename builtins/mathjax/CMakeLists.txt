ExternalProject_Add(
   MATHJAX
   URL ${CMAKE_SOURCE_DIR}/documentation/doxygen/mathjax.tar.gz # TODO move to LCG
   URL_HASH SHA256=c5e22e60430a65963a87ab4dcc8856b9be5bd434d3b3871f27ee65b584c3c3ea
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   SOURCE_DIR ${CMAKE_BINARY_DIR}/js/mathjax/
   TIMEOUT 600
)
