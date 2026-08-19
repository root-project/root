
find_path(MathJax3_PATH
  NAMES
    "es5/tex-mml-chtml.js" # Used by Doxygen
    "es5/tex-svg.js" # Used by jsroot
  HINTS
    $ENV{MathJax_ROOT}
    ${MathJax_ROOT}
    $ENV{MathJax_DIR}
    ${MathJax_DIR}
  PATHS
    /usr/share/mathjax/
    /usr/share/mathjax3/
    /usr/share/javascript/mathjax
    /opt/local/share/javascript/mathjax
    /usr/lib/node_modules/mathjax
  DOC "Path to MathJax3 installation")

if(MathJax3_PATH)
  set(MathJax_DIR "${MathJax3_PATH}")
  set(MathJax_VERSION 3)
else()
  message(STATUS "Could not find MathJax3 installation. Looking for MathJax2.")

  find_path(MathJax2_PATH
    NAMES MathJax.js
    HINTS
      $ENV{MathJax_ROOT}
      ${MathJax_ROOT}
      $ENV{MathJax_DIR}
      ${MathJax_DIR}
    PATHS
      /usr/share/mathjax/
      /usr/share/mathjax2/
      /usr/share/javascript/mathjax
      /opt/local/share/javascript/mathjax
      /usr/lib/node_modules/mathjax
    DOC "Path to MathJax2 installation")

  if(MathJax2_PATH)
    set(MathJax_DIR "${MathJax2_PATH}")
    set(MathJax_VERSION 2)
  else()
    message(STATUS "Could not find MathJax2 or MathJax3 installation.")
    set(MathJax_DIR "MathJax_DIR-NOTFOUND")
    set(MathJax_VERSION "MathJax_VERSION-NOTFOUND")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MathJax
  REQUIRED_VARS MathJax_DIR MathJax_VERSION
  VERSION_VAR MathJax_VERSION)
mark_as_advanced(MathJax3_PATH MathJax2_PATH)

