# Find MathJax v4, if not there v3, and if not there, fall back to finding v2
# MathJax_ROOT or MathJax_DIR can be set as hints for the search
# The script first searchs for es5/tex-mml-chtml.js and es5/tex-svg.js files
# which are required by ROOT doxygen and jsroot/webgui respectively, for MathJax v3.
# If not found, it then searches for MathJax.js for MathJax v2
# The following variables are then set
#
# ``MathJax_FOUND``
#   True if MathJax v4 or v3 or v2 were found.
# ``MathJax_DIR``
#   The base directory of the MathJax v4 or v3 or v2 installations
#   or MathJax_DIR-NOTFOUND if nothing found
# ``MathJax_VERSION``
#   4, 3 or 2, or MathJax_VERSION-NOTFOUND if nothing found

find_path(MathJax4_PATH
  NAMES
    "node-main.js"
    "tex-mml-chtml.js" # Used by Doxygen
    "tex-svg.js" # Used by jsroot
  HINTS
    $ENV{MathJax_ROOT}
    ${MathJax_ROOT}
    $ENV{MathJax_DIR}
    ${MathJax_DIR}
  PATHS
    /usr/share/mathjax/
    /usr/share/mathjax4/
    /usr/share/javascript/mathjax
    /opt/local/share/javascript/mathjax
    /usr/lib/node_modules/mathjax
  DOC "Path to MathJax4 installation")

if(MathJax4_PATH)
  set(MathJax_DIR "${MathJax4_PATH}")
  set(MathJax_VERSION 4)
else()
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
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MathJax
  REQUIRED_VARS MathJax_DIR MathJax_VERSION
  VERSION_VAR MathJax_VERSION)
mark_as_advanced(MathJax4_PATH MathJax3_PATH MathJax2_PATH)

