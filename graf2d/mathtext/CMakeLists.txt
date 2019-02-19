ROOT_LINKER_LIBRARY(mathtext
  fontembed.cxx
  fontembedps.cxx
  mathrender.cxx
  mathrenderstyle.cxx
  mathrendertoken.cxx
  mathtext.cxx
  mathtextencode.cxx
  mathtextparse.cxx
  mathtextview.cxx
  TYPE
    STATIC
  NOINSTALL
)

set_property(TARGET mathtext PROPERTY POSITION_INDEPENDENT_CODE ON)
