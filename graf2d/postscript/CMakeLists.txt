# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

set(POSTSCRIPT_HEADERS
  TImageDump.h
  TPDF.h
  TPostScript.h
  TSVG.h
  TTeXDump.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Postscript
  HEADERS
    ${POSTSCRIPT_HEADERS}
  SOURCES
    src/TImageDump.cxx
    src/TPDF.cxx
    src/TPostScript.cxx
    src/TSVG.cxx
    src/TTeXDump.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ZLIB::ZLIB
    mathtext::mathtext
  DEPENDENCIES
    Graf
)

set(RELATIVE_INC_HEADERS ${POSTSCRIPT_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Postscript
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    src/AdobeGlyphList.h
    inc/LinkDef.h
)

check_cxx_compiler_flag(-Wbidi-chars=none GCC_HAS_BIDI_CHARS_FLAG)
if(GCC_HAS_BIDI_CHARS_FLAG)
    set_source_files_properties(src/TPostScript.cxx COMPILE_FLAGS "-Wbidi-chars=none")
endif()
