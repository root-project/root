project(ROOT_documentation)

cmake_minimum_required(VERSION 3.10)
set (DOCU_LOCATION $ENV{HOME}/rootdoc CACHE STRING "Documentation output directory")
set (DOCU_QHG_LOCATION qhelpgenerator-qt5 CACHE STRING "Path to qhelpgenerator")

find_package(Doxygen REQUIRED dot) # OPTIONAL_COMPONENTS dot

set (DOCU_INPUT
./mainpage.md
../../core/base/
../../core/dictgen/
../../core/cont/
../../core/foundation/
../../core/gui/
../../core/macosx/
../../core/meta/
../../core/metacling/
../../core/clingutils/
../../core/multiproc/
../../core/rint/
../../core/testsupport/
../../core/thread/
../../core/unix/
../../core/winnt/
../../core/imt/
../../core/zip/inc/Compression.h
../../geom/
../../graf2d/asimage/
../../graf2d/cocoa/
../../graf2d/fitsio/
../../graf2d/gpad/
../../graf2d/gpadv7/
../../graf2d/graf/
../../graf2d/gviz/
../../graf2d/postscript/
../../graf2d/quartz/
../../graf2d/win32gdk/
../../graf2d/x11/
../../graf2d/x11ttf/
../../graf3d/eve/
../../graf3d/eve7/
../../graf3d/g3d/
../../graf3d/gl/
../../graf3d/gviz3d/
../../gui/
../../hist/
../../io/doc/TFile
../../io/dcache/
../../io/io/
../../io/sql/
../../io/xml/
../../io/xmlparser/
../../main/src/hadd.cxx
../../math/
../../montecarlo/
../../net/doc/
../../net/auth/
../../net/davix/
../../net/http/
../../net/net/
../../net/netxng/
../../net/httpsniff/
../../tmva/
../../roofit/
../../tree/
../../sql/
../../tutorials/
../../bindings/tpython/
../../bindings/pyroot/
../../bindings/pyroot/pythonizations/python/ROOT/_pythonization/__init__.py
../../bindings/r/
${DOCU_LOCATION}/pyzdoc/
  CACHE STRING "Doxyfile input files or folders")

set(DOXYGEN_OUTPUT_DIRECTORY       "${DOCU_LOCATION}")
set(DOXYGEN_PROJECT_NAME           "ROOT")
set(DOXYGEN_PROJECT_BRIEF          "Reference Guide")
set(DOXYGEN_PROJECT_LOGO           "rootlogo.gif")
set(DOXYGEN_PYTHON_DOCSTRING       NO)
set(DOXYGEN_ALWAYS_DETAILED_SEC    YES)
set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
set(DOXYGEN_QT_AUTOBRIEF           YES)
set(DOXYGEN_TAB_SIZE               3)
set(DOXYGEN_ALIASES                "legacy{1}=\\htmlonly<div class=\\\"legacybox\\\"><h2>Legacy Code</h2> \\1 is a legacy interface: there will be no bug fixes nor new developments. Therefore it is not recommended to use it in new long-term production code. But, depending on the context, using \\1 might still be a valid solution.</div>\\endhtmlonly" "legacy{2}=\\htmlonly<div class=\\\"legacybox\\\"><h2>Legacy Code</h2> \\1 is a legacy interface: there will be no bug fixes nor new developments. Therefore it is not recommended to use it in new long-term production code. But, depending on the context, using \\1 might still be a valid solution. \\2</div>\\endhtmlonly"
)
set(DOXYGEN_EXTENSION_MAPPING      h=C++ icc=C++ pyzdoc=C++ cu=C++ cuh=C++)
set(DOXYGEN_TOC_INCLUDE_HEADINGS   3)
set(DOXYGEN_BUILTIN_STL_SUPPORT    YES)
set(DOXYGEN_LOOKUP_CACHE_SIZE      4)
set(DOXYGEN_EXTRACT_ALL            YES)
set(DOXYGEN_EXTRACT_PRIVATE        YES)
set(DOXYGEN_EXTRACT_STATIC         YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS      YES)
set(DOXYGEN_CASE_SENSE_NAMES       YES) # default value is system dependent, so need to set this !
set(DOXYGEN_SORT_BRIEF_DOCS        YES)
set(DOXYGEN_SORT_MEMBERS_CTORS_1ST YES)
set(DOXYGEN_GENERATE_TODOLIST      NO)
set(DOXYGEN_GENERATE_BUGLIST       NO)
set(DOXYGEN_LAYOUT_FILE            DoxygenLayout.xml)
set(DOXYGEN_WARN_LINE_FORMAT       "$file:$line")
set(DOXYGEN_FILE_PATTERNS          *.c *.C *.cc *.cpp *.cxx *.def *.dox *.f *.h *.hh *.hpp *.hxx *.icc *.inc *.inl *.js *.m *.md *.mm *.py *.pyzdoc)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_EXCLUDE_PATTERNS       */G__*
                                   */test/*
                                   */src/unuran-*
                                   */libAfterImage/*
                                   */doc/v6*
                                   */doc/v5*
                                   */win32gdk/gdk/*
                                   */bindings/pyroot/*.py
                                   *gl2ps*
                                   *CsgOps*
                                   LinkDef*.h
                                   launcher.py
                                   */io/io/res/*
                                   */src/lexertk.hpp
                                   *.xml
                                   *.dtd
                                   */graf3d/eve7/glu/*
                                   */gui/cefdisplay/*
                                   */gui/qt6webdisplay/*
                                   */tutorials/visualisation/webgui/qtweb/*
                                   */math/mathcore/src/CDT*
                                   )
set(DOXYGEN_EXCLUDE_SYMBOLS        std
                                   cling*)
set(DOXYGEN_IMAGE_PATH             "${DOCU_LOCATION}/html")
set(DOXYGEN_INPUT_FILTER           "./filter") # ./ is important
set(DOXYGEN_SOURCE_BROWSER         YES)
set(DOXYGEN_STRIP_CODE_COMMENTS    NO)
set(DOXYGEN_STRIP_FROM_PATH        "${DOCU_LOCATION}" "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
set(DOXYGEN_IGNORE_PREFIX          T)
set(DOXYGEN_HTML_HEADER            "htmlheader.html")
set(DOXYGEN_HTML_FOOTER            "htmlfooter.html")
set(DOXYGEN_HTML_EXTRA_STYLESHEET  "ROOT.css")
set(DOXYGEN_HTML_EXTRA_FILES       "./rootlogo_s.gif"
                                   "./notebook.gif")
set(DOXYGEN_HTML_COLORSTYLE        TOGGLE)
set(DOXYGEN_GENERATE_QHP           YES)
set(DOXYGEN_QCH_FILE               "ROOT.qch")
set(DOXYGEN_QHP_NAMESPACE          cern.ch.ROOT)
set(DOXYGEN_QHG_LOCATION           "${DOCU_QHG_LOCATION}")
set(DOXYGEN_QHP_VIRTUAL_FOLDER     "rootdoc")
set(DOXYGEN_DISABLE_INDEX          YES)
set(DOXYGEN_GENERATE_TREEVIEW      YES)
set(DOXYGEN_USE_MATHJAX            YES)
set(DOXYGEN_MATHJAX_RELPATH        ./mathjax)
set(DOXYGEN_MATHJAX_VERSION        MathJax_3)
set(DOXYGEN_GENERATE_LATEX         NO)
set(DOXYGEN_LATEX_CMD_NAME        latex)
set(DOXYGEN_MACRO_EXPANSION        YES)
set(DOXYGEN_PREDEFINED             "R__CLING_PTRCHECK" "R__USE_IMT" "R__SUGGEST_ALTERNATIVE(x)=" "__attribute__(x)=" "__declspec(x)=" "__pragma(x)=")
set(DOXYGEN_SKIP_FUNCTION_MACROS   NO)
set(DOXYGEN_INCLUDE_PATH         ../../core/base/inc
                                 ../../core/meta/inc
                                 ../../bindings/r/inc
                                 ../../interpreter/llvm-project/clang/include
                                 ../../graf2d/graf/inc
                                 ../../graf2d/gpadv7/inc
                                 ../../tmva/tmva/inc
                                 ../../hist/hist/inc
                                 ../../io/io/inc
                                 ../../tree/tree/inc
                                 ../../roofit/roostats/inc
                                 ../../roofit/histfactory/inc
                                 ../../roofit/roofitcore/inc)
set(DOXYGEN_GENERATE_TAGFILE       "${DOCU_LOCATION}/html/ROOT.tag")
set(DOXYGEN_HIDE_UNDOC_RELATIONS   NO)
set(DOXYGEN_DOT_MULTI_TARGETS      YES)
set(DOXYGEN_GROUP_GRAPHS           NO)
set(DOXYGEN_DOT_IMAGE_FORMAT       svg)
set(DOXYGEN_INTERACTIVE_SVG        YES)
set(DOXYGEN_DOT_GRAPH_MAX_NODES    200)
set(DOXYGEN_TIMESTAMP              YES)

doxygen_add_docs(dox ALL
  ${DOCU_INPUT}
  #ALLOW_DUPLICATE_CUSTOM_TARGETS
  #USE_STAMP_FILE
  COMMENT "Generating doxygen documentation for ${PROJECT_NAME}"
)
