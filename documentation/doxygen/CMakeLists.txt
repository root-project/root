cmake_minimum_required(VERSION 3.16)
project(ROOT_documentation)

#~ set (QHG_LOCATION qhelpgenerator CACHE STRING "Path to qhelpgenerator")
set (QHG_LOCATION /usr/lib/x86_64-linux-gnu/qt4/bin/qhelpgenerator CACHE STRING "Path to qhelpgenerator")
#~ set (DOCU_LOCATION $HOME/rootdoc CACHE STRING "Documentation output directory")
set (DOCU_LOCATION /tmp/rootdoc CACHE STRING "Documentation output directory")

list(APPEND CMAKE_MODULE_PATH "/opt/doxygen")

option(BUILD_DOCUMENTATION "Create and install the HTML based ROOT documentation (requires Doxygen)" ON)

if(BUILD_DOCUMENTATION)

    # ROOT_DIR https://sft.its.cern.ch/jira/si/jira.issueviews:issue-html/ROOT-8062/ROOT-8062.html https://cliutils.gitlab.io/modern-cmake/chapters/packages/ROOT.html
    if(DEFINED ROOTSYS)
    set (ROOT_DIR ${ROOTSYS} CACHE STRING "ROOT build directory")
    else()
    set (ROOT_DIR /opt/root_bld/ CACHE STRING "ROOT build directory")
    endif()
    #~ set (PYTHONPATH ${ROOT_DIR}/lib CACHE STRING "PYTHONPATH")
    find_package(ROOT 6.25 CONFIG REQUIRED)
    set(ENV{PYTHONPATH} ${ROOTSYS}/lib)

    find_package(Doxygen REQUIRED dot)
    #~ find_package(Doxygen 1.9.4 OPTIONAL_COMPONENTS dot)

    find_package (Python COMPONENTS Interpreter)

    if(DOXYGEN_FOUND)

        set(DOXYGEN_EXECUTABLE "/opt/doxygen/build/bin/doxygen")#Temporary

        # TODO: should we remove output directory beforehand to ensure a clean rebuild?

        execute_process(COMMAND ${ROOT_root_CMD}-config --cxx OUTPUT_VARIABLE ROOT_CXX_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE)

        add_custom_target(Preparation

            COMMAND mkdir -p ${DOCU_LOCATION}/macros ${DOCU_LOCATION}/images ${DOCU_LOCATION}/notebooks

            # Copy stuff
            COMMAND scp -p -r ${CMAKE_CURRENT_SOURCE_DIR}/images/* ${DOCU_LOCATION}/images/
            COMMAND tar xfz ${CMAKE_CURRENT_SOURCE_DIR}/mathjax.tar.gz -C ${DOCU_LOCATION}/
            COMMAND scp -p -r ${CMAKE_CURRENT_SOURCE_DIR}/../../js ${DOCU_LOCATION}/

            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/makehtmlfooter.sh > htmlfooter.html

            COMMAND ${ROOT_CXX_COMPILER} -DROOT_COMMAND=\\\"${ROOT_root_CMD}\\\" -DDOXYGEN_OUTPUT_DIRECTORY=\\\"${DOCU_LOCATION}\\\" -DDOXYGEN_SOURCE_DIRECTORY=\\\"${CMAKE_CURRENT_SOURCE_DIR}/../../\\\" -DPYTHON_EXECUTABLE=\\\"${Python_EXECUTABLE}\\\" -DCMAKE_BUILD_DIRECTORY=\\\"${CMAKE_CURRENT_BINARY_DIR}\\\" -o filter ${CMAKE_CURRENT_SOURCE_DIR}/filter.cxx -std=c++14
        )

        set(DOXYGEN_OUTPUT_DIRECTORY       "${DOCU_LOCATION}")
        set(DOXYGEN_PROJECT_NAME           "ROOT")
        set(DOXYGEN_PROJECT_BRIEF          "Reference Guide")
        set(DOXYGEN_PROJECT_LOGO           "./rootlogo.gif")
        set(DOXYGEN_ALWAYS_DETAILED_SEC    YES)
        set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
        #~ set(DOXYGEN_NUM_PROC_THREADS 0)
        set(DOXYGEN_QT_AUTOBRIEF           YES)
        set(DOXYGEN_TAB_SIZE 3)
        set(DOXYGEN_ALIASES "legacy{1}=\\htmlonly<div class=\\\"legacybox\\\"><h2>Legacy Code</h2> \\1 is a legacy interface: it is not recommended to use it in new code. There will be no bug fixes nor new developments.</div>\\endhtmlonly")
        set(DOXYGEN_EXTENSION_MAPPING      h=C++ icc=C++ def=C++ pyzdoc=C++)
        set(DOXYGEN_TOC_INCLUDE_HEADINGS   3)
        set(DOXYGEN_BUILTIN_STL_SUPPORT    YES)
        set(DOXYGEN_LOOKUP_CACHE_SIZE      4)
        set(DOXYGEN_EXTRACT_ALL            YES)
        set(DOXYGEN_EXTRACT_PRIVATE        YES)
        set(DOXYGEN_EXTRACT_STATIC         YES)
        set(DOXYGEN_EXTRACT_ANON_NSPACES   YES)
        set(DOXYGEN_HIDE_IN_BODY_DOCS      YES)
        # set(DOXYGEN_WARN_NO_PARAMDOC       YES) # If you set this to yes, you need to set EXTRACT_ALL to false
        set(DOXYGEN_SORT_BRIEF_DOCS        YES)
        set(DOXYGEN_SORT_MEMBERS_CTORS_1ST YES)
        set(DOXYGEN_GENERATE_TODOLIST      NO)
        set(DOXYGEN_GENERATE_BUGLIST       NO)
        set(DOXYGEN_LAYOUT_FILE            DoxygenLayout.xml)
        #~ set(DOXYGEN_WARN_LOGFILE           ROOT-Doxygen-warnings.txt)
        set(DOXYGEN_FILE_PATTERNS          *.c *.C *.cc *.cpp *.cxx *.def *.dox *.f *.h *.hh *.hpp *.hxx *.icc *.inc *.inl *.js *.m *.md *.mm *.py *.pyzdoc)
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_EXCLUDE_PATTERNS       #*/G__*
                                           #~ */test/*
                                           #~ */src/unuran-*
                                           #~ */libAfterImage/*
                                           #~ */doc/v6*
                                           #~ */doc/v5*
                                           */win32gdk/gdk/*
                                           #~ */bindings/pyroot/*.py
                                           #~ *gl2ps*
                                           #~ *CsgOps*
                                           #~ LinkDef*.h
                                           #~ launcher.py
                                           #~ */io/io/res/*
                                           #~ */src/lexertk.hpp
                                           )
        #~ set(DOXYGEN_EXCLUDE_SYMBOLS        std
                                           #~ cling*)
        set(DOXYGEN_EXAMPLE_PATH           "${DOCU_LOCATION}/macros")
        set(DOXYGEN_IMAGE_PATH             "${DOCU_LOCATION}/images")
        set(DOXYGEN_INPUT_FILTER           "${CMAKE_CURRENT_BINARY_DIR}/filter")
        set(DOXYGEN_SOURCE_BROWSER         YES)
        set(DOXYGEN_STRIP_CODE_COMMENTS    NO)
        set(DOXYGEN_IGNORE_PREFIX          T)
        set(DOXYGEN_HTML_HEADER            "htmlheader.html")
        set(DOXYGEN_HTML_FOOTER            "${CMAKE_CURRENT_BINARY_DIR}/htmlfooter.html")
        set(DOXYGEN_HTML_EXTRA_STYLESHEET  "ROOT.css")
        set(DOXYGEN_HTML_EXTRA_FILES       "./rootlogo_s.gif"
                                           "./notebook.gif")
        set(DOXYGEN_HTML_TIMESTAMP         YES)
        set(DOXYGEN_GENERATE_QHP           YES)
        set(DOXYGEN_QCH_FILE               "ROOT.qch")
        set(DOXYGEN_QHP_NAMESPACE          cern.ch.ROOT)
        set(DOXYGEN_QHG_LOCATION           "${QHG_LOCATION}")
        set(DOXYGEN_QHP_VIRTUAL_FOLDER     "rootdoc")
        set(DOXYGEN_DISABLE_INDEX          YES)
        set(DOXYGEN_GENERATE_TREEVIEW      YES)
        set(DOXYGEN_USE_MATHJAX            YES)
        set(DOXYGEN_MATHJAX_RELPATH        ../mathjax)
        set(DOXYGEN_GENERATE_LATEX         NO)
        #~ set(DOXYGEN_LATEX_CMD_NAME         latex)
        set(DOXYGEN_MACRO_EXPANSION        YES)
        set(DOXYGEN_SKIP_FUNCTION_MACROS   NO)
        #~ set(DOXYGEN_EXPAND_ONLY_PREDEF     YES)
        #~ set(DOXYGEN_SEARCH_INCLUDES        NO)
        #"ClassDef(x,y)=" "ClassImp(x)=" "ClassImpQ(x)=" "templateClassImp(x)=" "NamespaceImp(x)=" "R__SUGGEST_ALTERNATIVE(x)="
        set(DOXYGEN_PREDEFINED             "R__CLING_PTRCHECK" "R__USE_IMT" "__attribute__(x)=" "__declspec(x)=" "__pragma(x)=")
        set(DOXYGEN_INCLUDE_PATH           ../../core/base/inc  # needed for detecting ClassDef and not raising Warning on ::Streamer
                                           ../../bindings/r/inc # needed for bindings/r includes to not raise warnings
                                           ../../interpreter/llvm/src/tools/clang/include # needed to avoid warning of LangOptions.def not found
                                           ../../graf2d/gpadv7/inc # need to avoid getting doxygen confused with macro call on RAttrAxis
                                           ../../tmva/tmva/inc # same for REGISTER_METHOD on tmva
                                           )
        set(DOXYGEN_GENERATE_TAGFILE       "${DOXYGEN_OUTPUT_DIRECTORY}/html/ROOT.tag")
        set(DOXYGEN_HIDE_UNDOC_RELATIONS   NO)
        #~ set(DOXYGEN_HAVE_DOT               YES)
        set(DOXYGEN_HAVE_DOT               NO)
        set(DOXYGEN_GROUP_GRAPHS           NO)
        set(DOXYGEN_DOT_IMAGE_FORMAT       svg)
        set(DOXYGEN_INTERACTIVE_SVG        YES)
        set(DOXYGEN_DOT_GRAPH_MAX_NODES    200)
        #~ set(DOXYGEN_ENABLED_SECTIONS       HIDDEN_SYMBOLS)

#~ if(NOT GRAPHVIZ_FOUND) set(gvizveto...)

        doxygen_add_docs(dox ALL
            ./mainpage.md
            ../../core/base
            ../../core/
            ../../geom/
            ../../graf2d
            ../../graf3d
            ../../gui/
            ../../hist/
            ../../html/
            ../../io/
            ../../main/
            ../../math/
            ../../montecarlo/
            ../../net/
            ../../proof/
            ../../tmva/
            ../../roofit/
            ../../tree/
            ../../sql/
            #~ ../../tutorials/
            ../../bindings/

            #~ ../../core/base/
            #~ ../../core/dictgen/
            #~ ../../core/cont/
            #~ ../../core/foundation/
            #~ ../../core/gui/
            #~ ../../core/macosx/
            #~ ../../core/meta/
            #~ ../../core/metacling/
            #~ ../../core/clingutils/
            #~ ../../core/multiproc/
            #~ ../../core/rint/
            #~ ../../core/testsupport/
            #~ ../../core/thread/
            #~ ../../core/unix/
            #~ ../../core/winnt/
            #~ ../../core/imt/
            #~ ../../core/zip/inc/Compression.h
            #~ ../../geom/
            #~ ../../graf2d/asimage/
            #~ ../../graf2d/cocoa/
            #~ ../../graf2d/fitsio/
            #~ ../../graf2d/gpad/
            #~ ../../graf2d/gpadv7/
            #~ ../../graf2d/graf/
            #~ ../../graf2d/gviz/
            #~ ../../graf2d/postscript/
            #~ ../../graf2d/quartz/
            #~ ../../graf2d/win32gdk/
            #~ ../../graf2d/x11/
            #~ ../../graf2d/x11ttf/
            #~ ../../graf3d/eve/
            #~ ../../graf3d/eve7/
            #~ ../../graf3d/g3d/
            #~ ../../graf3d/gl/
            #~ ../../graf3d/gviz3d/
            #~ ../../gui/
            #~ ../../hist/
            #~ ../../html/
            #~ ../../io/doc/TFile
            #~ ../../io/dcache/
            #~ ../../io/gfal/
            #~ ../../io/io/
            #~ ../../io/sql/
            #~ ../../io/xml/
            #~ ../../io/xmlparser/
            #~ ../../main/src/hadd.cxx
            #~ ../../math/
            #~ ../../montecarlo/
            #~ ../../net/alien/
            #~ ../../net/auth/
            #~ ../../net/davix/
            #~ ../../net/http/
            #~ ../../net/monalisa/
            #~ ../../net/net/
            #~ ../../net/netx/
            #~ ../../net/netxng/
            #~ ../../proof/
            #~ ../../tmva/
            #~ ../../roofit/
            #~ ../../tree/
            #~ ../../sql/
            #~ ../../tutorials/
            #~ ../../bindings/tpython/
            #~ ../../bindings/pyroot/
            #~ ../../bindings/r/

          #ALLOW_DUPLICATE_CUSTOM_TARGETS
          #USE_STAMP_FILE
          COMMENT "Generating doxygen documentation for ${PROJECT_NAME}"
        )
        add_dependencies(dox Preparation)

        #~ # install generated files
        #~ install(
          #~ DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          #~ TYPE DOC
          #~ OPTIONAL # because available only after "make doc"
        #~ )

        #~ set(ENV{DOXYGEN_OUTPUT_DIRECTORY} ${DOXYGEN_OUTPUT_DIRECTORY})
        #~ execute_process(COMMAND listlibs.sh)

    else()
         message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

# echo "        ../../core/clib/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/lzma/                 \\" >> Doxyfile_INPUT
# echo "        ../../core/newdelete/            \\" >> Doxyfile_INPUT
# echo "        ../../core/textinput/            \\" >> Doxyfile_INPUT
# echo "        ../../graf2d/mathtext/           \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/ftgl/               \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/glew/               \\" >> Doxyfile_INPUT
# echo "        ../../graf3d/x3d/                \\" >> Doxyfile_INPUT
# echo "        ../../net/rootd/                 \\" >> Doxyfile_INPUT
# echo "        ../../net/rpdutils/              \\" >> Doxyfile_INPUT


# Add to the list of files to be analyzed the .pyzdoc files created by extract_docstrings.py
# and print_roofit_pyz_doctrings.py
#~ ls $DOXYGEN_PYZDOC_PATH/*.pyzdoc | sed -e "s/$/ \\\\/"  \
#~ >> Doxyfile_INPUT


    #~ add_custom_target(dox ALL
      #~ #DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp
      #~ DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    #~ )
    #~ include(make_documentation)
    #~ PARSE_CMAKE_DOCUMENTATION(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" EXCLUDES "${CMAKE_CURRENT_BINARY_DIR}/*")
    #~ WRITE_CMAKE_DOCUMENTATION( "${CMAKE_CURRENT_SOURCE_DIR}/cmake.dox" SORTED )
endif()
