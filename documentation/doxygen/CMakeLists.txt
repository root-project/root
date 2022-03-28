cmake_minimum_required(VERSION 3.16)
project(ROOT_documentation)

set(CMAKE_VERBOSE_MAKEFILE ON) # set to ON for debugging
set (DOCU_QHG_LOCATION qhelpgenerator CACHE STRING "Path to qhelpgenerator")
#Pass instead -DDOCU_QHG_LOCATION=/usr/lib/x86_64-linux-gnu/qt4/bin/qhelpgenerator if qt4dev-tools or qt5/ if qt5-assistant
#Pass instead -DDOCU_QHG_LOCATION=/opt/Qt/5.15.2/gcc_64/bin/qhelpgenerator e.h. if custom installation
set (DOCU_LOCATION ${HOME}/rootdoc CACHE STRING "Documentation output directory")
#Pass instead -DDOCU_LOCATION=/tmp/rootdoc if temporary installation
set (DOCU_THREADS 0 CACHE STRING "Number of threads for building the documentation (use 0 for maximum threads, N for specific number)")
set (DOCU_LOGFILE "" CACHE STRING "File where to write warnings to (leave empty for printing into console)")


#list(APPEND CMAKE_MODULE_PATH "/opt/doxygen") # Uncomment this if you have a custom installation of Doxygen in /opt/

set (DOCU_INPUT

    ./mainpage.md
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
    ../../tutorials/
    ../../bindings/
    ${DOCU_LOCATION}/pyzdoc/

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

    CACHE STRING "Doxyfile input files or folders")

option(BUILD_DOCUMENTATION "Create and install the HTML based ROOT documentation (requires Doxygen)" ON)

if(BUILD_DOCUMENTATION)

    # ROOT_DIR https://sft.its.cern.ch/jira/si/jira.issueviews:issue-html/ROOT-8062/ROOT-8062.html https://cliutils.gitlab.io/modern-cmake/chapters/packages/ROOT.html
    if(DEFINED ROOTSYS)
    set (ROOT_DIR ${ROOTSYS} CACHE STRING "ROOT build directory")
    else()
    set (ROOT_DIR /opt/root/ CACHE STRING "ROOT build directory")
    endif()

    set (PYTHONPATH ${ROOT_DIR}/lib CACHE STRING "PYTHONPATH")
    find_package(ROOT 6.25 CONFIG REQUIRED)
    set(PYZ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../bindings/pyroot/pythonizations/python/ROOT/_pythonization")

    find_package(Doxygen 1.9.4 REQUIRED dot) # OPTIONAL_COMPONENTS dot

    find_package (Python COMPONENTS Interpreter)

    if(DOXYGEN_FOUND)

        set(DOXYGEN_EXECUTABLE ${DOXYGEN_EXECUTABLE} CACHE STRING "Path to doxygen executable")#In case you want to point somewhere else

        # TODO: should we remove output directory beforehand to ensure a clean rebuild?

        execute_process(COMMAND ${ROOT_root_CMD}-config --cxx OUTPUT_VARIABLE ROOT_CXX_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${ROOT_root_CMD} -l -b -q -e gROOT->GetGitBranch\(\) OUTPUT_VARIABLE ROOT_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
        STRING(REGEX REPLACE "^\n\\(const char \\*\\) " "" ROOT_GIT_VERSION ${ROOT_GIT_VERSION})
        #message(${ROOT_GIT_VERSION})
        execute_process(COMMAND find ../../ -type d -name dictpch -prune -o -type d -name inc OUTPUT_VARIABLE ROOT_INC_FOLDERS OUTPUT_STRIP_TRAILING_WHITESPACE WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        string(REPLACE "\n" ";" ROOT_INC_FOLDERS ${ROOT_INC_FOLDERS})
        list(APPEND ROOT_INC_FOLDERS ../../interpreter/llvm/src/tools/clang/include)#different pattern

        #execute_process(COMMAND ${ROOT_root_CMD}-config --has-roofit OUTPUT_VARIABLE HAS_ROOFIT OUTPUT_STRIP_TRAILING_WHITESPACE)
        #https://stackoverflow.com/questions/34393562/ternary-operator-in-cmakes-generator-expressions

        add_custom_target(Preparation

            COMMAND mkdir -p ${DOCU_LOCATION}/macros ${DOCU_LOCATION}/images ${DOCU_LOCATION}/notebooks ${DOCU_LOCATION}/pyzdoc

            COMMAND scp -p -r ${CMAKE_CURRENT_SOURCE_DIR}/images/* ${DOCU_LOCATION}/images/
            COMMAND tar xfz ${CMAKE_CURRENT_SOURCE_DIR}/mathjax.tar.gz -C ${DOCU_LOCATION}/
            COMMAND scp -p -r ${CMAKE_CURRENT_SOURCE_DIR}/../../js ${DOCU_LOCATION}/

            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/makehtmlfooter.sh > ${CMAKE_CURRENT_BINARY_DIR}/htmlfooter.html

            COMMAND ${ROOT_CXX_COMPILER} -DROOT_COMMAND=\\\"${ROOT_root_CMD}\\\" -DDOXYGEN_OUTPUT_DIRECTORY=\\\"${DOCU_LOCATION}\\\" -DDOXYGEN_SOURCE_DIRECTORY=\\\"${CMAKE_CURRENT_SOURCE_DIR}/../../\\\" -DPYTHON_EXECUTABLE=\\\"${Python_EXECUTABLE}\\\" -DCMAKE_BUILD_DIRECTORY=\\\"${CMAKE_CURRENT_BINARY_DIR}\\\" -o ${CMAKE_CURRENT_BINARY_DIR}/filter ${CMAKE_CURRENT_SOURCE_DIR}/filter.cxx -std=c++14

            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/extract_docstrings.py ${PYZ_DIR} ${DOCU_LOCATION}/pyzdoc
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/print_roofit_pyz_doctrings.py > ${DOCU_LOCATION}/pyzdoc/_roofit.pyzdoc

            # Precompile here to prevent later multithreading issues
#            COMMAND ${ROOT_root_CMD} -l -b -e '.L libs.C+' -q WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#            COMMAND ${ROOT_root_CMD} -l -b -e '.L makeimage.C+' -q WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#            COMMAND ${ROOT_root_CMD} -l -b -e '.L MakeTCanvasJS.C+' -q WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#            COMMAND ${ROOT_root_CMD} -l -b -e '.L MakeRCanvasJS.C+' -q WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

        set(DOXYGEN_OUTPUT_DIRECTORY       "${DOCU_LOCATION}")
        set(DOXYGEN_PROJECT_NAME           "ROOT")
        set(DOXYGEN_PROJECT_NUMBER         "${ROOT_GIT_VERSION}")
        set(DOXYGEN_PROJECT_BRIEF          "Reference Guide")
        set(DOXYGEN_PROJECT_LOGO           "rootlogo.gif")
        set(DOXYGEN_ALWAYS_DETAILED_SEC    YES)
        set(DOXYGEN_JAVADOC_AUTOBRIEF      YES)
        set(DOXYGEN_QT_AUTOBRIEF           YES)
        set(DOXYGEN_TAB_SIZE               3  )
        set(DOXYGEN_ALIASES "legacy{1}=\\htmlonly<div class=\\\"legacybox\\\"><h2>Legacy Code</h2> \\1 is a legacy interface: it is not recommended to use it in new code. There will be no bug fixes nor new developments.</div>\\endhtmlonly")
        set(DOXYGEN_EXTENSION_MAPPING      h=C++ icc=C++ def=C++ pyzdoc=C++)
        set(DOXYGEN_TOC_INCLUDE_HEADINGS   3  )
        set(DOXYGEN_BUILTIN_STL_SUPPORT    YES)
        set(DOXYGEN_LOOKUP_CACHE_SIZE      4  )
        set(DOXYGEN_EXTRACT_ALL            YES)
        set(DOXYGEN_EXTRACT_PRIVATE        YES)
        set(DOXYGEN_EXTRACT_STATIC         YES)
        #set(DOXYGEN_EXTRACT_ANON_NSPACES   YES)
        set(DOXYGEN_HIDE_IN_BODY_DOCS      YES)
        #set(DOXYGEN_WARN_NO_PARAMDOC       YES) # If you set this to yes for finding all missing documentation, you need to set EXTRACT_ALL to false, see https://github.com/doxygen/doxygen/issues/880
        set(DOXYGEN_SORT_BRIEF_DOCS        YES)
        set(DOXYGEN_SORT_MEMBERS_CTORS_1ST YES)
        set(DOXYGEN_GENERATE_TODOLIST      NO)
        set(DOXYGEN_GENERATE_BUGLIST       NO)
        set(DOXYGEN_LAYOUT_FILE            DoxygenLayout.xml)
        set(DOXYGEN_WARN_LOGFILE           ${DOCU_LOGFILE})
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
        #set(DOXYGEN_EXCLUDE_SYMBOLS        std
                                           #cling*)
        set(DOXYGEN_EXAMPLE_PATH           "${DOCU_LOCATION}/macros")
        set(DOXYGEN_IMAGE_PATH             "${DOCU_LOCATION}/images")
        set(DOXYGEN_INPUT_FILTER           "${CMAKE_CURRENT_BINARY_DIR}/filter")
        set(DOXYGEN_SOURCE_BROWSER         YES)
        set(DOXYGEN_STRIP_CODE_COMMENTS    NO)
        set(DOXYGEN_IGNORE_PREFIX          T)
        set(DOXYGEN_HTML_HEADER            "htmlheader.html")
        set(DOXYGEN_HTML_FOOTER            "${CMAKE_CURRENT_BINARY_DIR}/htmlfooter.html")
        set(DOXYGEN_HTML_EXTRA_STYLESHEET  "ROOT.css")
        set(DOXYGEN_HTML_EXTRA_FILES       "rootlogo_s.gif"
                                           "notebook.gif")
        set(DOXYGEN_HTML_TIMESTAMP         YES)
        set(DOXYGEN_GENERATE_QHP           YES)
        set(DOXYGEN_QCH_FILE               "ROOT.qch")
        set(DOXYGEN_QHP_NAMESPACE          cern.ch.ROOT)
        set(DOXYGEN_QHG_LOCATION           "${DOCU_QHG_LOCATION}")
        set(DOXYGEN_QHP_VIRTUAL_FOLDER     "rootdoc")
        set(DOXYGEN_DISABLE_INDEX          YES)
        set(DOXYGEN_GENERATE_TREEVIEW      YES)
        set(DOXYGEN_USE_MATHJAX            YES)
        set(DOXYGEN_MATHJAX_RELPATH        ../mathjax)
        set(DOXYGEN_MATHJAX_VERSION        MathJax_3)
        set(DOXYGEN_GENERATE_LATEX         NO)
        #set(DOXYGEN_LATEX_CMD_NAME         latex)
        set(DOXYGEN_MACRO_EXPANSION        YES)
        set(DOXYGEN_SKIP_FUNCTION_MACROS   NO )
        #set(DOXYGEN_EXPAND_ONLY_PREDEF     YES)
        #set(DOXYGEN_SEARCH_INCLUDES        NO)
        set(DOXYGEN_PREDEFINED             "R__CLING_PTRCHECK" "R__USE_IMT" "__attribute__(x)=" "__declspec(x)=" "__pragma(x)=")
        #"ClassDef(x,y)=" "ClassImp(x)=" "ClassImpQ(x)=" "templateClassImp(x)=" "NamespaceImp(x)=" "R__SUGGEST_ALTERNATIVE(x)="
        set(DOXYGEN_INCLUDE_PATH           "${ROOT_INC_FOLDERS}")
        set(DOXYGEN_STRIP_FROM_PATH        "${DOCU_LOCATION}" "../../")
        set(DOXYGEN_STRIP_FROM_INC_PATH    "${ROOT_INC_FOLDERS}")
        set(DOXYGEN_GENERATE_TAGFILE       "${DOCU_LOCATION}/html/ROOT.tag")
        set(DOXYGEN_HIDE_UNDOC_RELATIONS   NO )
        set(DOXYGEN_HAVE_DOT               YES) # set to NO for getting faster results when debugging
        set(DOXYGEN_DOT_MULTI_TARGETS      YES)
        set(DOXYGEN_DOT_NUM_THREADS        ${DOCU_THREADS})
        set(DOXYGEN_NUM_PROC_THREADS       ${DOCU_THREADS})
        set(DOXYGEN_GROUP_GRAPHS           NO )
        set(DOXYGEN_DOT_IMAGE_FORMAT       svg)
        set(DOXYGEN_INTERACTIVE_SVG        YES)
        set(DOXYGEN_DOT_GRAPH_MAX_NODES    200)
        #set(DOXYGEN_ENABLED_SECTIONS       HIDDEN_SYMBOLS) # uncomment this to also show in the HTML the internal documentation inteded only for ROOT developers

        #TODO: if(NOT GRAPHVIZ_FOUND) set(gvizveto...)

        doxygen_add_docs(dox ALL
          ${DOCU_INPUT}
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

        add_custom_target(Finalization ALL
            COMMAND cmake -E env DOXYGEN_OUTPUT_DIRECTORY="${DOCU_LOCATION}" NTHREADS=${DOCU_THREADS} ROOT_CMD=${ROOT_root_CMD} ./listlibs.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND gzip -f ${DOCU_LOCATION}/html/ROOT.tag
            COMMAND gzip -f ${DOCU_LOCATION}/html/ROOT.qch
        )
        add_dependencies(Finalization dox)

        #TODO cleaning:
        #rm -rf files c1* *.ps *.eps *.png *.jpg *.tex *.svg *.pdf *.root *.xpm *.out *.dat *.dtd *.dot *.txt *.csv *.log *.rs
        #rm -rf tmva* data* result* config* test* Roo* My* Freq*
        #rm -f Doxyfile_INPUT filter htmlfooter.html MDF.C pca.C
        #rm -f greek.gif hsumanim.gif mathsymb.gif parallelcoordtrans.gif
        #rm -f ${DOCU_LOCATION}/notebooks/*.root
        #rm -rf ${DOCU_LOCATION}

    else()
         message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    #~ add_custom_target(dox ALL
      #~ #DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp
      #~ DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    #~ )
    #~ TODO: https://github.com/root-project/root/issues/8999
    #~ include(make_documentation)
    #~ PARSE_CMAKE_DOCUMENTATION(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" EXCLUDES "${CMAKE_CURRENT_BINARY_DIR}/*")
    #~ WRITE_CMAKE_DOCUMENTATION( "${CMAKE_CURRENT_SOURCE_DIR}/cmake.dox" SORTED )
endif()
