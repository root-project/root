# Template related stuff

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

REFLEX_ADD_SINGLE_TEST("TemplateDefaultArg"
                       HEADERS TemplateDefaultArg.hpp
                       SELECTION TemplateDefaultArg.xml
                       TEST TemplateDefaultArgTests.cxx)
