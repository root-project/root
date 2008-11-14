#####################  dictionary  ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

REFLEX_GENERATE_DICTIONARIES(dictfile
                             HEADERS Reflex.h
                             SELECTION selection.xml
                             OPTIONS --quiet)
ADD_LIBRARY(test_ReflexRflx MODULE ${dictfile})
TARGET_LINK_LIBRARIES(test_ReflexRflx Reflex)
SET_TARGET_PROPERTIES(test_ReflexRflx PROPERTIES PREFIX "lib")
