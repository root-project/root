#####################  dictionary  ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

REFLEX_GENERATE_DICTIONARIES(dictfile
                             HEADERS Class2Dict.h
                             SELECTION selection.xml
                             OPTIONS --quiet)
ADD_LIBRARY(test_Class2DictRflx MODULE ${dictfile})
TARGET_LINK_LIBRARIES(test_Class2DictRflx Reflex)
SET_TARGET_PROPERTIES(test_Class2DictRflx PROPERTIES PREFIX "lib")
