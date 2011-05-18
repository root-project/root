#####################  dictionary  ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

REFLEX_ADD_DICTIONARY(test_Class2DictRflx TEST WITH_PREFIX Class2Dict.h
                      SELECTION selection.xml
                      OPTIONS --quiet)
