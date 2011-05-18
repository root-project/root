#####################  dictionary  ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})

REFLEX_ADD_DICTIONARY(test_ReflexRflx TEST WITH_PREFIX Reflex.h
                      SELECTION selection.xml
                      OPTIONS --quiet)
