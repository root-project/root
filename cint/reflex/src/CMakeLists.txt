FIND_PACKAGE(DL)

# collect headers and sources to create a library
FILE(GLOB Reflex_SRCS *.cxx *.h)
FILE(GLOB_RECURSE Reflex_HDRS ${REFLEX_INCLUDE_DIR}/*.h)
ADD_LIBRARY(Reflex ${Reflex_SRCS} ${Reflex_HDRS})

# link in dl if it exists
TARGET_LINK_LIBRARIES(Reflex ${DL_LIBRARY})

# install
INSTALL(TARGETS Reflex ${INSTALL_TARGETS_DEFAULT_ARGS})
