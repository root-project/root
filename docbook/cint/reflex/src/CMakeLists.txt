MACRO_FIND_PACKAGE_IF(DEMANGLE SOLARIS REQUIRED)

#####################   sources   ####################

FILE(GLOB Reflex_SRCS *.cxx *.h)
# Do not include ROOT's CINT dictionary
FILE(GLOB Reflex_SRCS_VETO G__Reflex.*)
IF(Reflex_SRCS_VETO)
   LIST(REMOVE_ITEM Reflex_SRCS ${Reflex_SRCS_VETO})
ENDIF(Reflex_SRCS_VETO)
FILE(GLOB_RECURSE Reflex_HDRS ${REFLEX_INCLUDE_DIR}/*.h)

#####################    build    ####################

INCLUDE_DIRECTORIES(${REFLEX_INCLUDE_DIR})
REFLEX_ADD_LIBRARY(Reflex ${Reflex_SRCS} ${Reflex_HDRS})
TARGET_LINK_LIBRARIES(Reflex ${DL_LIBRARY} ${DEMANGLE_LIBRARY})

# hack: unit tests currently use non-exported symbols that cause link
# errors under msvc. possible solutions:
# 1) stop testing non-API methods (preferred) and test API-visible features instead
# 2) add a new RFLX_INTERNAL or equivalent macro that exports symbols.
IF (MSVC AND REFLEX_ENABLE_TESTS AND BUILD_SHARED_LIBS)
   MACRO_ADD_LINK_FLAGS(Reflex "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/Reflex.def")
ENDIF (MSVC AND REFLEX_ENABLE_TESTS AND BUILD_SHARED_LIBS)

#####################   install   ####################

INSTALL(TARGETS Reflex ${INSTALL_TARGETS_DEFAULT_ARGS})
