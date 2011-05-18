SET(REFLEX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

#####################   install   ####################

INSTALL(DIRECTORY Reflex
        DESTINATION ${INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN ".svn" EXCLUDE)

