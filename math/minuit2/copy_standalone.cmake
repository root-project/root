# This adds a copy function. You give a directory and a
# file, and it copies from one to the other. All paths
# are relative to the source directory, not the build
# directory. You can see the copied files with
# STANDALONE_VERBOSE=ON
# 
# This is designed for a "standalone" folder inside a
# larger project.
#
# prepend_path(OUTVAR, ADDITION, ITEM1, [ITEM2...])
#   Prepend a path ADDITION to each item in the list and set OUTVAR with it

# copy_standalone(ORIGINAL_DIR NEW_DIR NAME1 [NAME2...])
#   Does something similar to:
#   cp ORIGINAL_DIR/NAME NEW_DIR/NAME
#
# MAKE_STANDALONE:         A global setting to turn on copying 
# COPY_STANDALONE_LISTING: A GLOBAL PROPERTY listing all files
#                          added (to set up purging)

function(PREPEND_PATH OUTVAR ADDITION)
    set(listVar "")
    foreach(f ${ARGN})
        list(APPEND listVar "${ADDITION}/${f}")
    endforeach(f)
    set(${OUTVAR} "${listVar}" PARENT_SCOPE)
endfunction()

set_property(GLOBAL PROPERTY COPY_STANDALONE_LISTING "")

function(COPY_STANDALONE ORIGINAL_DIR NEW_DIR)
    # Loop over all filenames given
    foreach(FILENAME ${ARGN})
        # All paths are relative to master directory
        set(ORIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${ORIGINAL_DIR}/${FILENAME}")
        set(NEW_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${NEW_DIR}/${FILENAME}")

        # Normalize paths
        get_filename_component(ORIG_FILE "${ORIG_FILE}" ABSOLUTE)
        get_filename_component(NEW_FILE "${NEW_FILE}" ABSOLUTE)

        # Verify that the file would not copy over itself
        if("${NEW_FILE}" STREQUAL "${ORIG_FILE}")
            message(FATAL_ERROR "You cannot set both directories to the same path! ${NEW_FILE}")
        endif()

        # This is a configure setting to turn on/off copying
        if(MAKE_STANDALONE)
            # Error if file to copy is missing
            if(NOT EXISTS "${ORIG_FILE}")
                message(FATAL_ERROR "The file ${ORIG_FILE} does not exist and COPY_STANDALONE_ACTIVATE was set to ON")
            endif()

            # Actually do the copy here
            configure_file("${ORIG_FILE}" "${NEW_FILE}" COPYONLY)

            # Allow cleaning with make purge
            set_property(GLOBAL APPEND PROPERTY COPY_STANDALONE_LISTING "${NEW_FILE}")
        else()
            # Error if file to copy to is missing (since copy is off)
            if(NOT EXISTS "${NEW_FILE}")
                message(FATAL_ERROR "The file ${NEW_FILE} does not exist and COPY_STANDALONE_ACTIVATE was not set to ON")
            endif()
        endif()
    endforeach()
    string(REPLACE ";" " " LISTING "${ARGN}")
    message(STATUS "Copied ${LISTING}")
endfunction()

