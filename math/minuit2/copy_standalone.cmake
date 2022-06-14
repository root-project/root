# This adds a copy function. You give a directory and a
# file, and it copies from one to the other. All paths
# are relative to the source directory, not the build
# directory. You can see the copied files with
# STANDALONE_VERBOSE=ON
#
# This is designed for a "standalone" folder inside a
# larger project.
#
# prepend_path(outvar addition item1 [item2...])
#   Prepend a path ADDITION to each item in the list and set OUTVAR with it
#
# copy_standalone(SOURCE original_dir
#                 DESTINATION new_dir
#                 [OUTPUT variable_name]
#                 FILES name1 [name2...])
#
#   For each file, does something similar to:
#
#   If minuit2_inroot and minuit2_standalone:
#     cp ORIGINAL_DIR/NAME NEW_DIR/NAME
#     set the NAME in ${OUTPUT} to NEW_DIR/NAME
#
#   If minuit2_inroot and not minuit2_standalone:
#     set the NAME in ${OUTPUT} to OLD_DIR/NAME
#
#   If not minuit2_inroot:
#     set the NAME in ${OUTPUT} to NEW_DIR/NAME
#
# minuit2_inroot:          A global setting that indicates that we are in the ROOT source
# minuit2_standalone:      A global setting to turn on copying
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

# Needed for CMake 3.4 and lower:
include(CMakeParseArguments)

function(COPY_STANDALONE)

    # CMake keyword arguments
    set(options "")
    set(oneValueArgs OUTPUT SOURCE DESTINATION)
    set(multiValueArgs FILES)
    cmake_parse_arguments(COPY_STANDALONE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Error messages
    if(COPY_STANDALONE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "copy_standalone requires keywords before all arguments")
    endif()

    if(NOT COPY_STANDALONE_FILES)
        message(FATAL_ERROR "copy_standalone requires files to work on")
    endif()

    # Get and normalize path to new directory
    set(NEW_DIR_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${COPY_STANDALONE_DESTINATION}")
    get_filename_component(NEW_DIR_FULL "${NEW_DIR_FULL}" ABSOLUTE)

    # Keep track of all files listed
    set(FILENAMES "")

    # Loop over all filenames given
    foreach(FILENAME ${COPY_STANDALONE_FILES})
        # All paths are relative to master directory
        set(ORIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${COPY_STANDALONE_SOURCE}/${FILENAME}")
        set(NEW_FILE "${NEW_DIR_FULL}/${FILENAME}")

        # Normalize paths
        get_filename_component(ORIG_FILE "${ORIG_FILE}" ABSOLUTE)
        get_filename_component(NEW_FILE "${NEW_FILE}" ABSOLUTE)

        if(minuit2_inroot)
            # Error if file to copy is missing
            if(NOT EXISTS "${ORIG_FILE}")
                message(FATAL_ERROR "The file ${ORIG_FILE} does not exist and minuit2_inroot was set to ON")
            endif()

            # This is a configure setting to turn on/off copying
            if(minuit2_standalone)

                # Verify that the file would not copy over itself
                if("${NEW_FILE}" STREQUAL "${ORIG_FILE}")
                    message(FATAL_ERROR "You cannot set both directories to the same path! ${NEW_FILE}")
                endif()

                # Actually do the copy here
                file(COPY "${ORIG_FILE}" DESTINATION "${NEW_DIR_FULL}")

                # Allow cleaning with make purge
                set_property(GLOBAL APPEND PROPERTY COPY_STANDALONE_LISTING "${NEW_FILE}")

                # Add new file to filename listing
                list(APPEND FILENAMES "${NEW_FILE}")
            else()
                # Add old file to filename listing
                list(APPEND FILENAMES "${ORIG_FILE}")
            endif()
        else()
            # Error if file to copy to is missing (since copy is off)
            if(NOT EXISTS "${NEW_FILE}")
                message(FATAL_ERROR "The file ${NEW_FILE} does not exist and minuit2_inroot was not set to ON")
            endif()

            # Add new file to filename listing
            list(APPEND FILENAMES "${NEW_FILE}")
        endif()
    endforeach()

    if(minuit2_inroot AND minuit2_standalone)
        string(REPLACE ";" ", " LISTING "${COPY_STANDALONE_FILES}")
        message(STATUS "Copied to ${NEW_DIR_FULL}: ${LISTING}")
    endif()

    # Output list of file names
    if(COPY_STANDALONE_OUTPUT)
        set(${COPY_STANDALONE_OUTPUT} ${FILENAMES} PARENT_SCOPE)
    endif()
endfunction()

