############################################################################
# CMakeLists.txt file for building ROOT core/metacling package
############################################################################

add_subdirectory(src)

if(testing)
    add_subdirectory(test)
endif()
