# Reproducer of https://github.com/root-project/root/issues/19022

# TODO: Figure out how to load the dictionary in the writing test on Windows
if(MSVC AND NOT win_broken_tests)
 return()
endif()

# Generate the dictionary for the ATLAS-like container class
# Note that we use a selection.xml file, not a LinkDef, as this is what
# is done in Athena as well. Also note that in the selection we have a rule
# that uses a pattern AtlasLikeDataVector<CustomStruct,*> so we don't rely
# on an explicit declaration of the full class template with 2 arguments.
ROOTTEST_GENERATE_DICTIONARY(
    AtlasLikeDataVectorDict
    ${CMAKE_CURRENT_SOURCE_DIR}/AtlasLikeDataVector.hxx
    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/selection.xml
    FIXTURES_SETUP atlas_datavector_dict_setup
)

# Generate a shared library from the dictionary sources
add_library(AtlasLikeDataVector SHARED AtlasLikeDataVectorDict.cxx)
target_link_libraries(AtlasLikeDataVector Core RIO ROOTNTuple) 
add_dependencies(AtlasLikeDataVector AtlasLikeDataVectorDict)

# Generate an executable to write an RNTuple with a field of type
# AtlasLikeDataVector<CustomStruct>. Note that we don't link explicitly
# the executable against the shared library, but the information of the
# dictionary should be autoloaded, thus the rule to drop the second template
# argument should kick in when writing the class to disk.
ROOTTEST_GENERATE_EXECUTABLE(
    write
    write.cxx
    LIBRARIES Core RIO ROOTNTuple
    FIXTURES_REQUIRED atlas_datavector_dict_setup
    FIXTURES_SETUP atlas_datavector_write_setup)

# Generate an executable to read back the RNTuple with a field of type
# AtlasLikeDataVector<CustomStruct>. Note that we don't link explicitly
# the executable against the shared library, but the information of the
# dictionary should be autoloaded, thus the rule to drop the second template
# argument should kick in when reading the field.
ROOTTEST_GENERATE_EXECUTABLE(
    read
    read.cxx
    LIBRARIES Core RIO ROOTNTuple gtest gtest_main
    FIXTURES_REQUIRED atlas_datavector_dict_setup
    FIXTURES_SETUP atlas_datavector_read_setup)

ROOTTEST_ADD_TEST(write
    EXEC ${CMAKE_CURRENT_BINARY_DIR}/write
    FIXTURES_REQUIRED atlas_datavector_write_setup
    FIXTURES_SETUP atlas_datavector_write_done)

# In the reading test we rely on the templated call that uses the shortened
# class type AtlasLikeDataVector<CustomStruct>, so we need to include the header
ROOTTEST_ADD_TEST(read
    EXEC ${CMAKE_CURRENT_BINARY_DIR}/read
    COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/AtlasLikeDataVector.hxx
    FIXTURES_REQUIRED "atlas_datavector_read_setup;atlas_datavector_write_done")
