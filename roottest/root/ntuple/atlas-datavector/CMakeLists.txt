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

# We don't use ROOTTEST_GENERATE_DICTIONARY as it does not reproduce well enough
# the ATLAS Athena build. Contributors
# - It always builds the C++ module from the dictionary (in a module build of ROOT)
# - It changes names of artifacts (e.g. removes the "lib" prefix from the shared library)
set(CMAKE_ROOTTEST_DICT OFF)
set(ATLASLIKEDATAVECTOR_DICTNAME AtlasLikeDataVectorDict)
ROOT_GENERATE_DICTIONARY(
	${ATLASLIKEDATAVECTOR_DICTNAME}
	${CMAKE_CURRENT_SOURCE_DIR}/AtlasLikeDataVector.hxx
	LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/selection.xml
)
ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_DICTIONARY_TEST ${ATLASLIKEDATAVECTOR_DICTNAME})
set(GENERATE_DICTIONARY_TEST ${GENERATE_DICTIONARY_TEST}-build)

# Generate a shared library from the dictionary sources
set(ATLASLIKEDATAVECTOR_LIB ${ATLASLIKEDATAVECTOR_DICTNAME}libgen)
add_library(${ATLASLIKEDATAVECTOR_LIB} SHARED ${ATLASLIKEDATAVECTOR_DICTNAME}.cxx)
set_target_properties(${ATLASLIKEDATAVECTOR_LIB} PROPERTIES ${ROOT_LIBRARY_PROPERTIES})
set_target_properties(${ATLASLIKEDATAVECTOR_LIB} PROPERTIES OUTPUT_NAME ${ATLASLIKEDATAVECTOR_DICTNAME})
target_link_libraries(${ATLASLIKEDATAVECTOR_LIB} Core RIO ROOTNTuple)
add_dependencies(${ATLASLIKEDATAVECTOR_LIB} ${ATLASLIKEDATAVECTOR_DICTNAME})

add_test(NAME ${GENERATE_DICTIONARY_TEST}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                           --target ${ATLASLIKEDATAVECTOR_DICTNAME} ${ATLASLIKEDATAVECTOR_LIB})
set_property(TEST ${GENERATE_DICTIONARY_TEST} PROPERTY FIXTURES_SETUP atlas_datavector_dict_setup)

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
    LIBRARIES Core RIO ROOTNTuple GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
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

# Regression test for the Python-only side of the issue. The lazy template
# instantiation done by the Python bindings was leading to a faulty behaviour:
# the user-provided class template "AtlasLikeDataVector<CustomStruct>" does not
# correspond to what the compiler sees, so we need to inject this information.
ROOT_ADD_PYUNITTEST(
    rentry_getptr
    rentry_getptr.py
    GENERIC
    PYTHON_DEPS pytest
    FIXTURES_REQUIRED atlas_datavector_write_done)

ROOT_ADD_PYUNITTEST(
    template_instantiation
    template_instantiation.py
    GENERIC
    PYTHON_DEPS pytest)
