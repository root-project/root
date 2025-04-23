
ROOTTEST_ADD_TEST(ArrayCompress
                  MACRO runArrayCompress.C
                  OUTREF ArrayCompress.ref)

ROOTTEST_ADD_TEST(PolyMarker
                  MACRO runPolyMarker.C
                  OUTREF PolyMarker.ref)

ROOTTEST_GENERATE_DICTIONARY(libJsonTestClasses
                  test_classes.h
                  LINKDEF LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(BasicTypes
                  MACRO runBasicTypes.C
                  OUTREF BasicTypes.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(Objects
                  MACRO runObjects.C
                  OUTREF Objects.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(RootClasses
                  MACRO runRootClasses.C
                  OUTREF RootClasses.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(STL
                  MACRO runSTL.C
                  OUTREF STL.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(STL0
                  MACRO runSTL0.C
                  OUTREF STL0.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(STL1
                  MACRO runSTL1.C
                  OUTREF STL1.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(StreamerLoop
                  MACRO runStreamerLoop.C
                  OUTREF StreamerLoop.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(Map
                  MACRO runMap.C
                  OUTREF Map.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)

ROOTTEST_ADD_TEST(String
                  MACRO runString.C
                  OUTREF String.ref
                  FIXTURES_REQUIRED jsonTestClasses_fixture)
