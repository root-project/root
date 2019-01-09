ROOT_ADD_UNITTEST_DIR(Imt Thread)

ROOT_ADD_GTEST(testImt testTFuture.cxx testTTaskGroup.cxx LIBRARIES Imt)
