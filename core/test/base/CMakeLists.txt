set (link_libs Core)

ROOT_ADD_GTEST(CoreBaseTests
  TQObjectTests.cxx
  LIBRARIES ${link_libs}
  )
