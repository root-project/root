# FIXME: The tests in core should require only libCore. OTOH, TQObjectTests uses the interpreter to register the class.
# This means that if we run make CoreBaseTests the executable wouldn't be runnable because it requires libCling and
# onepcm targets to be built.

if(NOT MSVC)
  set(dllib dl)
endif()

ROOT_ADD_GTEST(CoreBaseTests
  TNamedTests.cxx
  TQObjectTests.cxx
  LIBRARIES Core Cling RIO ${dllib})
