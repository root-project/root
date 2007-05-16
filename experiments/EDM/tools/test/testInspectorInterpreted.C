#include "Inspector_FOO.hh"

bool testInspectorInterpreted(size_t o, size_t voidExp) {
  using namespace Inspect;
  ResultMap_t* expected = (ResultMap_t*)voidExp;
  void* obj = (void*)o;
  Inspector_FOO inspFOO2(obj);
  inspFOO2.SetExpectedResults(*expected);
  inspFOO2.Inspect();
  return inspFOO2.AsExpected();
}
