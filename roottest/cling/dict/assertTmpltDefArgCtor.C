#include <TClass.h>
#include <TListOfFunctions.h>
#include <iostream>

struct TheStruct {
  const char* fMember = nullptr;

  template <class T = float>
  TheStruct(char): fMember(T()) {}

  TheStruct(char*) {}
  TheStruct() = default;
};

#ifdef __CLING__
#pragma link C++ class TheStruct;
#endif

int assertTmpltDefArgCtor() {
  auto cl = TClass::GetClass("TheStruct");
  if (!cl) {
    std::cerr << "ERROR: cannot get TClass\n";
    exit(1);
  }
  TListOfFunctions* lof
    = dynamic_cast<TListOfFunctions*>(cl->GetListOfMethods());
  auto ctors = lof->GetListForObject("TheStruct");
  if (!ctors) {
    std::cerr << "ERROR: cannot get constructors\n";
    exit(1);
  }

  // Expect:
  //   TheStruct()
  //   TheStruct(TheStruct const&)
  //   TheStruct(TheStruct&&)
  //   TheStruct(char*)
  // but NOT TheStruct(char)!
  if (ctors->GetSize() != 4) {
    std::cerr << "ERROR: wrong constructor count: "
      "expected 4, got " << ctors->GetSize() << '\n';
    exit(1);
  }
  return 0;
}
