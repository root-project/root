#include "TInterpreter.h"
#include "TClass.h"
#include "TError.h"
#include <set>
#include <string>

int execTypeinfo()
{
  if (gInterpreter->IsLoaded("libsetDict")) {
     Error("execTypeinfo","libsetDict.so is already loaded");
     return 1;
  }
  TClass *cl = TClass::GetClass(typeid(std::set<int>));
  if (!cl || !cl->IsLoaded()) {
     Error("execTypeinfo","The TClass for set<int> did not properly load from a typeinfo.");
     return 2;
  }
  if (!gInterpreter->IsLoaded("libsetDict")) {
     Error("execTypeinfo","The library libsetDict.so did not load.");
     return 3;
  }
  return 0;
}
