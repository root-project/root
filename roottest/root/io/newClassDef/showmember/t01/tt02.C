namespace ROOT {
  void func() {};
}
#include <typeinfo>

template <class T> TClass* tfunc(T&obj) { 
  return gROOT->GetClass(typeid(obj).name());};
