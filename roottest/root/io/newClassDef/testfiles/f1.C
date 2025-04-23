#include <stdio.h>

template <class T> class MyTemplate {
public:
  T* GetValue() { return 0; };
  const char* Class_Name();
  static const char* fgIsA;
  
  MyTemplate *getThis() { return this; }
};

template <> const char* MyTemplate<int>::Class_Name() {
  return "template of int";
}

template <> const char* MyTemplate<double>::Class_Name() {
  return "template of double";
}

template <> const char *MyTemplate<int >::fgIsA = 0;
template <> const char *MyTemplate<double >::fgIsA = 0;

