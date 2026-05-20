#ifndef _classes_
#define _classes_

#include <set>
#include "h1.h"
#include "h2.h"

myClass8<E> a;

// ROOT-10777
template <class T, bool> class TemplSpec;
template <class T> class TemplSpec<T*, true> {
  void func();
};

template <class T>
void TemplSpec<T*, true>::func() {}

#endif
