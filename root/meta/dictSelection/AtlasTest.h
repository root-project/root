// Dear emacs, this is -*- c++ -*-
/*
This test has been provided by Attila Krasznahorkay.
*/
#ifndef DICTRULES_DICT_H
#define DICTRULES_DICT_H

// Local include(s):
#include "ClassA.h"
#include "ClassB.h"
#include "ClassC.h"

namespace {
   struct DUMMY_INSTANTIATION {
      Atlas::ClassA< Atlas::ClassB > dummy1;
      Atlas::ClassA< Atlas::ClassC > dummy2;
      ROOT_SELECTION_NS::Atlas::ClassA< Atlas::ClassB > dummy3;
      ROOT_SELECTION_NS::Atlas::ClassA< Atlas::ClassC > dummy4;
   };
}

#endif // not DICTRULES_DICT_H
