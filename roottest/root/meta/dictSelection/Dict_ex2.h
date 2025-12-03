// Dear emacs, this is -*- c++ -*-
#ifndef DICTRULES_DICT_H
#define DICTRULES_DICT_H

// System include(s):
#include <vector>

// Local include(s):
#include "ClassA_ex2.h"
#include "ClassB_ex2.h"
#include "ClassC_ex2.h"
#include "ClassD_ex2.h"

namespace {
   struct DUMMY_INSTANTIATION {
      Atlas::ClassA< Atlas::ClassB > dummy1;
      Atlas::ClassA< Atlas::ClassC > dummy2;
      std::vector< Atlas::ClassA< Atlas::ClassB > > dummy3;
      std::vector< Atlas::ClassA< Atlas::ClassC > > dummy4;

      Atlas::ClassD< Atlas::ClassB > dummy5;
      Atlas::ClassD< Atlas::ClassC > dummy6;

      Atlas::ClassA< Atlas::ClassD< Atlas::ClassB > > dummy7;
      Atlas::ClassA< Atlas::ClassD< Atlas::ClassC > > dummy8;

      std::vector< Atlas::ClassA< Atlas::ClassD< Atlas::ClassB > > > dummy9;
      std::vector< Atlas::ClassA< Atlas::ClassD< Atlas::ClassC > > > dummy10;
   };
}

#endif // not DICTRULES_DICT_H
