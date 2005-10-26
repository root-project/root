/*
  File: roottest/python/stl/StlTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 10/25/05
  Last: 10/25/05
*/

#include <list>
#include <vector>


class JustAClass {
public:
   int m_i;
};

namespace {
   std::vector< JustAClass > jv1;
   std::list< JustAClass >   jl1;
}

#ifdef __MAKECINT__
#pragma link C++ class std::vector< JustAClass >-;
#pragma link C++ class std::vector< JustAClass >::iterator-;
#pragma link C++ class std::list< JustAClass* >-;
#pragma link C++ class std::list< JustAClass* >::iterator-;
#endif
