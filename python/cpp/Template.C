/*
  File: roottest/python/cpp/Template.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 01/07/08
  Last: 01/07/08
*/

#include <vector>

template< class T >
class MyTemplatedClass {
public:
   T m_b;
};

#ifdef __CINT__
#pragma link C++ class MyTemplatedClass< std::vector< float > >;
#endif
