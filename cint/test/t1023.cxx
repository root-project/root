/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#define FORCE_TRUE

#include <stdio.h>

template<class T >
class myvector {
 public:
  typedef T value_type;
   
   class iterator {};

#if defined(FORCE_TRUE)
  typedef iterator reverse_iterator;
#else // (G__GNUC>=3 && G__GNUC_MINOR>=1) 
   is not here;
  class reverse_iterator 
#if defined(IS_NOT_DEFINED) 
#endif
	{
  };
  should not be here at line 19;
#endif // defined(FORCE_TRUE)

};

myvector<int> a;

int main() {
  printf("success\n");
  return 0;
}

