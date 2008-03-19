/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// demo/exception/ehdemo.cxx
// This source has to be interpreted

#include <stdio.h>

#if defined(interp) && defined(makecint)
#include "test.dll"
#else
#include "ehdemo.h"
#endif

////////////////////////////////////////////////////////////////////////
// You can use simple try, catch block in an interpreted code. However,
// that portion can not be bytecode optimized. 
////////////////////////////////////////////////////////////////////////
void test2(char* name) {
  try {
    test(name);
  }
  catch(my_exception& x) {
    printf("%s\n",x.what()); 
  }
  catch(exception& z) {
    printf("This is a std::exception '%s'\n",z.what()); 
  }
  catch(unhandled_exception& z) {
    // Sorry, this case can not be handled because unhandled_exception
    // is not derived from std::exception
    printf("unhandled_exception '%s'\n",z.what()); 
  }
  catch(...) {
    // Sorry, this case is implicitly handled within Cint. This interpreted
    // code has no effect
    printf("unknown exception\n"); 
  }
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
int main() {
  int i;
  printf("------- START -------\n");
  // Calling compiled function that throws exception with interpreted
  // try, catch block
  for(i=0;i<2;i++) {
    test2("abcd");  // success
    test2(0);       // my_exception NULL pointer
    test2("");      // my_exception Empty string
    test2("error"); // std::exception
    // Following call terminates Cint
    //test2("unknown"); // unhandled_exception
  }

  // Calling compiled function that throws exception without interpreted
  // try, catch block
  for(i=0;i<2;i++) {
    test("abcd");  // success
    // Following calls terminates Cint
    //test(0);       // my_exception NULL pointer
    //test("");      // my_exception Empty string
    //test("error"); // std::exception
    //test("unknown"); // unhandled_exception
  }
  printf("-------- END --------\n");
  return 0;
}
