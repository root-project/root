#include "i1.h"

int func();

template <class T, class B> class test {};

int main() {
  /*
  
  init<A>::run();
  init<B>::run();
  init<C>::run();
  */

  //  init i1( ((A*)0x0) );
  //  test<A,  grabBehavior( (A*)0x0 ) > a;

  func();

  return 0;
}

