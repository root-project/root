#include "i1.h"

template <class RootClass> A_Behavior<RootClass> grabBehavior( const A*, RootClass*) { return A_Behavior<RootClass>(); }


int func() {

  init<A>::run(); // output A
  init<B>::run(); // output B
  init<C>::run(); // output Default
  init<D>::run(); // output A
  init<E>::run(); // output A

  //  init i1( ((A*)0x0) );
  //  test<A,  grabBehavior( (A*)0x0 ) > a;
  
  return 0;
}

