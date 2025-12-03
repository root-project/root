#include <list>
class A {};
template <class T> class B{};
template <class T> class C {};

#ifdef __CINT__
#pragma link C++ class A+;
#pragma link C++ class B<A*>+;
#pragma link C++ class C<B<A*>*>+;
#pragma link C++ class C<list<A*>*>+;
#endif

