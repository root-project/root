#ifndef __CINT__
#include "Reflex/Tools.h"
#endif

void IT_Print(const char* what, bool expect) {
   bool res = Reflex::Tools::IsTemplated(what);
   std::cout << what << ": "
             << ( res ? "true" : "false")
             << " (" << ((res == expect) ? "OK" : "ERROR!") << ")"
             << std::endl;
}


void IsTemplated() {
   IT_Print("int", false);
   IT_Print("int f() const throw()", false);
   IT_Print("int f<T>() const throw()", true);
   IT_Print("A<int>", true);
   IT_Print("A<B<int>>", true);
   IT_Print("A< B< int > >", true);
   IT_Print("void (B::operator< int >)", false);
   IT_Print("void f<int>(B::operator< int >)", true);
   IT_Print("operator>()", false);
   IT_Print("operator>>()", false);
   IT_Print("A<T>::operator>()", false);
   IT_Print("A<T>::operator>>()", false);
   IT_Print("operator>><A>()", true);
   IT_Print("operator>>(A<B>)", false);
   IT_Print("A<T>::f(A<B>) const throw(C<D>)", false);
   IT_Print("A<T>::f<X>(A<B>) const throw(C<D>)", true);
   IT_Print("operator A<B>()", true);
}

void runTools() {
   gSystem->Load("libReflexDict");

   IsTemplated();
}
