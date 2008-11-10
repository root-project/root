//----- some code
#include "Api.h"
class TCanvas {};

#include <iostream>
#include <vector>

#ifdef __CINT__
#pragma link C++ class std::vector< TCanvas* >;
#pragma link C++ class std::vector< TCanvas* >::iterator;
#else
template class std::vector< TCanvas* >;
#endif

int main() {
   std::string sgcl = "vector<TCanvas*>";
   G__ClassInfo gcl( sgcl.c_str() );
   std::cout << "is valid class (" << sgcl << "): " << gcl.IsValid() << std::endl;

//   std::cout << "full method according to meta: " <<
//      TClass( sgcl.c_str() ).GetMethodAny( "push_back" )->GetPrototype() << std::endl;

   long offset = 0;
 
#if 0
    TCanvas *  ptr = 0;
   TCanvas * const &refptr = ptr;
   const TCanvas *&cintptr = ptr;
   std::cout << typeid(ptr).name() << endl;
   std::cout << typeid(refptr).name() << endl;
   std::cout << typeid(cintptr).name() << endl;
#endif 
   {
      std::string sti1 = "TCanvas* const";
      G__TypeInfo ti1( sti1.c_str() );
      std::cout << "is valid type (" << sti1 << "): " << ti1.IsValid() << std::endl;
      G__MethodInfo meth1 = gcl.GetMethod(
                                          "push_back", sti1.c_str(), &offset, G__ClassInfo::ExactMatch );
      std::cout << "is valid method (" << sti1 << "): " << meth1.IsValid() << std::endl;
      
      std::string sti2 = "const TCanvas*";
      G__TypeInfo ti2( sti2.c_str() );
      std::cout << "is valid type (" << sti2 << "): " << ti2.IsValid() << std::endl;
      G__MethodInfo meth2 = gcl.GetMethod(
                                          "push_back", sti2.c_str(), &offset, G__ClassInfo::ExactMatch );
      std::cout << "is valid method (" << sti2 << "): " << meth2.IsValid() << std::endl;   
   }
   {
      std::string sti1 = "TCanvas* const&";
      G__TypeInfo ti1( sti1.c_str() );
      std::cout << "is valid type (" << sti1 << "): " << ti1.IsValid() << std::endl;
      G__MethodInfo meth1 = gcl.GetMethod(
                                          "push_back", sti1.c_str(), &offset, G__ClassInfo::ExactMatch );
      std::cout << "is valid method (" << sti1 << "): " << meth1.IsValid() << std::endl;
      
      std::string sti2 = "const TCanvas*&";
      G__TypeInfo ti2( sti2.c_str() );
      std::cout << "is valid type (" << sti2 << "): " << ti2.IsValid() << std::endl;
      G__MethodInfo meth2 = gcl.GetMethod(
                                          "push_back", sti2.c_str(), &offset, G__ClassInfo::ExactMatch );
      std::cout << "is valid method (" << sti2 << "): " << meth2.IsValid() << std::endl;
   }
}
