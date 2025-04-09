#include <iostream>
using namespace std;
#include "templateMembersClasses.h"

void templateMembersCode() {

   // code in this file works both compiled and interpreted.
   // code in main.cxx works only compiled.
   
   cout << "\nConstruction...........\n";
   Base b;
   Derived d;
   TemplateClass<Derived> td;
   my_shared_ptr<Base> spb;
   my_shared_ptr<Derived> spd;
   
   cout << "\nMember Functions........\n";
   // template member function calls
   spb.f1(d);
   spb.f2<Derived>();
   spb.f3(td);
   spb.f4(td);
   spb.f5(spd);
   spb.f6(spd);
   spb.f7(spd);
   spb.f8(spb);
   spb.f8(spd);
   
   spb.n1(spb);
   spb.n2(spb);
   spb.n3(spb);
   spb.n4(spb);
   
   spb.n1(spd);
   spb.n2(spd);
   spb.n3(spd);
   spb.n4(spb);
      
   spb = spd;
   
   my_shared_ptr<Base> spb1(spd);
   

}
