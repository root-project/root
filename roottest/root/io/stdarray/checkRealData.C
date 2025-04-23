#include <iostream>
#include "TClass.h"
#include "TList.h"
#include "TRealData.h"
#define ARRAYHOLDER_STDARRAY
#include "arrayHolder.h"

class A2{
#ifdef ARRAYHOLDER_STDARRAY
   std::array<float,3> m_a2;
#else
   float m_a2[3];
#endif
};

class A1{
   A2 m_a1;
};

class B{int a; std::array<int,3> b; int c[3];};


void checkRealData() {

   for (auto&& clName : {"B", "ArrayHolder", "MetaArrayHolder", "MetaArrayHolder2", "A1"}){
      auto c = TClass::GetClass(clName);
      c->BuildRealData();
      auto lrd = c->GetListOfRealData();
      unsigned int i=0;
      for (auto rdo : *lrd) {
         auto rd = (TRealData*)rdo;
         cout << clName << " Real Data " << i++ << " " << rd->GetName() << " offset " << rd->GetThisOffset() << endl;
      }
      std::cout << "----------\n";
   }
}

