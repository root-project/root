#include <iostream>
#include "TCut.h"

void runTCut()
{
   TCut a1="x>0";
   TCut a2="x>0";
   TCut a3="y>0";
   if (a1!=a2) std::cout << "not equal\n"; else std::cout << "equal\n";
   if (a1!=a2.GetTitle()) std::cout << "not equal\n"; else std::cout << "equal\n";
   if (a1!=a3) std::cout << "not equal\n"; else std::cout << "equal\n";
   if (TString(a1)!=TString(a2)) std::cout << "TString not equal\n"; else std::cout << "TString equal\n";
   if (TString(a1)!=TString(a3)) std::cout << "TString not equal\n"; else std::cout << "TString equal\n";
#if 0
   if (a1<a2) std::cout << "Did a meaningless comparison (<)\n";
#endif
//   TCut a4 = a1 - a2;
//   std::cout << "a4 :" << a4 << endl;
}
