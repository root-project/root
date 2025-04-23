class TopLevel { public: virtual ~TopLevel() {} };
class BottomOne : public TopLevel {};
#ifndef __CINT__
#ifndef __CLING__
class BottomMissing : public TopLevel {};
#endif
#endif

#include "Riostream.h"
#include "TClass.h"

#ifdef __CLING__
void missingInfo();
#else
void missingInfo() {
   TopLevel *one = new BottomOne;
   TopLevel *missing = new BottomMissing;

   TClass *cl;
   
   cl = TClass::GetClass(typeid(TopLevel));
   TClass *top = cl;
   if (cl) {
      cout << "For toplevel found " << endl; 
      //cout << (void*)cl << endl;
      cl->Print();
   } else {
      cout << "For toplevel cl is missing \n";
   }

   cl = TClass::GetClass(typeid(*one));
   if (cl) {
      cout << "For one found " << endl; 
      //cout << (void*)cl << endl;
      cl->Print();
   } else {
      cout << "For one cl is missing \n";
   }

   cl = TClass::GetClass(typeid(*missing));
   if (cl) {
      cout << "For missing found " << endl; 
      //cout << (void*)cl << endl;
      cl->Print();
   } else {
      cout << "For missing cl is missing \n";
   }

   if (top) {
      cl = top->GetActualClass(one);
      if (cl) {
         cout << "For one found " << endl; 
         //cout << (void*)cl << endl;
         cl->Print();
      } else {
         cout << "For one cl is missing \n";
      }

      cl = top->GetActualClass(missing);
      if (cl) {
         cout << "For missing found " << endl; 
         //cout << (void*)cl << endl;
         cl->Print();
      } else {
         cout << "For missing cl is missing \n";
      }

   }
}
#endif // hidden from cling/rootcling
