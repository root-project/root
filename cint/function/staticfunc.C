#ifndef __CINT__
#include "TString.h"
#include "Riostream.h"
#endif

class Top {

public:
   void func(TString &s) {
      cout << s.Data() << endl;
   }
   void Run() {
      func("test");
   }
};

int staticfunc() {
   Top p;
   p.Run();
   Top::Run();
   return 0;
}
