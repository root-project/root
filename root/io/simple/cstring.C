class TXmlEx4 {
   public: 
   TXmlEx4() : fStr2(0), fStr3(0) {}
      const char* fStr2;
      const char* fStr3;
};

#include "TFile.h"

void cstring() {
   TXmlEx4 a;
   TFile f("test.root","RECREATE");
   f.WriteObject(&a,"xmltest");
}
