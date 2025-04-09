#include <atomic>
#include "TFile.h"
#include "TVirtualStreamerInfo.h"
#include "TTree.h"

class Holder {
public:
   std::atomic<const void*> fCache; //!

   ClassDef(Holder,2);
};

#ifdef __ROOTCLING__
#pragma link C++ class Holder+;
#pragma read sourceClass="Holder" targetClass="Holder" versions="[1-]" source="" target="fCache" code="{ fCache = nullptr; }"
#endif


int execAtomTransient()
{
   Holder obj;
   auto f = TFile::Open("AtomTransient.root","RECREATE");
   // auto t = new TTree("t","t");
   // t->Branch("obj.",&obj);
   auto sinfo = TClass::GetClass("Holder")->GetStreamerInfo();
   sinfo->ForceWriteInfo(f);
   return 0;
}