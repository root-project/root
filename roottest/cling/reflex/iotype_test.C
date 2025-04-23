#include "TClass.h"
#include "TDataMember.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include "TestHelper.h"
#include "iotype.h"

void checkIOType(const char* name) {
   printf("\ndata member '%s' \n", name);
   TClass* cl = TClass::GetClass("CIoType");
   RflxAssert(cl);
   TDataMember* dm = cl->GetDataMember(name);
   RflxAssert(dm);
   printf("\ndata member '%s', type '%s'\n", name, dm->GetTypeName());
   if (strstr("32", dm->GetName())) {
      const char* tname = dm->GetTypeName();
      RflxAssert(strstr(tname, "Double32_t"));
   }
}

void writeIOT() {
   TFile* f = new TFile("iotype.root", "RECREATE");
   TTree* t = new TTree("iotype", "rootest/iotype_test");
   CIoType* o = new CIoType();
   t->Branch("b", o);
   for (int i = 0; i < 1000; ++i) {
      o->Set(12. + i % 100);
      t->Fill();
   }
   t->Write();
   delete f;
}

void readIOT() {
   TFile* f = new TFile("iotype.root");
   TTree* t = 0;
   f->GetObject("iotype", t);
   TBranch* bEL = t->GetBranch("b");
   TIter iBranch(bEL->GetListOfBranches());
   TBranch* b = 0;
   while ((b = (TBranch*) iBranch())) {
      // aka EndsWith():
      TString name(b->GetName());
      if (name.EndsWith("32")) {
         // Remove d32:
         name.Remove(name.Length() - 3);
         // and maybe yet another 'd' from dd32:
         if (name.EndsWith("d")) {
            name.Remove(name.Length() - 1);
         }
         name += 'f';
         printf("Comparing %s with %s\n", b->GetName(), name.Data());
         TBranch* fb = t->GetBranch(name);
         size_t nBytesDbl32 = b->GetTotBytes("*");
         size_t nBytesFloat = fb->GetTotBytes("*");
         float relBytesDiff2 = (nBytesDbl32 - nBytesFloat) / (nBytesDbl32 + nBytesFloat);
         relBytesDiff2 *= relBytesDiff2;
         if (!strcmp(b->GetName(), "Tvd32")) {
            printf("EXPECTED FAILURE (missing dictionary for branch's %s "
                   "type 'Template<std::vector<Double32_t> >')\n",
                   b->GetName());
         } else if (!strcmp(b->GetName(), "Tvdd32")) {
            printf("EXPECTED FAILURE (missing dictionary for branch's %s "
                   "type 'Template<std::vector<Double32_t> >')\n",
                   b->GetName());
         } else {
            RflxAssert(relBytesDiff2 < 0.001);
         }
      }
   }
   delete f;
}

void iotype_test() {
   auto t = TClass::GetClass("CIoType");
   RflxAssert(t);
   for (auto d : *t->GetListOfDataMembers()) {
      checkIOType(d->GetName());
   }
   writeIOT();
   readIOT();
}
