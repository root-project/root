#include "pairtest.h"
#include <iostream>
#include "TClass.h"
#include "TBufferFile.h"
#include "TTree.h"
#include "TVirtualStreamerInfo.h"

int outerPair()
{
   std::pair<SameAsShort, short> p;
   auto cl = TClass::GetClass<decltype(p)>();
   if (!cl) {
      Error("emulatePairs", "Could not get the TClass for %s from the templated TClass::GetClass", typeid(p).name());
      return 1;
   }
   if (cl->GetClassSize() != sizeof(p)) {
      Error("emulatePairs", "TClass for %s has the wrong size: %d instead of %d", typeid(p).name(), cl->GetClassSize(), (int)sizeof(p));
      return 2;
   }
   auto offset = ((char*)&(p.second)) - (char*)&p;
   auto elem = dynamic_cast<TStreamerElement*>(cl->GetStreamerInfo()->GetElements()->FindObject("second"));
   if (!elem) {
      Error("emulatePairs", "TClass for %s could not find the TStreamerElement for second", typeid(p).name());
      return 3;
   }
   if (elem->GetOffset() != offset) {
      Error("emulatePairs", "TClass for %s has the wrong offset for second: %d instead of %d", typeid(p).name(), elem->GetOffset(), (int)offset);
      return 4;
   }

   p.first.fValue = 11;
   p.second = 12;
   cl->Dump(&p);

   if (!cl->GetListOfRealData())
      cl->BuildRealData(&p);

   if (!cl->GetListOfRealData()) {
      Error("emulatePairs", "TClass for %s can not create ListOfRealData", typeid(p).name());
      return 5;
   }
   // cl->GetListOfRealData()->ls();

   TTree t("t", "t");
   t.Branch("pair.", &p);
   t.Fill();
   t.Show(0);

   return 0;
}

int main(int, char **)
{
   gErrorIgnoreLevel = kPrint; // Get all the output even if .rootrc or env says otherwise                                               

   gInterpreter->SetClassAutoLoading(false);
   TInterpreter::SuspendAutoParsing s(gInterpreter);

   // gInterpreter->Declare("#include \"pairtest.h\"");
   fprintf(stdout, "Contains 1\n");
   fprintf(stderr, "Contains 2\n");

   gSystem->Load("libPairs.so");

   Contains c;
   TClass *cl = TClass::GetClass("Contains");

   c.fShort.first = '1';
   c.fShort.second = 2;
   c.fSameAs.first = '3';
   c.fSameAs.second.fValue = 4;
   cl->Dump(&c);
   //cl->GetListOfRealData()->ls();
   TTree t("t", "t");
   t.Branch("pair.", &c);
   t.Fill();
   t.Show(0);

   //cl->GetStreamerInfos()->ls();
   //cl->GetListOfRealData()->ls();
   //TClass *subcl = TClass::GetClass("pair<unsigned char, SameAsShort>");
   //subcl->GetStreamerInfo()->ls();

   std::cout << "size of pair<char, SameAsShort>: " << TClass::GetClass("pair<unsigned char, SameAsShort>")->GetClassSize() << '\n';

   TBufferFile b(TBuffer::kWrite);
   cl->Streamer(&c, b);
   b.SetReadMode();
   b.SetBufferOffset(0);

   Contains cr;
   cl->Streamer(&cr, b);
   std::cout << cr.fShort.first << '\n';
   std::cout << cr.fShort.second << '\n';
   std::cout << cr.fSameAs.first << '\n';
   std::cout << cr.fSameAs.second.fValue << '\n';

   std::cout << "size of pair<char, SameAsShort>: " << TClass::GetClass("pair<unsigned char, SameAsShort>")->GetClassSize() << '\n';

   return outerPair();
}
