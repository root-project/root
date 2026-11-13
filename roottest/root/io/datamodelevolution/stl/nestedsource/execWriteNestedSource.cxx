// Write a file with the old data model, where the interesting values are
// stored inside the nested `Deep` struct.  They are read back by
// execReadNestedSource.cxx with a data model in which those values have been
// moved to the top level of `Event`.
// See https://github.com/root-project/root/issues/19773

#include <Rtypes.h>
#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <vector>

struct Deep {
   float fDeepMember = 0.;
   int fDeepInt = 0;

   ClassDefNV(Deep, 1)
};

struct Event {
   int fId = 0;
   Deep fDeep;

   ClassDefNV(Event, 1)
};

#ifdef __ROOTCLING__
#pragma link C++ class Deep+;
#pragma link C++ class Event+;
#pragma link C++ class std::vector<Event>+;
#endif

int execWriteNestedSource()
{
   TFile file("nestedsource.root", "RECREATE");

   Event single;
   single.fId = 1;
   single.fDeep.fDeepMember = 10.;
   single.fDeep.fDeepInt = 100;
   file.WriteObject(&single, "e");

   std::vector<Event> vec;
   for (int i = 1; i <= 3; ++i) {
      Event event;
      event.fId = i;
      event.fDeep.fDeepMember = 10. * i;
      event.fDeep.fDeepInt = 100 * i;
      vec.push_back(event);
   }
   auto *vecPtr = &vec;
   file.WriteObject(vecPtr, "ve");

   TTree tree("t", "");
   tree.Branch("ve0", "std::vector<Event>", &vecPtr, 32000, 0);
   tree.Branch("ve99", "std::vector<Event>", &vecPtr, 32000, 99);
   tree.Fill();

   file.Write();

   std::cout << "Wrote " << vec.size() << " events.\n";

   return 0;
}
