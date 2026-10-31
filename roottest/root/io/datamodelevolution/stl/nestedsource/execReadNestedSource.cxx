// Read the file written by execWriteNestedSource.cxx with a data model in
// which the members of the nested `Deep` struct have been moved to the top
// level of `Event`, where they are filled by an I/O rule whose source is the
// nested struct.
//
// All four cases below must report the same values; before the fix for issue
// https://github.com/root-project/root/issues/19773 the elements read from the
// split TTree branch were left with the default values because the on-file
// staging area the rule reads from was never filled.

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
   float fMember = 0.;
   int fIntMember = 0;

   ClassDefNV(Event, 2)
};

#ifdef __ROOTCLING__
#pragma link C++ class Deep+;
#pragma link C++ class Event+;
#pragma link C++ class std::vector<Event>+;

#pragma read sourceClass = "Event" source = "Deep fDeep" version = "[1]" targetClass = "Event" \
   target = "fMember,fIntMember" code = "{ fMember = onfile.fDeep.fDeepMember; fIntMember = onfile.fDeep.fDeepInt; }"
#endif

void Print(const char *what, const Event &event)
{
   std::cout << what << ": fId=" << event.fId << " fMember=" << event.fMember << " fIntMember=" << event.fIntMember
             << "\n";
}

void Print(const char *what, const std::vector<Event> &events)
{
   for (unsigned int i = 0; i < events.size(); ++i) {
      std::cout << what << " i=" << i;
      std::cout << ": fId=" << events[i].fId;
      std::cout << " fMember=" << events[i].fMember;
      std::cout << " fIntMember=" << events[i].fIntMember;
      std::cout << "\n";
   }
}

int execReadNestedSource()
{
   TFile file("nestedsource.root", "READ");

   auto *single = file.Get<Event>("e");
   Print("Plain object", *single);

   auto *vec = file.Get<std::vector<Event>>("ve");
   Print("Plain vector", *vec);

   auto *tree = file.Get<TTree>("t");

   std::vector<Event> *ve0 = nullptr;
   std::vector<Event> *ve99 = nullptr;
   tree->SetBranchAddress("ve0", &ve0);
   tree->SetBranchAddress("ve99", &ve99);

   tree->GetEntry(0);

   Print("Tree (splitlevel 0)", *ve0);
   Print("Tree (splitlevel 99)", *ve99);

   return 0;
}
