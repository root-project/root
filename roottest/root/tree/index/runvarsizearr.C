#include "TTree.h"
#include "TTreeIndex.h"
#include <vector>
#include <iostream>

// https://github.com/root-project/root/pull/19619
int runvarsizearr()
{
   const Long64_t nEvents = 2;
   const size_t nElements = 3;
   std::vector<int> otherNumbers(nElements);
   Long64_t eventNumber;
   TTree t("tree", "tree");
   t.Branch("eventNumber", &eventNumber);
   t.Branch("otherNumbers", &otherNumbers);
   for (Long64_t i = 0; i < nEvents; i++) {
      eventNumber = i;
      for (size_t j = 0; j < nElements; ++j)
         otherNumbers[j] = -2. * eventNumber + 3500 * j;
      t.Fill();
   }
   std::cout << t.GetEntries() << std::endl;
   // use Scan to show what otherNumbers[2] contains
   t.Scan("eventNumber:otherNumbers", "", "", nEvents);

   TTreeIndex firstIndex(&t, "otherNumbers[2]", "eventNumber");
   firstIndex.Print("2"); // before the fix: major was always a garbage value and events were wrongly sorted
   TTreeIndex secondIndex(&t, "otherNumbers", "eventNumber");
   secondIndex.Print("2"); // major is otherNumbers[0] since index not specified (fine before and after fix)
   return 0;
}
