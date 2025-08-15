#include "TTree.h"
#include "TTreeIndex.h"
#include <vector>
#include <iostream>

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
   firstIndex.Print("2"); // wasn't working before: major was always a garbage value
   TTreeIndex secondIndex(&t, "otherNumbers", "eventNumber");
   secondIndex.Print("2"); // major is always otherNumbers[0] since index not specified (before and after fix)
   return 0;
}
