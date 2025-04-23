#include "Rtypes.h"

class Empty {
  ClassDef(Empty,0);
};

class Top {
public: int i;
};

class Bottom: public Top, public Empty {
public:
  int j;
};

#include <vector>

class Holder {
public:
   Bottom fObject;
   std::vector<Bottom> fColl;
};

#include "TObjArray.h"
#include "TBranch.h"

int countBranches(TObjArray *blist)
{
   int result = 0;
   TIter iter(blist);
   TBranch *b;
   while ( (b = (TBranch*)iter()) ) {
      ++result;
      result += countBranches(b->GetListOfBranches());
   }
   return result;
}

#include "TTree.h"
#include "TError.h"

int execEmptyBase()
{
   Bottom b;
   auto t = new TTree("T","T");
   t->Branch("b",&b);
   Holder h;
   t->Branch("h",&h);
   int result = countBranches(t->GetListOfBranches());
   if (result != 12) {
      Error("base3","The number of branch is %d rather than the 12 expected branches",result);
      t->Print();
      return 1;
   }
   return 0;
}
