#include <TNamed.h>
#include <unordered_set>

using std::unordered_set;

typedef unordered_set<Long64_t> uset_t;

class SetProblem : public TNamed
{
protected:

   unordered_set<int> fIntBad;
   unordered_set<Int_t> fInt;

   unordered_set<Long64_t> fNull;
   unordered_set<long long> fStraight;


   unordered_set<ULong64_t> fUNull;

   uset_t fGood;
   std::unordered_set<Int_t> fInt2;

   std::unordered_set<Long64_t> sfNull;
   std::unordered_set<ULong64_t> sfUNull;

public:
   SetProblem() {}
   SetProblem(const char * name) : TNamed(name, "") {}

   ClassDef(SetProblem,1) // Abstract base column of TFTable

};

#ifdef __MAKECINT__
#pragma link C++ typedef set_t;
#pragma link C++ class SetProblem+;
#endif

