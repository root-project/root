#include <TNamed.h>
#include <set>

using std::set;

typedef set<Long64_t> set_t;

class SetProblem : public TNamed
{
protected:

   set<int> fIntBad;
   set<Int_t> fInt;

   set<Long64_t> fNull;
   set<long long> fStraight;


   set<ULong64_t> fUNull;

   set_t fGood;
   std::set<Int_t> fInt2;

   std::set<Long64_t> sfNull;
   std::set<ULong64_t> sfUNull;

public:
   SetProblem() {}
   SetProblem(const char * name) : TNamed(name, "") {}

   ClassDefOverride(SetProblem,1) // Abstract base column of TFTable

};

#ifdef __MAKECINT__
#pragma link C++ typedef set_t;
#pragma link C++ class SetProblem+;
#endif

/* Issue
   when set<long long> is seen first, it's TClass become a 2nd TClass.

   If std::set<ULong64_t> is used we end up with set<long long>
*/
