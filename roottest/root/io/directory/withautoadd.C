#include "TDirectory.h"
#include <iostream>
#include "TClass.h"
using namespace std;

class Top {
public:
   Top() : name("Top") {}
   Top(const char *n) : name(n) {}

   TString name;
   TDirectory *fDirectory;
   void DirectoryAutoAdd(TDirectory *where) {
      cout << "DirectoryAutoAdd for " << name << " with ";
      if (where) {
         cout << where->GetName() << endl;
      } else {
         cout << "null\n";
      }
   }
};

class Bottom : public TNamed, public Top {
public:
   Bottom() : Top("Bottom") {}

   ClassDefOverride(Bottom, 1);
};

void check(const char *what)
{
   if (what==0) return;

   TClass * cl = TClass::GetClass(what);

   if (!cl) return;
   ROOT::DirAutoAdd_t func = cl->GetDirectoryAutoAdd();
   if (func) {
      cout << "Found wrapper for " << cl->GetName() << endl;
      void *obj = cl->New();
      func(obj,gDirectory);
   } else {
      cout << "Missing wrapper for " << cl->GetName() << endl;
   }
}

void withautoadd()
{
   check("Top");
   check("Bottom");
}
