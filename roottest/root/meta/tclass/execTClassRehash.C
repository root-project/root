#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "THashTable.h"
#include "TError.h"
#include "TString.h"
#include "TSystem.h"

#include <stdio.h>

void LoadAllTClass()
{
   Int_t totalNumberOfClasses = gClassTable->Classes();
   for (Int_t i = 0; i < totalNumberOfClasses; i++) {

      // get class name
      const char *cname = gClassTable->Next();
      if (!cname)
         continue;

      // get class & filename - use TROOT::GetClass, as we also
      // want those classes without decl file name!
      TClass *classPtr = TClass::GetClass((const char *)cname, kTRUE);
      if (!classPtr)
         continue;
   }
}

TClass *FindTClassWithSameSlotAs(const char *name = "TQClass")
{
   // Load all libraries and instantiate all TClasses.
   gSystem->LoadAllLibraries();
   LoadAllTClass();

   TString cname(name);

   auto cap = gROOT->GetListOfClasses()->Capacity();

   auto target = cname.Hash();

   TIter next(gROOT->GetListOfClasses());
   TObject *o;
   while( (o = next()) ) {
      if ( o->Hash() % cap == target)
         return (TClass*)o;
   }
   return nullptr;
}

void VerifyMatch(TClass *cl, const char *name = "TQClass") {

   if (!cl) {
      auto rep = FindTClassWithSameSlotAs();
      Fatal("execTClassRehash","Need a proper class matching TQClass but have none so far, try using %s",
            rep ? rep->GetName() : " ... humm we can not find something to try");
   }

   auto cap = gROOT->GetListOfClasses()->Capacity();
   TString cname(name);

   if ( ( cl->Hash() % cap) != (cname.Hash() % cap )) {
      auto rep = FindTClassWithSameSlotAs();
      Fatal("execTClassRehash","The class %s no longer has the sanme slot as %s, try with %s",
            cl->GetName(), name,
            rep ? rep->GetName() : " ... humm we can not find something to try");
   }
}

int execTClassRehash()
{
   // Add a class to TQCloss slot ... so that adding TQClass to the list will
   // not change/reduce the average collision and thus lead to a Rehash if
   // TQClass is added during a Rehash.
   auto c = TClass::GetClass("TParameter<double>");

   // The test relies on "TParameter<double>" and "TQClass" having the same
   // slot in the ListOfClasses, so let's verify that we are still good to go.

   VerifyMatch(c);

   // If this test fail, we need to replace "TParameter<double>" with another
   // class.  To find a suitable class use the function FindTClassWithSameSlotAs

   // Now fill until we are close to a Rehash.
   c = TObject::Class();

   THashTable *h = dynamic_cast<THashTable*>(gROOT->GetListOfClasses());
   if (h == nullptr) {
      Fatal("execTClassRehash","The list of classes is no longer a THashTable:");
   }

   float usedSlots =  h->GetEntries() / h->AverageCollisions();
   float target = usedSlots * h->GetRehashLevel();
   // e+1 < hl * us
   // Rehash condition is:
   //      AverageCollisions() > fRehashLevel
   // With
   //      AverageCollisions = fEntries / fUsedSlots
   int i = 0;
   while( h->GetEntries() + 1 <= target) {
      // printf("%d\n",i);
      gROOT->GetListOfClasses()->Add(c);
      ++i;
   }
   // fprintf(stderr,"target is %f while entries is %d\n",target,h->GetEntries());

   auto goodsize = h->GetEntries();
   auto oldcap = h->Capacity();

   // Provoke a Rehash
   gROOT->GetListOfClasses()->Add(c);

   ++goodsize;
   auto newsize = h->GetEntries();
   auto newcap = h->Capacity();

   if (oldcap == newcap) {
      Fatal("execTClassRehash","The list of classes was not rehashing after the last insert. (capacity = %d",
            oldcap);
   }

   if (newsize != goodsize) {
      Error("execTClassRehash","The rehash of the list of classes changed the number of element old=%d new=%d",
            goodsize, newsize);
      return 1;
   }

   return 0;
}
