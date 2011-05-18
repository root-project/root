// @(#)root/eve:$Id: text_test.C 26717 2008-12-07 22:07:55Z matevz $
// Author: Matevz Tadel

// Test signals from TEveSelection class.

#include "TEveSelection.h" // To enforce auto-loading of libEve.

class SigTestSpitter
{
   TEveSelection *fSel;
   TString        fPrefix;

public:
   SigTestSpitter(TEveSelection* sel, const TString& prefix) :
      fSel(sel), fPrefix(prefix)
   {
      fSel->Connect("SelectionAdded(TEveElement*)", "SigTestSpitter", this, "Added(TEveElement*)");
      fSel->Connect("SelectionRemoved(TEveElement*)", "SigTestSpitter", this, "Removed(TEveElement*)");
      fSel->Connect("SelectionCleared()", "SigTestSpitter", this, "Cleared()");
   }
   ~SigTestSpitter()
   {
      fSel->Disconnect("SelectionAdded(TEveElement*)", this, "Added(TEveElement*)");
      fSel->Disconnect("SelectionRemoved(TEveElement*)", this, "Removed(TEveElement*)");
      fSel->Disconnect("SelectionCleared()", this, "Cleared()");
   }
   // ----------------------------------------------------------------
   void Added(TEveElement* el)
   {
      printf("%s Added 0x%lx '%s'\n", fPrefix.Data(), el, el ? el->GetElementName() : "");
   }
   void Removed(TEveElement* el)
   {
      printf("%s Removed 0x%lx '%s'\n", fPrefix.Data(), el, el ? el->GetElementName() : "");
   }
   void Cleared()
   {
      printf("%s Cleared'\n", fPrefix.Data());
   }
};

void selection_sigs()
{
   TEveManager::Create();

   new SigTestSpitter(gEve->GetSelection(), "Selection");
   new SigTestSpitter(gEve->GetHighlight(), "Highlight");
}
