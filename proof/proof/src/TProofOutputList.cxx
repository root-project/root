// @(#)root/proofplayer:$Id$
// Author: G. Ganis   04/08/2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofOutputList
\ingroup proofkernel

Derivation of TList with an overload of ls() and Print() allowing to filter
out some of the variables

*/

#include "TObjString.h"
#include "TProof.h"
#include "TProofOutputList.h"
#include "TRegexp.h"
#include "TString.h"

ClassImp(TProofOutputList);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TProofOutputList::TProofOutputList(const char *dontshow) : TList()
{
   fDontShow = new TList();
   TString regs(dontshow), reg;
   Int_t from = 0;
   while (regs.Tokenize(reg, from, ",")) {
      fDontShow->Add(new TObjString(reg));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TProofOutputList::~TProofOutputList()
{
   fDontShow->SetOwner(kTRUE);
   SafeDelete(fDontShow);
}

////////////////////////////////////////////////////////////////////////////////
/// Attach to list 'alist'

void TProofOutputList::AttachList(TList *alist)
{
   if (!alist) return;

   if (GetSize() > 0) Clear();

   TIter nxo(alist);
   TObject *obj = 0;
   while ((obj = nxo())) { Add(obj); }
   SetOwner(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// List the content of the list

void TProofOutputList::ls(Option_t *option) const
{
   TString opt(option);
   opt.ToUpper();
   if (opt.BeginsWith("ALL")) {
      opt.Remove(0,3);
      TList::ls(opt);
   } else {
      TIter nxos(fDontShow);
      TObjString *os = 0;
      TList doShow;
      doShow.SetOwner(kFALSE);

      Bool_t hasmissing = kFALSE;
      TIter nxo(this);
      TObject *obj = 0;
      while ((obj = nxo())) {
         TString s = obj->GetName();
         if (s == kPROOF_MissingFiles) {
            TList *mf = dynamic_cast<TList *> (obj);
            if (mf && mf->GetSize() > 0) hasmissing = kTRUE;
         } else {
            nxos.Reset();
            Bool_t doadd = kTRUE;
            while ((os = (TObjString *) nxos())) {
               TRegexp rg(os->GetName(), kTRUE);
               if (s.Index(rg) != kNPOS) {
                  doadd = kFALSE;
                  break;
               }
            }
            if (doadd) doShow.Add(obj);
         }
      }
      doShow.ls(option);
      // Notify if missing files were found
      if (hasmissing)
         Printf(" +++ Missing files list not empty: use ShowMissingFiles to display it +++");
   }
   // Done
   return;
}


////////////////////////////////////////////////////////////////////////////////
/// Print the content of the list

void TProofOutputList::Print(Option_t *option) const
{
   TString opt(option);
   opt.ToUpper();
   if (opt.BeginsWith("ALL")) {
      opt.Remove(0,3);
      TList::Print(opt);
   } else {
      TIter nxos(fDontShow);
      TObjString *os = 0;
      TList doShow;
      doShow.SetOwner(kFALSE);

      Bool_t hasmissing = kFALSE;
      TIter nxo(this);
      TObject *obj = 0;
      while ((obj = nxo())) {
         TString s = obj->GetName();
         if (s == kPROOF_MissingFiles) {
            TList *mf = dynamic_cast<TList *> (obj);
            if (mf && mf->GetSize() > 0) hasmissing = kTRUE;
         } else {
            nxos.Reset();
            Bool_t doadd = kTRUE;
            while ((os = (TObjString *) nxos())) {
               TRegexp rg(os->GetName());
               if (s.Index(rg) != kNPOS) {;
                  doadd = kFALSE;
                  break;
               }
            }
            if (doadd) doShow.Add(obj);
         }
      }
      doShow.Print(option);
      // Notify if missing files were found
      if (hasmissing)
         Printf(" +++ Missing files list not empty: use ShowMissingFiles to display it +++");
   }
   // Done
   return;
}
