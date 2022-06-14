// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   28/10/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TDrawFeedback
\ingroup proofkernel

Utility class to draw objects in the feedback list during queries.
Draws histograms in separated canvases and user-defined objects via
Draw(). Users requiring advanced treatment should implement their
own version following this example. See also TStatsFeedback.

*/

#include "TDrawFeedback.h"

#include "THashList.h"
#include "TObjString.h"
#include "TProof.h"
#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TError.h"
#include "TSeqCollection.h"
#include "TVirtualPad.h"
#include "TProofDebug.h"

ClassImp(TDrawFeedback);


////////////////////////////////////////////////////////////////////////////////
/// Constructor

TDrawFeedback::TDrawFeedback(TProof *proof, TSeqCollection *names)
  : fAll(kFALSE)
{
   fNames = new THashList;
   fNames->SetOwner();

   if (proof == 0) proof = gProof;

   TProof *p = dynamic_cast<TProof*>(proof);
   if (p == 0) {
      Error("TDrawFeedback","no valid proof session found");
      return;
   }
   fProof = p;
   fName = fProof->GetSessionTag();

   Bool_t ok = proof->Connect("Feedback(TList*)", "TDrawFeedback",
                  this, "Feedback(TList*)");

   if ( !ok ) {
      Error("TDrawFeedback","Connect() failed");
      return;
   }

   if (names != 0) {
      TIter next(names);
      TObjString *name;
      while((name = dynamic_cast<TObjString*>(next())) != 0) {
         fNames->Add(new TNamed(name->GetName(),""));
      }
   } else {
      fAll = kTRUE;
   }
   fOption = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TDrawFeedback::~TDrawFeedback()
{
   delete fNames;

   // Required since we overload TObject::Hash.
   ROOT::CallRecursiveRemoveIfNeeded(*this);

   fProof->Disconnect("Feedback(TList*)", this, "Feedback(TList*");
}

////////////////////////////////////////////////////////////////////////////////
/// Display feedback

void TDrawFeedback::Feedback(TList *objs)
{
   TSeqCollection *canvases = gROOT->GetListOfCanvases();
   TVirtualPad *save = gPad;

   PDB(kFeedback,1) Info("Feedback","%d Objects", objs->GetSize());

   TIter next(objs);
   TObject *o;
   while( (o = next()) )
   {
      TString name = o->GetName();
      if (fAll || fNames->FindObject(name.Data())) {

         if (TH1 *h = dynamic_cast<TH1*>(o)) {

            // Basic service provided fro histograms, each one drawn in
            // a separate canvas named '<histogram_name>_canvas'

            name += "_canvas";

            TVirtualPad *p = (TVirtualPad*) canvases->FindObject(name.Data());

            if ( p == 0 ) {
               gROOT->MakeDefCanvas();
               gPad->SetName(name);
               PDB(kFeedback,2) Info("Feedback","Created canvas %s", name.Data());
            } else {
               p->cd();
               PDB(kFeedback,2) Info("Feedback","Used canvas %s", name.Data());
            }

            h->DrawCopy(fOption);
            gPad->Update();

         } else {

            // Call the Draw method of the object; this is intended for user-defined
            // objects handling their canvas needs inside Draw() as needed
            o->Draw();
         }

      }
   }

   if (save != 0) {
      save->cd();
   } else {
      gPad = 0;
   }
}
