// @(#)root/proof:$Name:  $:$Id: TDrawFeedback.cxx,v 1.2 2004/07/09 01:34:51 rdm Exp $
// Author: Maarten Ballintijn   28/10/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDrawFeedback                                                        //
//                                                                      //
// Utility class to draw objects in the feedback list during queries.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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


ClassImp(TDrawFeedback)


//______________________________________________________________________________
TDrawFeedback::TDrawFeedback(TVirtualProof *proof, TSeqCollection *names)
  : fAll(kFALSE)
{
   if (proof == 0) proof = gProof;

   TProof *p = dynamic_cast<TProof*>(proof);
   if (p == 0) {
      Error("TDrawFeedback","no valid proof session found");
      return;
   }

   proof->Connect("Feedback(TList *objs)", "TDrawFeedback",
                  this, "Feedback(TList *objs)");

   fNames = new THashList;

   if (names != 0) {
      TIter next(names);
      TObjString *name;
      while((name = dynamic_cast<TObjString*>(next())) != 0) {
         fNames->Add(new TObjString(*name));
      }
   } else {
      fAll = kTRUE;
   }
}


//______________________________________________________________________________
TDrawFeedback::~TDrawFeedback()
{
   delete fNames;
}


//______________________________________________________________________________
void TDrawFeedback::Feedback(TList *objs)
{
   TSeqCollection *canvases = gROOT->GetListOfCanvases();
   TVirtualPad *save = gPad;

   TIter next(objs);
   TObject *o;
   while( (o = next()) )
   {
      TString name = o->GetName();
      if ( (fAll || fNames->FindObject(name.Data())) && o->InheritsFrom("TH1")) {

         TH2 *h2 = dynamic_cast<TH2*>(o);
         if (h2 != 0) {
            h2->SetMarkerStyle(4);
         } else {
            TH1 *h = dynamic_cast<TH1*>(o);
            h->SetMinimum(0);
         }

         name += "_canvas";

         TVirtualPad *p = (TVirtualPad*) canvases->FindObject(name.Data());

         if ( p == 0 ) {
            (gROOT->GetMakeDefCanvas())();
            gPad->SetName(name);
            ((TH1*)o)->DrawCopy();
            Info("Feedback","Created canvas %s", name.Data());
         } else {
            p->cd();
            ((TH1*)o)->DrawCopy();
            Info("Feedback","Used canvas %s", name.Data());
         }
         gPad->Update();
      }
   }

   if (save != 0) {
      save->cd();
   } else {
      gPad = 0;
   }
}
