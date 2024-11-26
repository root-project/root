// @(#)root/hist:$Id$
// Author: Rene Brun   14/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGraphTime.h"
#include "TVirtualPad.h"
#include "TH1.h"
#include "TROOT.h"
#include "TTimer.h"
#include "TObjArray.h"
#include "TSystem.h"

ClassImp(TGraphTime);

/** \class TGraphTime
    \ingroup Graphs
TGraphTime is used to draw a set of objects evolving with nsteps in time between tmin and tmax.
Each time step has a new list of objects. This list can be identical to
the list of objects in the previous steps, but with different attributes.
see example of use in $ROOTSYS/tutorials/visualisation/graphs/gtime.C
*/

////////////////////////////////////////////////////////////////////////////////
/// default constructor.

TGraphTime::TGraphTime()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Create a TGraphTime with nsteps in range [xmin,xmax][ymin,ymax]

TGraphTime::TGraphTime(Int_t nsteps, Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
   if (nsteps <= 0) {
      Warning("TGraphTime", "Number of steps %d changed to 100", nsteps);
      nsteps = 100;
   }
   fSleepTime = 0;
   fNsteps    = nsteps;
   fXmin      = xmin;
   fXmax      = xmax;
   fYmin      = ymin;
   fYmax      = ymax;
   fSteps     = new TObjArray(nsteps+1);
   fFrame = new TH1D("frame", "", 100, fXmin, fXmax);
   fFrame->SetMinimum(ymin);
   fFrame->SetMaximum(ymax);
   fFrame->SetStats(false);
}


////////////////////////////////////////////////////////////////////////////////
/// GraphTime default destructor.

TGraphTime::~TGraphTime()
{
   Animate(kFALSE);

   if (fSteps) {
      fSteps->Delete();
      delete fSteps;
      fSteps = nullptr;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// copy constructor.

TGraphTime::TGraphTime(const TGraphTime &gtime) : TNamed(gtime)
{
   fSleepTime = gtime.fSleepTime;
   fNsteps = gtime.fNsteps;
   fXmin = gtime.fXmin;
   fXmax = gtime.fXmax;
   fYmin = gtime.fYmin;
   fYmax = gtime.fYmax;
   fSteps = new TObjArray(fNsteps + 1);
   fFrame = new TH1D("frame", "", 100, fXmin, fXmax);
   fFrame->SetMinimum(fYmin);
   fFrame->SetMaximum(fYmax);
   fFrame->SetStats(false);
}

////////////////////////////////////////////////////////////////////////////////
/// Add one object to a time slot.
/// TGraphTime becomes the owner of this object.
/// object will be drawn with option

Int_t TGraphTime::Add(const TObject *obj, Int_t slot, Option_t *option)
{
   if (!fSteps) {
      fNsteps = 100;
      fSteps = new TObjArray(fNsteps+1);
   }
   if (slot < 0 || slot >= fNsteps)
      return -1;
   TList *list = (TList*)fSteps->UncheckedAt(slot);
   if (!list) {
      list = new TList();
      fSteps->AddAt(list,slot);
   }
   list->Add((TObject*)obj, option);
   return slot;
}

////////////////////////////////////////////////////////////////////////////////
/// Start animation of TGraphTime.
/// Triggers drawing of steps - but does not block macro execution which will continues

void TGraphTime::Animate(Bool_t enable)
{
   if (!enable) {
      fAnimateCnt = -1;
      if (fAnimateTimer) {
         fAnimateTimer->Stop();
         delete fAnimateTimer;
         fAnimateTimer = nullptr;
      }
      return;
   }

   if (!gPad) {
      gROOT->MakeDefCanvas();
      gPad->SetFillColor(41);
      gPad->SetFrameFillColor(19);
      gPad->SetGrid();
   }
   if (fFrame)
      fFrame->SetTitle(GetTitle());

   fAnimateCnt = 0;
   if (!fAnimateTimer) {
      fAnimateTimer = new TTimer(this, fSleepTime > 0 ? fSleepTime : 1);
      fAnimateTimer->Start();
   }

   Notify();
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this TGraphTime.
/// for each time step the list of objects added to this step are drawn.

void TGraphTime::Draw(Option_t *)
{
   if (!gPad) {
      gROOT->MakeDefCanvas();
      gPad->SetFillColor(41);
      gPad->SetFrameFillColor(19);
      gPad->SetGrid();
   }
   if (fFrame)
      fFrame->SetTitle(GetTitle());

   for (Int_t s = 0; s < fNsteps; s++) {
      if (DrawStep(s)) {
         gPad->Update();
         if (fSleepTime > 0)
            gSystem->Sleep(fSleepTime);
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Draw single step

Bool_t TGraphTime::DrawStep(Int_t nstep) const
{
   if (!fSteps)
      return kFALSE;

   auto list = static_cast<TList *>(fSteps->UncheckedAt(nstep));
   if (!list)
      return kFALSE;

   if (fFrame)
      gPad->Remove(fFrame);
   gPad->GetListOfPrimitives()->Clear();
   if (fFrame)
      gPad->Add(fFrame);

   auto lnk = list->FirstLink();
   while(lnk) {
      gPad->Add(lnk->GetObject(), lnk->GetAddOption());
      lnk = lnk->Next();
   }

   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Method used for implementing animation of TGraphTime

Bool_t TGraphTime::HandleTimer(TTimer *)
{
   if ((fAnimateCnt < 0) || !fSteps || !gPad)
      return kTRUE;

   if (fAnimateCnt > fSteps->GetLast())
      fAnimateCnt = 0;

   if (DrawStep(fAnimateCnt++))
      gPad->Update();

   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint all objects added to each time step

void TGraphTime::Paint(Option_t *)
{
   Error("Paint", "Not implemented, use Draw() instead");
}


////////////////////////////////////////////////////////////////////////////////
/// Save this object to filename as an animated gif file
/// if filename is specified it must be of the form xxx.gif
/// otherwise a file yyy.gif is produced where yyy is the object name

void TGraphTime::SaveAnimatedGif(const char *filename) const
{
   if (!gPad) {
      Error("SaveAnimatedGif", "Not possible to create animated GIF without gPad");
      return;
   }

   if (gPad->IsWeb()) {
      Error("SaveAnimatedGif", "Not possible to create animated GIF with web canvas");
      return;
   }

   TString farg = TString::Format("%s+", filename && *filename ? filename : GetName());

   for (Int_t s = 0; s < fNsteps; s++) {
      if (DrawStep(s))
         gPad->Print(farg.Data());
   }
}
