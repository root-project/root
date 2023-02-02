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
#include "TObjArray.h"
#include "TSystem.h"

ClassImp(TGraphTime);

/** \class TGraphTime
    \ingroup Graphs
TGraphTime is used to draw a set of objects evolving with nsteps in time between tmin and tmax.
Each time step has a new list of objects. This list can be identical to
the list of objects in the previous steps, but with different attributes.
see example of use in $ROOTSYS/tutorials/graphs/gtime.C
*/

////////////////////////////////////////////////////////////////////////////////
/// default constructor.

TGraphTime::TGraphTime(): TNamed()
{
   fSleepTime = 0;
   fNsteps    = 0;
   fXmin      = 0;
   fXmax      = 1;
   fYmin      = 0;
   fYmax      = 1;
   fSteps     = 0;
   fFrame     = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a TGraphTime with nsteps in range [xmin,xmax][ymin,ymax]

TGraphTime::TGraphTime(Int_t nsteps, Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
      :TNamed()
{
   if (nsteps <= 0) {
      Warning("TGraphTime", "Number of steps %d changed to 100",nsteps);
      nsteps = 100;
   }
   fSleepTime = 0;
   fNsteps    = nsteps;
   fXmin      = xmin;
   fXmax      = xmax;
   fYmin      = ymin;
   fYmax      = ymax;
   fSteps     = new TObjArray(nsteps+1);
   fFrame     = new TH1D("frame","",100,fXmin,fXmax);
   fFrame->SetMinimum(ymin);
   fFrame->SetMaximum(ymax);
   fFrame->SetStats(0);
}


////////////////////////////////////////////////////////////////////////////////
/// GraphTime default destructor.

TGraphTime::~TGraphTime()
{
   if (!fSteps) return;
   fSteps->Delete();
   delete fSteps; fSteps=0;
}


////////////////////////////////////////////////////////////////////////////////
/// copy constructor.

TGraphTime::TGraphTime(const TGraphTime &gtime) : TNamed(gtime)
{
   fSleepTime = gtime.fSleepTime;
   fNsteps    = gtime.fNsteps;
   fXmin      = gtime.fXmin;
   fXmax      = gtime.fXmax;
   fYmin      = gtime.fYmin;
   fYmax      = gtime.fYmax;
   fSteps     = new TObjArray(fNsteps+1);
   fFrame     = new TH1D("frame","",100,fXmin,fXmax);
   fFrame->SetMinimum(fYmin);
   fFrame->SetMaximum(fYmax);
   fFrame->SetStats(0);
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
   if (slot < 0 || slot >= fNsteps) return -1;
   TList *list = (TList*)fSteps->UncheckedAt(slot);
   if (!list) {
      list = new TList();
      fSteps->AddAt(list,slot);
   }
   list->Add((TObject*)obj, option);
   return slot;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this TGraphTime.
/// for each time step the list of objects added to this step are drawn.

void TGraphTime::Draw(Option_t *option)
{
   if (!gPad) {
      gROOT->MakeDefCanvas();
      gPad->SetFillColor(41);
      gPad->SetFrameFillColor(19);
      gPad->SetGrid();
   }
   if (fFrame) {
      fFrame->SetTitle(GetTitle());
      fFrame->Draw();
   }
   Paint(option);

}

////////////////////////////////////////////////////////////////////////////////
/// Paint all objects added to each time step

void TGraphTime::Paint(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   TObject *frame = gPad->GetPrimitive("frame");
   TList *list = 0;
   TObjLink *lnk;

   for (Int_t s=0;s<fNsteps;s++) {
      list = (TList*)fSteps->UncheckedAt(s);
      if (list) {
         gPad->GetListOfPrimitives()->Remove(frame);
         gPad->GetListOfPrimitives()->Clear();
         if (frame) gPad->GetListOfPrimitives()->Add(frame);
         lnk = list->FirstLink();
         while(lnk) {
            TObject *obj = lnk->GetObject();
            obj->Draw(lnk->GetAddOption());
            lnk = lnk->Next();
         }
         gPad->Update();
         if (fSleepTime > 0) gSystem->Sleep(fSleepTime);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save this object to filename as an animated gif file
/// if filename is specified it must be of the form xxx.gif
/// otherwise a file yyy.gif is produced where yyy is the object name

void TGraphTime::SaveAnimatedGif(const char *filename) const
{
   TObject *frame = gPad->GetPrimitive("frame");
   TList *list = nullptr;
   TObjLink *lnk;

   for (Int_t s=0;s<fNsteps;s++) {
      list = (TList*)fSteps->UncheckedAt(s);
      if (list) {
         gPad->GetListOfPrimitives()->Remove(frame);
         gPad->GetListOfPrimitives()->Clear();
         if (frame) gPad->GetListOfPrimitives()->Add(frame);
         lnk = list->FirstLink();
         while(lnk) {
            TObject *obj = lnk->GetObject();
            obj->Draw(lnk->GetAddOption());
            lnk = lnk->Next();
         }
         gPad->Update();
         if (filename && strlen(filename) > 0)
            gPad->Print(Form("%s+", filename));
         else
            gPad->Print(Form("%s+", GetName()));
         if (fSleepTime > 0)
            gSystem->Sleep(fSleepTime);
      }
   }
}
