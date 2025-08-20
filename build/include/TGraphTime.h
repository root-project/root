// @(#)root/hist:$Id$
// Author: Rene Brun 13/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphTime
#define ROOT_TGraphTime


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphTime                                                           //
//                                                                      //
// An array of objects evolving with time                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TH1;
class TObjArray;
class TTimer;

class TGraphTime : public TNamed {

protected:
   Int_t fSleepTime = 0;        ///< Time (msec) to wait between time steps
   Int_t fNsteps = 0;           ///< Number of time steps
   Double_t fXmin = 0.;         ///< Minimum for X axis
   Double_t fXmax = 0.;         ///< Maximum for X axis
   Double_t fYmin = 0.;         ///< Minimum for Y axis
   Double_t fYmax = 0.;         ///< Maximum for Y axis
   TObjArray *fSteps = nullptr; ///< Array of TLists for each time step
   TH1 *fFrame = nullptr;       ///< TH1 object used for the pad range
   Int_t fAnimateCnt = -1;      ///<! counter used in Animate() method
   TTimer *fAnimateTimer = nullptr; ///<! timer to implement animation

   Bool_t DrawStep(Int_t nstep) const;

public:

   TGraphTime();
   TGraphTime(Int_t nsteps, Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax);
   TGraphTime(const TGraphTime &gr);
   ~TGraphTime() override;

   virtual Int_t Add(const TObject *obj, Int_t slot, Option_t *option = "");
   void Animate(Bool_t enable = kTRUE);
   void Draw(Option_t *chopt = "") override;
   TObjArray *GetSteps() const { return fSteps; }
   Bool_t HandleTimer(TTimer *) override;
   void Paint(Option_t *chopt = "") override;
   virtual void SaveAnimatedGif(const char *filename = "") const;
   virtual void SetSleepTime(Int_t stime = 0) { fSleepTime = stime; }

   ClassDefOverride(TGraphTime,1)  //An array of objects evolving with time
};

#endif
