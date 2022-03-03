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

class TGraphTime : public TNamed {

protected:

   Int_t              fSleepTime; ///< Time (msec) to wait between time steps
   Int_t              fNsteps;    ///< Number of time steps
   Double_t           fXmin;      ///< Minimum for X axis
   Double_t           fXmax;      ///< Maximum for X axis
   Double_t           fYmin;      ///< Minimum for Y axis
   Double_t           fYmax;      ///< Maximum for Y axis
   TObjArray         *fSteps;     ///< Array of TLists for each time step
   TH1               *fFrame;     ///< TH1 object used for the pad range

public:

   TGraphTime();
   TGraphTime(Int_t nsteps, Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax);
   TGraphTime(const TGraphTime &gr);
   virtual ~TGraphTime();

   virtual Int_t   Add(const TObject *obj, Int_t slot, Option_t *option="");
   virtual void    Draw(Option_t *chopt="");
   TObjArray      *GetSteps() const {return fSteps;}
   virtual void    Paint(Option_t *chopt="");
   virtual void    SaveAnimatedGif(const char *filename="") const;
   virtual void    SetSleepTime(Int_t stime=0) {fSleepTime = stime;}

   ClassDef(TGraphTime,1)  //An array of objects evolving with time
};

#endif
