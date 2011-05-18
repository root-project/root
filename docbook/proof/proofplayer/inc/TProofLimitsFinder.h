// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn   19/04/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofLimitsFinder
#define ROOT_TProofLimitsFinder

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLimitsFinder                                                   //
//                                                                      //
// Class to find nice axis limits and synchronize them between workers  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_THLimitsFinder
#include "THLimitsFinder.h"
#endif

class TH1;
class TString;

class TProofLimitsFinder : public THLimitsFinder {

public:
   TProofLimitsFinder() { }
   virtual ~TProofLimitsFinder() { }
   virtual Int_t FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax);
   virtual Int_t FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax);
   virtual Int_t FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax, Axis_t zmin, Axis_t zmax);

   static void AutoBinFunc(TString& key,
                           Double_t& xmin, Double_t& xmax,
                           Double_t& ymin, Double_t& ymax,
                           Double_t& zmin, Double_t& zmax);

   ClassDef(TProofLimitsFinder,0)  //Find and communicate best axis limits
};

#endif
