// @(#)root/hist:$Name:  $:$Id: THLimitsFinder.h,v 1.1 2002/01/15 10:22:27 brun Exp $
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_THLimitsFinder
#define ROOT_THLimitsFinder


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THLimitsFinder                                                       //
//                                                                      //
// class to find nice axis limits                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TH1;

class THLimitsFinder : public TObject {

protected:
    static THLimitsFinder *fgLimitsFinder;   //!Pointer to hist limits finder

public:
    THLimitsFinder();
    virtual ~THLimitsFinder();
    virtual Int_t      FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax);
    virtual Int_t      FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax);
    virtual Int_t      FindGoodLimits(TH1 *h, Axis_t xmin, Axis_t xmax, Axis_t ymin, Axis_t ymax, Axis_t zmin, Axis_t zmax);

    static  void       Optimize(Double_t A1,  Double_t A2,  Int_t nold
                       ,Double_t &BinLow, Double_t &BinHigh, Int_t &nbins, Double_t &BWID, Option_t *option="");
    static void        OptimizeLimits(Int_t nbins, Int_t &newbins, Axis_t &xmin, Axis_t &xmax, Bool_t isInteger);
    static THLimitsFinder *GetLimitsFinder();
    static  void       SetLimitsFinder(THLimitsFinder *finder);

    ClassDef(THLimitsFinder,0)  //Class to find best axis limits
};

#endif
