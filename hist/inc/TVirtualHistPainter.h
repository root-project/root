// @(#)root/hist:$Name:  $:$Id: TVirtualHistPainter.h,v 1.3 2001/12/09 17:33:49 brun Exp $
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualHistPainter
#define ROOT_TVirtualHistPainter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualHistPainter                                                  //
//                                                                      //
// Abstract base class for Histogram painters                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TClass
#include "TClass.h"
#endif

class TH1;
class TF1;
class TObjArray;
class TPaveStats;

class TVirtualHistPainter : public TObject {


private:
   static TClass   *fgPainter; //Pointer to class painter

public:
    TVirtualHistPainter();
    virtual ~TVirtualHistPainter();
    virtual Int_t      DistancetoPrimitive(Int_t px, Int_t py) = 0;
    virtual void       DrawPanel() = 0;
    virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py) = 0;
    virtual void       FitPanel() = 0;
    virtual char      *GetObjectInfo(Int_t px, Int_t py) const = 0;
    virtual TObjArray *GetStack() const = 0;
    virtual TPaveStats*GetStats() const = 0;
    virtual void       Paint(Option_t *option="") = 0;
    virtual void       PaintStat(Int_t dostat, TF1 *fit) = 0;
    virtual void       SetHistogram(TH1 *h) = 0;
    virtual void       SetStack(TObjArray *stack) = 0;

   static  TVirtualHistPainter *HistPainter(TH1 *obj);
   static void      SetPainter(const char *painter);

    ClassDef(TVirtualHistPainter,0)  //Abstract interface for histogram painters
};

#endif
