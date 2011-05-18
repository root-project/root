// @(#)root/hist:$Id$
// Author: Olivier Couet 20/05/08

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualGraphPainter
#define ROOT_TVirtualGraphPainter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGraphPainter                                                 //
//                                                                      //
// Abstract base class for Graph painters                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGraph;
class TF1;

class TVirtualGraphPainter : public TObject {

private:
   static TVirtualGraphPainter   *fgPainter; //Pointer to class painter

public:
   TVirtualGraphPainter() { }
   virtual ~TVirtualGraphPainter() { }

   virtual Int_t DistancetoPrimitiveHelper(TGraph *theGraph, Int_t px, Int_t py) = 0;
   virtual void  DrawPanelHelper(TGraph *theGraph) = 0;
   virtual void  ExecuteEventHelper(TGraph *theGraph, Int_t event, Int_t px, Int_t py) = 0;
   virtual char *GetObjectInfoHelper(TGraph *theGraph, Int_t px, Int_t py) const = 0;
   virtual void  PaintHelper(TGraph *theGraph, Option_t *option) = 0;
   virtual void  PaintGraph(TGraph *theGraph, Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt) = 0;
   virtual void  PaintGrapHist(TGraph *theGraph, Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt) = 0;
   virtual void  PaintStats(TGraph *theGraph, TF1 *fit) = 0;

   static TVirtualGraphPainter *GetPainter();
   static void                  SetPainter(TVirtualGraphPainter *painter);

   ClassDef(TVirtualGraphPainter,0)  //Abstract interface for histogram painters
};

#endif
