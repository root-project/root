// @(#)root/histpainter:$Id: TGraphPainter.h,v 1.00
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphPainter
#define ROOT_TGraphPainter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphPainter                                                        //
//                                                                      //
// helper class to draw graphs                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGraphPainter.h"

class TGraph;
class TF1;

class TGraphPainter : public TVirtualGraphPainter {

public:

   TGraphPainter();

   ~TGraphPainter() override;

   void           ComputeLogs(Int_t npoints, Int_t opt);
   Int_t  DistancetoPrimitiveHelper(TGraph *theGraph, Int_t px, Int_t py) override;
   void   DrawPanelHelper(TGraph *theGraph) override;
   void   ExecuteEventHelper(TGraph *theGraph, Int_t event, Int_t px, Int_t py) override;
   char  *GetObjectInfoHelper(TGraph *theGraph, Int_t px, Int_t py) const override;
   virtual Int_t  GetHighlightPoint(TGraph *theGraph) const;
   virtual void   HighlightPoint(TGraph *theGraph, Int_t hpoint, Int_t distance);
   virtual void   PaintHighlightPoint(TGraph *theGraph, Option_t *option);
   void           PaintHelper(TGraph *theGraph, Option_t *option) override;
   void   PaintGraph(TGraph *theGraph, Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt) override;
   void   PaintGrapHist(TGraph *theGraph, Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt) override;
   void           PaintGraphAsymmErrors(TGraph *theGraph, Option_t *option);
   void           PaintGraphMultiErrors(TGraph *theGraph, Option_t *option);
   void           PaintGraphBentErrors(TGraph *theGraph, Option_t *option);
   void           PaintGraphErrors(TGraph *theGraph, Option_t *option);
   void           PaintGraphPolar(TGraph *theGraph, Option_t *option);
   void           PaintGraphQQ(TGraph *theGraph, Option_t *option);
   void           PaintGraphReverse(TGraph *theGraph, Option_t *option);
   void           PaintGraphSimple(TGraph *theGraph, Option_t *option);
   void           PaintPolyLineHatches(TGraph *theGraph, Int_t n, const Double_t *x, const Double_t *y);
   void           PaintStats(TGraph *theGraph, TF1 *fit) override;
   void   SetHighlight(TGraph *theGraph) override;
   void           Smooth(TGraph *theGraph, Int_t npoints, Double_t *x, Double_t *y, Int_t drawtype);
   static void    SetMaxPointsPerLine(Int_t maxp=50);

protected:

   static Int_t   fgMaxPointsPerLine;  //Number of points per chunks' line when drawing a graph.

   ClassDefOverride(TGraphPainter,0)  // TGraph painter
};

#endif
