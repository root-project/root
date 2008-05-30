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

#ifndef ROOT_Object
#include "TVirtualGraphPainter.h"
#endif

class TGraph;
class TGraph2D;
class TGraphDelaunay;
class TList;
class TF1;

class TGraphPainter : public TVirtualGraphPainter {

protected:

   Double_t       *fX;        //!Pointer to fGraph2D->fX or fGraph->fX
   Double_t       *fY;        //!Pointer to fGraph2D->fY or fGraph->fY
   Double_t       *fZ;        //!Pointer to fGraph2D->fZ
   Double_t       *fEX;       //!Pointer to fGraph->fEX (for TGraphErrors)
   Double_t       *fEY;       //!Pointer to fGraph->fEY (for TGraphErrors)
   Double_t       *fEXlow;    //!Pointer to fGraph->fEXlow (for TGraphAsymmErrors and TGraphBentErrors)
   Double_t       *fEXhigh;   //!Pointer to fGraph->fEXhigh (for TGraphAsymmErrors and TGraphBentErrors)
   Double_t       *fEYlow;    //!Pointer to fGraph->fEYlow (for TGraphAsymmErrors and TGraphBentErrors)
   Double_t       *fEYhigh;   //!Pointer to fGraph->fEYhigh (for TGraphAsymmErrors and TGraphBentErrors)
   Double_t       *fEXlowd;   //!Pointer to fGraph->fEXlowd (for TGraphBentErrors)
   Double_t       *fEXhighd;  //!Pointer to fGraph->fEXhighd (for TGraphBentErrors)
   Double_t       *fEYlowd;   //!Pointer to fGraph->fEYlowd (for TGraphBentErrors)
   Double_t       *fEYhighd;  //!Pointer to fGraph->fEYhighd (for TGraphBentErrors)
   Double_t       *fXN;       //!Pointer to fDelaunay->fXN
   Double_t       *fYN;       //!Pointer to fDelaunay->fYN
   Double_t        fXNmin;    //!Equal to fDelaunay->fXNmin
   Double_t        fXNmax;    //!Equal to fDelaunay->fXNmax
   Double_t        fYNmin;    //!Equal to fDelaunay->fYNmin
   Double_t        fYNmax;    //!Equal to fDelaunay->fYNmax
   Double_t        fXmin;     //!
   Double_t        fXmax;     //!
   Double_t        fYmin;     //! fGraph2D->fHistogram limits
   Double_t        fYmax;     //!
   Double_t        fZmin;     //!
   Double_t        fZmax;     //!
   Int_t           fNpoints;  //!Equal to fGraph2D->fNpoints or fGraph->fNpoints
   Int_t           fNdt;      //!Equal to fDelaunay->fNdt
   Int_t          *fPTried;   //!Pointer to fDelaunay->fPTried
   Int_t          *fNTried;   //!Pointer to fDelaunay->fNTried
   Int_t          *fMTried;   //!Pointer to fDelaunay->fMTried

   TGraphDelaunay *fDelaunay; // Pointer to the TGraphDelaunay to be painted
   TGraph2D       *fGraph2D;  // Pointer to the TGraph2D in fDelaunay
   TGraph         *fGraph;    // Pointer to graph to paint

   void     FindTriangles();
   void     PaintLevels(Int_t *T, Double_t *x, Double_t *y, Int_t nblev=0, Double_t *glev=0);
   void     PaintPolyMarker0(Int_t n, Double_t *x, Double_t *y);

public:

   TGraphPainter();
   TGraphPainter(TGraphDelaunay *gd);

   virtual ~TGraphPainter();

   void               ComputeLogs(Int_t npoints, Int_t opt);
   virtual Int_t      DistancetoPrimitive(Int_t px, Int_t py);
   virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TList             *GetContourList(Double_t contour);
   virtual char      *GetObjectInfo(Int_t px, Int_t py) const;
   void               Paint(Option_t *option);
   void               PaintContour(Option_t *option);
   virtual void       PaintGraph(Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt);
   virtual void       PaintGrapHist(Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt);
   void               PaintGraph2D(Option_t *option);
   void               PaintGraphAsymmErrors(Option_t *option);
   void               PaintGraphBentErrors(Option_t *option);
   void               PaintGraphErrors(Option_t *option);
   void               PaintGraphSimple(Option_t *option);
   void               PaintPolyLine(Option_t *option);
   void               PaintPolyLineHatches(Int_t n, const Double_t *x, const Double_t *y);
   void               PaintPolyMarker(Option_t *option);
   void               PaintStats(TF1 *fit);
   void               PaintTriangles(Option_t *option);
   void               SetGraph(TGraph *g);
   void               Smooth(Int_t npoints, Double_t *x, Double_t *y, Int_t drawtype);
   void               Zero(Int_t &k,Double_t AZ,Double_t BZ,Double_t E2,Double_t &X,Double_t &Y,Int_t maxiterations);

   ClassDef(TGraphPainter,0)  // TGraph painter
};

#endif
