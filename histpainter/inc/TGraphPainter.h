// @(#)root/histpainter:$Name:  $:$Id: TGraphPainter.h,v 1.00
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

#ifndef ROOT_TGraph2D
#include "TGraph2D.h"
#endif
#ifndef ROOT_TGraph
#include "TGraph.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TGraphDelaunay
#include "TGraphDelaunay.h"
#endif

class TView;

class TGraphPainter : public TObject {

protected:

   Double_t   *fX;            //!Pointer to fGraph2D->fX
   Double_t   *fY;            //!Pointer to fGraph2D->fY
   Double_t   *fZ;            //!Pointer to fGraph2D->fZ
   Double_t   *fXN;           //!Pointer to fDelaunay->fXN
   Double_t   *fYN;           //!Pointer to fDelaunay->fYN
   Double_t    fXNmin;        //!Equal to fDelaunay->fXNmin
   Double_t    fXNmax;        //!Equal to fDelaunay->fXNmax
   Double_t    fYNmin;        //!Equal to fDelaunay->fYNmin
   Double_t    fYNmax;        //!Equal to fDelaunay->fYNmax
   Double_t    fXmin;         //!
   Double_t    fXmax;         //!
   Double_t    fYmin;         //! fGraph2D->fHistogram limits
   Double_t    fYmax;         //!
   Double_t    fZmin;         //!
   Double_t    fZmax;         //!
   Int_t       fNpoints;      //!Equal to fGraph2D->fNpoints
   Int_t       fNdt;          //!Equal to fDelaunay->fNdt
   Int_t      *fPTried;       //!Pointer to fDelaunay->fPTried
   Int_t      *fNTried;       //!Pointer to fDelaunay->fNTried
   Int_t      *fMTried;       //!Pointer to fDelaunay->fMTried

   TGraphDelaunay *fDelaunay; // Pointer to the TGraphDelaunay to be painted
   TGraph2D *fGraph2D;        // Pointer to the TGraph2D in fDelaunay

   void     FindTriangles();
   void     PaintLevels(Int_t *T, Double_t *x, Double_t *y, Int_t nblev=0, Double_t *glev=0);
   void     PaintPolyMarker0(Int_t n, Double_t *x, Double_t *y);

public:

   TGraphPainter();
   TGraphPainter(TGraphDelaunay *gd);

   virtual ~TGraphPainter();

   TList *GetContourList(Double_t contour);
   void   Paint(Option_t *option);
   void   PaintTriangles(Option_t *option);
   void   PaintPolyMarker(Option_t *option);
   void   PaintContour(Option_t *option);

   ClassDef(TGraphPainter,1)  // TGraph painter
};

#endif
