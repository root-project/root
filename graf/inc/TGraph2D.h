// @(#)root/graf:$Name:  $:$Id: TGraph2D.h,v 1.00
// Author: Olivier Couet   23/10/03
// Author: Luke Jones (Royal Holloway, University of London) April 2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraph2D
#define ROOT_TGraph2D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraph2D                                                             //
//                                                                      //
// Graph 2D graphics class.                                             //
//                                                                      //
// This class uses the Delaunay triangles technique to interpolate and  //
// render the data set.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed 
#include "TNamed.h"  
#endif
#ifndef ROOT_TH2
#include "TH2.h"
#endif

class TGraph2D : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:

   Int_t     fNp;          // Number of points to in the data set
   Int_t     fNpx;         // number of bins along X in fHistogram
   Int_t     fNpy;         // number of bins along Y in fHistogram
   Int_t     fNdt;         //!Number of Delaunay triangles found
   Int_t     fNxt;         //!Number of non-Delaunay triangles found
   Int_t     fNhull;       //!Number of points in the hull
   Double_t *fX;           //[fNp] Data set to be plotted. It is 
   Double_t *fY;           //[fNp] stored in a normalized form.
   Double_t *fZ;           //[fNp]
   Double_t  fXmin;        //!Minimum value of fX
   Double_t  fXmax;        //!Maximum value of fX
   Double_t  fYmin;        //!Minimum value of fY
   Double_t  fYmax;        //!Maximum value of fY
   Double_t  fMargin;      // extra space (in %) around interpolated area for 2D histo
   Double_t  fZout;        // Histogram bin height for points lying outside the convex hull
   Double_t  fXoffset;     //!Offset fX
   Double_t  fYoffset;     //!Offset fY
   Double_t  fScaleFactor; //!Scale so the average of the fX and fY ranges is one
   Double_t *fDist;        //!Array used to order mass points by distance
   Int_t    *fTried;       //!Encoded triangles (see FileIt)
   Int_t    *fHullPoints;  //!Hull points
   Int_t    *fOrder;       //!Array used to order mass points by distance
   TH2D *fHistogram;       //!2D histogram of z values linearly interpolated
   
           Double_t ComputeZ(Double_t x, Double_t y);
	   void     CreateHistogram();
           Bool_t   Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t Ex) const;
           void     FileIt(Int_t tri);
           void     FillHistogram();
           void     FindHull();
           Bool_t   InHull(Int_t E, Int_t X) const;
           Double_t Interpolate(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const;
           void     PaintMarkers();
           void     PaintTriangles();
           Int_t    TriEncode(Int_t T1, Int_t T2, Int_t T3) const;
public:

           TGraph2D();
           TGraph2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *option="");
           virtual ~TGraph2D();
           Int_t    DistancetoPrimitive(Int_t px, Int_t py);
           void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
           Double_t GetMargin() const {return fMargin;}
           Int_t    GetNpx() const {return fNpx;}
           Int_t    GetNpy() const {return fNpy;}
           Double_t GetMarginBinsContent() const {return fZout;}
           void     Paint(Option_t *option="");
	   TH1     *Project(Option_t *option="x") const; // *MENU*
           void     SetMargin(Double_t m=0.1); // *MENU*
           void     SetNpx(Int_t npx=40); // *MENU*
           void     SetNpy(Int_t npx=40); // *MENU*
   virtual void     SetTitle(const char *title=""); // *MENU*
           void     SetMarginBinsContent(Double_t z=0.); // *MENU*
           void     Update();

   ClassDef(TGraph2D,1)  //Set of n x[i],y[i],z[i] points with 3-d graphics including Delaunay triangulation
};

#endif
