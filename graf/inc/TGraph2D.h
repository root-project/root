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

   Int_t     fNp;          // Number of points in the data set
   Int_t     fNpx;         // Number of bins along X in fHistogram
   Int_t     fNpy;         // Number of bins along Y in fHistogram
   Int_t     fNdt;         //!Number of Delaunay triangles found
   Int_t     fNxt;         //!Number of non-Delaunay triangles found
   Int_t     fNhull;       //!Number of points in the hull
   Int_t     fSize;        // Real size of fX, fY and fZ
   Double_t *fX;           //[fNp]
   Double_t *fY;           //[fNp] Data set to be plotted
   Double_t *fZ;           //[fNp]
   Double_t *fXN;          //!Normalized version of fX
   Double_t *fYN;          //!Normalized version of fY
   Double_t  fXNmin;       //!Minimum value of fXN
   Double_t  fXNmax;       //!Maximum value of fXN
   Double_t  fYNmin;       //!Minimum value of fYN
   Double_t  fYNmax;       //!Maximum value of fYN
   Double_t  fMargin;      // Extra space (in %) around interpolated area for 2D histo
   Double_t  fZout;        // Histogram bin height for points lying outside the convex hull
   Double_t *fDist;        //!Array used to order mass points by distance
   Int_t    *fTried;       //!Encoded triangles (see FileIt)
   Int_t    *fHullPoints;  //!Hull points
   Int_t    *fOrder;       //!Array used to order mass points by distance
   TH2D     *fHistogram;   //!2D histogram of z values linearly interpolated
   
   Double_t ComputeZ(Double_t x, Double_t y);
   void     CreateHistogram();
   Bool_t   Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t Ex) const;
   void     FileIt(Int_t tri);
   void     FindHull();
   Bool_t   InHull(Int_t E, Int_t X) const;
   void     Initialise(Int_t n);
   Double_t Interpolate(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const;
   void     PaintMarkers();
   void     PaintTriangles();
   Int_t    TriEncode(Int_t T1, Int_t T2, Int_t T3) const;

public:

           TGraph2D();
           TGraph2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *option="");
           TGraph2D(Int_t n, Option_t *option="");
           virtual ~TGraph2D();
           Int_t    DistancetoPrimitive(Int_t px, Int_t py);
           void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
           Double_t GetMargin() const {return fMargin;}
           Int_t    GetNpx() const {return fNpx;}
           Int_t    GetNpy() const {return fNpy;}
           Double_t GetMarginBinsContent() const {return fZout;}
           TH2D    *GetHistogram() const;
           Double_t GetXmax() const;
           Double_t GetXmin() const;
           Double_t GetYmax() const;
           Double_t GetYmin() const;
           Double_t GetZmax() const;
           Double_t GetZmin() const;
           void     Paint(Option_t *option="");
           TH1     *Project(Option_t *option="x") const; // *MENU*
   virtual void     SavePrimitive(ofstream &out, Option_t *option);
           void     SetMargin(Double_t m=0.1); // *MENU*
           void     SetNpx(Int_t npx=40); // *MENU*
           void     SetNpy(Int_t npx=40); // *MENU*
           void     SetPoint(Int_t point, Double_t x, Double_t y, Double_t z); // *MENU*
   virtual void     SetTitle(const char *title=""); // *MENU*
           void     SetMarginBinsContent(Double_t z=0.); // *MENU*
           void     Update();

   ClassDef(TGraph2D,1)  //Set of n x[i],y[i],z[i] points with 3-d graphics including Delaunay triangulation
};

#endif
