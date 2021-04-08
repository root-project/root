// @(#)root/hist:$Id: TGraphDelaunay.h,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphDelaunay
#define ROOT_TGraphDelaunay


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphDelaunay                                                       //
//                                                                      //
// This class uses the Delaunay triangles technique to interpolate and  //
// render the data set.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TGraph2D;
class TView;

class TGraphDelaunay : public TNamed {

private:

   TGraphDelaunay(const TGraphDelaunay&); // Not implemented
   TGraphDelaunay& operator=(const TGraphDelaunay&); // Not implemented

protected:

   Int_t       fNdt;          ///<! Number of Delaunay triangles found
   Int_t       fNpoints;      ///<! Number of data points in fGraph2D
   Int_t       fNhull;        ///<! Number of points in the hull
   Double_t   *fX;            ///<! Pointer to fGraph2D->fX
   Double_t   *fY;            ///<! Pointer to fGraph2D->fY
   Double_t   *fZ;            ///<! Pointer to fGraph2D->fZ
   Double_t   *fXN;           ///<! fGraph2D vectors normalized of size fNpoints
   Double_t   *fYN;           ///<! fGraph2D vectors normalized of size fNpoints
   Double_t    fXNmin;        ///<! Minimum value of fXN
   Double_t    fXNmax;        ///<! Maximum value of fXN
   Double_t    fYNmin;        ///<! Minimum value of fYN
   Double_t    fYNmax;        ///<! Maximum value of fYN
   Double_t    fXoffset;      ///<!
   Double_t    fYoffset;      ///<! Parameters used to normalize user data
   Double_t    fXScaleFactor; ///<!
   Double_t    fYScaleFactor; ///<!
   Double_t    fZout;         ///<! Histogram bin height for points lying outside the convex hull
   Double_t   *fDist;         ///<! Array used to order mass points by distance
   Int_t       fMaxIter;      ///<! Maximum number of iterations to find Delaunay triangles
   Int_t       fTriedSize;    ///<! Real size of the fxTried arrays
   Int_t      *fPTried;       ///<!
   Int_t      *fNTried;       ///<! Delaunay triangles storage of size fNdt
   Int_t      *fMTried;       ///<!
   Int_t      *fHullPoints;   ///<! Hull points of size fNhull
   Int_t      *fOrder;        ///<! Array used to order mass points by distance
   Bool_t      fAllTri;       ///<! True if FindAllTriangles() has been performed on fGraph2D
   Bool_t      fInit;         ///<! True if CreateTrianglesDataStructure() and FindHull() have been performed
   TGraph2D   *fGraph2D;      ///<! 2D graph containing the user data

   void     CreateTrianglesDataStructure();
   Bool_t   Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t Ex) const;
   void     FileIt(Int_t P, Int_t N, Int_t M);
   void     FindHull();
   Bool_t   InHull(Int_t E, Int_t X) const;
   Double_t InterpolateOnPlane(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const;

public:

   TGraphDelaunay();
   TGraphDelaunay(TGraph2D *g);

   virtual ~TGraphDelaunay();

   Double_t  ComputeZ(Double_t x, Double_t y);
   void      FindAllTriangles();
   TGraph2D *GetGraph2D() const {return fGraph2D;}
   Double_t  GetMarginBinsContent() const {return fZout;}
   Int_t     GetNdt() const {return fNdt;}
   Int_t    *GetPTried() const {return fPTried;}
   Int_t    *GetNTried() const {return fNTried;}
   Int_t    *GetMTried() const {return fMTried;}
   Double_t *GetXN() const {return fXN;}
   Double_t *GetYN() const {return fYN;}
   Double_t  GetXNmin() const {return fXNmin;}
   Double_t  GetXNmax() const {return fXNmax;}
   Double_t  GetYNmin() const {return fYNmin;}
   Double_t  GetYNmax() const {return fYNmax;}
   Double_t  Interpolate(Double_t x, Double_t y);
   void      SetMaxIter(Int_t n=100000);
   void      SetMarginBinsContent(Double_t z=0.);

   ClassDef(TGraphDelaunay,1)  // Delaunay triangulation
};

#endif
