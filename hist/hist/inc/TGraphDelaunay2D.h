// @(#)root/hist:$Id: TGraphDelaunay2D.h,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphDelaunay2D
#define ROOT_TGraphDelaunay2D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphDelaunay2D                                                     //
//                                                                      //
// This class uses the Delaunay triangles technique to interpolate and  //
// render the data set.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

#include "Math/Delaunay2D.h"

class TGraph2D;
class TView;

class TGraphDelaunay2D : public TNamed {

public:


private:
   TGraphDelaunay2D(const TGraphDelaunay2D&) = delete;
   TGraphDelaunay2D& operator=(const TGraphDelaunay2D&) = delete;

protected:

   TGraph2D   *fGraph2D;               ///<! 2D graph containing the user data
   ROOT::Math::Delaunay2D   fDelaunay; ///<! Delaunay interpolator class

public:

   typedef  ROOT::Math::Delaunay2D::Triangles Triangles;

   TGraphDelaunay2D(TGraph2D *g = 0);

   Double_t  ComputeZ(Double_t x, Double_t y) { return fDelaunay.Interpolate(x,y); }
   void      FindAllTriangles() { fDelaunay.FindAllTriangles(); }

   TGraph2D *GetGraph2D() const {return fGraph2D;}
   Double_t  GetMarginBinsContent() const {return fDelaunay.ZOuterValue();}
   Int_t     GetNdt() const {return fDelaunay.NumberOfTriangles(); }
   Double_t  GetXNmin() const {return fDelaunay.XMin();}
   Double_t  GetXNmax() const {return fDelaunay.XMax();}
   Double_t  GetYNmin() const {return fDelaunay.YMin();}
   Double_t  GetYNmax() const {return fDelaunay.YMax();}

   void      SetMarginBinsContent(Double_t z=0.) { fDelaunay.SetZOuterValue(z); }

   Triangles::const_iterator begin() const { return fDelaunay.begin(); }
   Triangles::const_iterator end()  const { return fDelaunay.end(); }

   ClassDefOverride(TGraphDelaunay2D,1)  // Delaunay triangulation

private:


};

#endif
