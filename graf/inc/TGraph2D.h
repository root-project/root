// @(#)root/graf:$Name:  $:$Id: TGraph2D.h,v 1.00
// Author: Olivier Couet, Luke Jones (Royal Holloway, University of London)

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
#ifndef ROOT_TF2
#include "TF2.h"
#endif

class TView;
class TDirectory;

class TGraph2D : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:

   Int_t       fNpoints;     // Number of points in the data set
   Int_t       fNpx;         // Number of bins along X in fHistogram
   Int_t       fNpy;         // Number of bins along Y in fHistogram
   Int_t       fNdt;         //!Number of Delaunay triangles found
   Int_t       fNhull;       //!Number of points in the hull
   Int_t       fSize;        //!Real size of fX, fY and fZ
   Double_t   *fX;           //[fNpoints]
   Double_t   *fY;           //[fNpoints] Data set to be plotted
   Double_t   *fZ;           //[fNpoints]
   Double_t   *fXN;          //!Normalized version of fX
   Double_t   *fYN;          //!Normalized version of fY
   Double_t    fXNmin;       //!Minimum value of fXN
   Double_t    fXNmax;       //!Maximum value of fXN
   Double_t    fYNmin;       //!Minimum value of fYN
   Double_t    fYNmax;       //!Maximum value of fYN
   Double_t    fXoffset;     //!
   Double_t    fYoffset;     //!Parameters used to normalize fX and fY
   Double_t    fScaleFactor; //!
   Double_t   *fGridLevels;  //!Grid levels along Z axis
   Double_t    fMargin;      // Extra space (in %) around interpolated area for 2D histo
   Double_t    fMinimum;     // Minimum value for plotting along z
   Double_t    fMaximum;     // Maximum value for plotting along z
   Double_t    fZout;        // Histogram bin height for points lying outside the convex hull
   Double_t   *fDist;        //!Array used to order mass points by distance
   Int_t       fMaxIter;     //!Maximum number of iterations to find Delaunay triangles
   Int_t       fTriedSize;   //!Real size of the fxTried arrays
   Int_t       fNbLevels;    //|Number of Grid levels
   Int_t      *fPTried;      //!
   Int_t      *fNTried;      //!Delaunay triangles storage
   Int_t      *fMTried;      //!
   Int_t      *fHullPoints;  //!Hull points
   Int_t      *fOrder;       //!Array used to order mass points by distance
   TList      *fFunctions;   // Pointer to list of functions (fits and user)
   TH2D       *fHistogram;   //!2D histogram of z values linearly interpolated
   TDirectory *fDirectory;   //!Pointer to directory holding this 2D graph
   TView      *fView;        //!TView used to paint the triangles
   
   void     Build(Int_t n);
   Double_t ComputeZ(Double_t x, Double_t y);
   void     DefineGridLevels();
   Bool_t   Enclose(Int_t T1, Int_t T2, Int_t T3, Int_t Ex) const;
   void     FileIt(Int_t P, Int_t N, Int_t M);
   void     FindAllTriangles();
   void     FindHull();
   Bool_t   InHull(Int_t E, Int_t X) const;
   Double_t InterpolateOnPlane(Int_t TI1, Int_t TI2, Int_t TI3, Int_t E) const;
   void     PaintLevels(Int_t *T, Double_t *x, Double_t *y, Double_t zmin, Double_t zmax, Int_t grid);
   void     PaintPolyMarker0(Int_t n, Double_t *x, Double_t *y);
   void     PaintTriangles(Option_t *option="");
   void     Reset(Int_t level=0);

public:
    // TGraph2D status bits
   enum {
      kFitInit = BIT(19)
   };

   TGraph2D();
   TGraph2D(Int_t n, Option_t *option="");
   TGraph2D(Int_t n, Int_t *x, Int_t *y, Int_t *z, Option_t *option="");
   TGraph2D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option="");
   TGraph2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *option="");
   TGraph2D(const char *name, const char *title, Int_t n, Double_t *x, Double_t *y, Double_t *z, Option_t *option="");
   TGraph2D(const char *filename, const char *format="%lg %lg %lg", Option_t *option="");
   TGraph2D(const TGraph2D &);

   virtual ~TGraph2D();

   TGraph2D operator=(const TGraph2D &);

   Int_t            DistancetoPrimitive(Int_t px, Int_t py);
    virtual void    Draw(Option_t *option="");
   void             ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TObject *FindObject(const char *name) const;
   virtual TObject *FindObject(const TObject *obj) const;
   virtual Int_t    Fit(const char *formula ,Option_t *option="" ,Option_t *goption=""); // *MENU*
   virtual Int_t    Fit(TF2 *f2 ,Option_t *option="" ,Option_t *goption=""); // *MENU*
   TDirectory      *GetDirectory() const {return fDirectory;}
   Double_t         GetMargin() const {return fMargin;}
   Int_t            GetNpx() const {return fNpx;}
   Int_t            GetNpy() const {return fNpy;}
   Double_t         GetMarginBinsContent() const {return fZout;}
   TH2D            *GetHistogram(Option_t *option="") const;
   TList           *GetListOfFunctions() const { return fFunctions; }
   virtual Double_t GetErrorX(Int_t bin) const;
   virtual Double_t GetErrorY(Int_t bin) const;
   virtual Double_t GetErrorZ(Int_t bin) const;
   Int_t            GetN() const {return fNpoints;}
   Double_t        *GetX() const {return fX;}
   Double_t        *GetY() const {return fY;}
   Double_t        *GetZ() const {return fZ;}
   Double_t         GetXmax() const;
   Double_t         GetXmin() const;
   Double_t         GetYmax() const;
   Double_t         GetYmin() const;
   Double_t         GetZmax() const;
   Double_t         GetZmin() const;
   Double_t         Interpolate(Double_t x, Double_t y) const;
   void             Paint(Option_t *option="");
   TH1             *Project(Option_t *option="x") const; // *MENU*
   Int_t            RemovePoint(Int_t ipoint); // *MENU*
   virtual void     SavePrimitive(ofstream &out, Option_t *option);
   virtual void     SetDirectory(TDirectory *dir);
   void             SetMargin(Double_t m=0.1); // *MENU*
   void             SetMaximum(Double_t maximum=-1111); // *MENU*
   void             SetMinimum(Double_t minimum=-1111); // *MENU*
   void             SetMaxIter(Int_t n=100000);
   virtual void     SetName(const char *name); // *MENU*
   void             SetNpx(Int_t npx=40); // *MENU*
   void             SetNpy(Int_t npx=40); // *MENU*
   void             SetPoint(Int_t point, Double_t x, Double_t y, Double_t z); // *MENU*
   virtual void     SetTitle(const char *title=""); // *MENU*
   void             SetMarginBinsContent(Double_t z=0.); // *MENU*

   ClassDef(TGraph2D,1)  //Set of n x[i],y[i],z[i] points with 3-d graphics including Delaunay triangulation
};

#endif
