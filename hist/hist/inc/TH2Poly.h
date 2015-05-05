// @(#)root/hist:$Id$
// Author: Olivier Couet, Deniz Gunceler

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH2Poly
#define ROOT_TH2Poly

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH2Poly                                                              //
//                                                                      //
// 2-Dim histogram with polygon bins                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TH2
#include "TH2.h"
#endif

#include "TList.h"

class TH2PolyBin: public TObject{

public:
   TH2PolyBin();
   TH2PolyBin(TObject *poly, Int_t bin_number);
   virtual ~TH2PolyBin();

   void      ClearContent(){fContent = 0;}
   void      Fill(Double_t w) {fContent = fContent+w; SetChanged(true);}
   Double_t  GetArea();
   Double_t  GetContent() const{return fContent;}
   Bool_t    GetChanged() const{return fChanged;}
   Int_t     GetBinNumber() const {return fNumber;}
   TObject  *GetPolygon() const {return fPoly;}
   Double_t  GetXMax();
   Double_t  GetXMin();
   Double_t  GetYMax();
   Double_t  GetYMin();
   Bool_t    IsInside(Double_t x, Double_t y) const;
   void      SetChanged(Bool_t flag){fChanged = flag;}
   void      SetContent(Double_t content){fContent = content; SetChanged(true);}

protected:
   Bool_t    fChanged;   //For the 3D Painter
   Int_t     fNumber;    //Bin number of the bin in TH2Poly
   TObject  *fPoly;      //Object holding the polygon definition
   Double_t  fArea;      //Bin area
   Double_t  fContent;   //Bin content
   Double_t  fXmin;      //X minimum value
   Double_t  fYmin;      //Y minimum value
   Double_t  fXmax;      //X maximum value
   Double_t  fYmax;      //Y maximum value

   ClassDef(TH2PolyBin,1)  //2-Dim polygon bins
};

class TList;
class TGraph;
class TMultiGraph;
class TPad;

class TH2Poly : public TH2 {

public:
   TH2Poly();
   TH2Poly(const char *name,const char *title, Double_t xlow, Double_t xup, Double_t ylow, Double_t yup);
   TH2Poly(const char *name,const char *title, Int_t nX, Double_t xlow, Double_t xup,  Int_t nY, Double_t ylow, Double_t yup);
   virtual ~TH2Poly();

   Int_t        AddBin(TObject *poly);
   Int_t        AddBin(Int_t n, const Double_t *x, const Double_t *y);
   Int_t        AddBin(Double_t x1, Double_t y1, Double_t x2, Double_t  y2);
   virtual Bool_t Add(const TH1 *h1, Double_t c1);
   virtual Bool_t Add(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1);
   virtual Bool_t Add(TF1 *h1, Double_t c1=1, Option_t *option="");
   void         ClearBinContents();                 // Clears the content of all bins
   void         ChangePartition(Int_t n, Int_t m);  // Sets the number of partition cells to another value
   virtual TH1 *DrawCopy(Option_t *option="") const;
   Int_t        Fill(Double_t x,Double_t y);
   Int_t        Fill(Double_t x,Double_t y, Double_t w);
   Int_t        Fill(const char* name, Double_t w);
   void         FillN(Int_t ntimes, const Double_t* x, const Double_t* y, const Double_t* w, Int_t stride = 1);
   Int_t        Fill(Double_t){return -1;}                              //MayNotUse
   Int_t        Fill(Double_t , const char *, Double_t){return -1;}     //MayNotUse
   Int_t        Fill(const char *, Double_t , Double_t ){return -1;}    //MayNotUse
   Int_t        Fill(const char *, const char *, Double_t ){return -1;} //MayNotUse
   void         FillN(Int_t, const Double_t*, const Double_t*, Int_t){return;}  //MayNotUse
   Int_t        FindBin(Double_t x, Double_t y, Double_t z = 0);
   TList       *GetBins(){return fBins;}                                // Returns the TList of all bins in the histogram
   Double_t     GetBinContent(Int_t bin) const;
   Double_t     GetBinContent(Int_t, Int_t) const {return 0;}           //MayNotUse
   Double_t     GetBinContent(Int_t, Int_t, Int_t) const {return 0;}    //MayNotUse
   Bool_t       GetBinContentChanged() const{return fBinContentChanged;}
   Double_t     GetBinError(Int_t bin) const;
   Double_t     GetBinError(Int_t , Int_t) const {return 0;}            //MayNotUse
   Double_t     GetBinError(Int_t , Int_t , Int_t) const {return 0;}    //MayNotUse
   const char  *GetBinName(Int_t bin) const;
   const char  *GetBinTitle(Int_t bin) const;
   Bool_t       GetFloat(){return fFloat;}
   Double_t     GetMaximum() const;
   Double_t     GetMaximum(Double_t maxval) const;
   Double_t     GetMinimum() const;
   Double_t     GetMinimum(Double_t minval) const;
   Bool_t       GetNewBinAdded() const{return fNewBinAdded;}
   Int_t        GetNumberOfBins() const{return fNcells;}
   void         Honeycomb(Double_t xstart, Double_t ystart, Double_t a, Int_t k, Int_t s);   // Bins the histogram using a honeycomb structure
   Double_t     Integral(Option_t* option = "") const;
   Double_t     Integral(Int_t, Int_t, const Option_t*) const{return 0;}                             //MayNotUse
   Double_t     Integral(Int_t, Int_t, Int_t, Int_t, const Option_t*) const{return 0;}               //MayNotUse
   Double_t     Integral(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, const Option_t*) const{return 0;} //MayNotUse
   Long64_t     Merge(TCollection *);
   void         Reset(Option_t *option);
   void         SavePrimitive(ostream& out, Option_t* option = "");
   virtual void Scale(Double_t c1 = 1, Option_t* option = "");
   void         SetBinContent(Int_t bin, Double_t content);
   void         SetBinContent(Int_t, Int_t, Double_t){return;}           //MayNotUse
   void         SetBinContent(Int_t, Int_t, Int_t, Double_t){return;}    //MayNotUse
   void         SetBinContentChanged(Bool_t flag){fBinContentChanged = flag;}
   void         SetFloat(Bool_t flag = true);
   void         SetNewBinAdded(Bool_t flag){fNewBinAdded = flag;}

protected:
   TList   *fBins;              //List of bins. The list owns the contained objects
   Double_t fOverflow[9];       //Overflow bins
   Int_t    fCellX;             //Number of partition cells in the x-direction of the histogram
   Int_t    fCellY;             //Number of partition cells in the y-direction of the histogram
   Int_t    fNCells;            //Number of partition cells: fCellX*fCellY
   TList   *fCells;             //[fNCells] The array of TLists that store the bins that intersect with each cell. List do not own the contained objects
   Double_t fStepX, fStepY;     //Dimensions of a partition cell
   Bool_t  *fIsEmpty;           //[fNCells] The array that returns true if the cell at the given coordinate is empty
   Bool_t  *fCompletelyInside;  //[fNCells] The array that returns true if the cell at the given coordinate is completely inside a bin
   Bool_t   fFloat;             //When set to kTRUE, allows the histogram to expand if a bin outside the limits is added.
   Bool_t   fNewBinAdded;       //!For the 3D Painter
   Bool_t   fBinContentChanged; //!For the 3D Painter

   void   AddBinToPartition(TH2PolyBin *bin);  // Adds the input bin into the partition matrix
   void   Initialize(Double_t xlow, Double_t xup, Double_t ylow, Double_t yup, Int_t n, Int_t m);
   Bool_t IsIntersecting(TH2PolyBin *bin, Double_t xclipl, Double_t xclipr, Double_t yclipb, Double_t yclipt);
   Bool_t IsIntersectingPolygon(Int_t bn, Double_t *x, Double_t *y, Double_t xclipl, Double_t xclipr, Double_t yclipb, Double_t yclipt);  

   ClassDef(TH2Poly,1)  //2-Dim histogram with polygon bins
};

#endif
