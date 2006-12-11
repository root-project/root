// @(#)root/graf:$Name:  $:$Id: TPie.h,v 1.4 2006/11/24 13:07:55 couet Exp $
// Author: Guido Volpi, Olivier Couet  03/11/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPie
#define ROOT_TPie
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TNamed
#include <TNamed.h>
#endif
#ifndef ROOT_TString
#include <TString.h>
#endif
#ifndef ROOT_TAttText
#include <TAttText.h>
#endif

class TH1;

class TPie : public TNamed , public TAttText {
private:
   void Init(Int_t np, Double_t ao, Double_t x, Double_t y, Double_t r);
   void MakeSlices(Bool_t force=kFALSE);
   void DrawGhost();

   Float_t  fSum;           //!Sum for the slice values
   Float_t *fSlices;        //!Subdivisions of the slices

protected:
   Double_t  fX;              // X coordinate of the pie centre
   Double_t  fY;              // Y coordinate of the pie centre
   Double_t  fRadius;         // Pie radius
   Double_t  fAngularOffset;  // Offset angular offset for the first slice
   Float_t   fLabelsOffset;   // LabelsOffset offset of label
   TString   fLabelFormat;    // Format format of the slices' label
   TString   fValueFormat;    // Vform numeric format for the value
   TString   fFractionFormat; // Rform numeric format for the fraction of a slice
   TString   fPercentFormat;  // Pfrom numeric format for the percent of a slice
   Int_t     fNvals;          // Number of elements
   Double_t *fVals;           //[fNvals] Elements' values
   TString  *fLabels;         //[fNvals] Labels
   Int_t    *fFillColors;     //[fNvals] Slice fill color
   Int_t    *fFillStyles;     //[fNvals] Slive fill style
   Int_t    *fLineColors;     //[fNvals] Slice outline color
   Int_t    *fLineStyles;     //[fNvals] Line style
   Int_t    *fLineWidths;     //[fNvals] Line width
   Double_t *fRadiusOffsets;  //[fNvals] Distance of a slice from the center
   Bool_t    fIs3D;           //! true if the pseudo-3d is enabled
   Double_t  fHeight;         // Pheight height of the slice

public:

   TPie();
   TPie(const char *,const char *, Int_t);
   TPie(const char *,const char *, Int_t, Double_t *,Int_t *cols=0, const char *lbls[]=0);
   TPie(const char *,const char *, Int_t, Float_t *,Int_t *cols=0, const char *lbls[]=0);
   TPie(const TH1 *h);
   TPie(const TPie&);
   ~TPie();

   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   Int_t          DistancetoSlice(Int_t,Int_t);
   virtual void   Draw(Option_t *option="l");
   virtual void   ExecuteEvent(Int_t,Int_t,Int_t);
   Double_t       GetAngularOffset() { return fAngularOffset; }
   Double_t       GetHeight() { return fHeight; }
   const char*    GetEntryLabel(Int_t);
   Int_t          GetEntryFillColor(Int_t);
   Int_t          GetEntryFillStyle(Int_t);
   Double_t       GetEntryRadiusOffset(Int_t);
   Double_t       GetEntryVal(Int_t);
   const char    *GetFractionFormat() { return fFractionFormat.Data(); }
   const char    *GetLabelFormat() { return fLabelFormat.Data(); }
   Float_t        GetLabelsOffset() { return fLabelsOffset; }
   const char    *GetPercentFormat() { return fPercentFormat.Data(); }
   Double_t       GetRadius() { return fRadius;}
   const char    *GetValueFormat() { return fValueFormat.Data(); }
   Double_t       GetX() { return fX; }
   Double_t       GetY() { return fY; }
   virtual void   Paint(Option_t *);
   void           SavePrimitive(ostream &out, Option_t *opts="");
   void           SetAngularOffset(Double_t);
   void           SetCircle(Double_t x=.5, Double_t y=.5, Double_t rad=.4);
   void           SetEntryLabel(Int_t, const char *text="Slice");
   void           SetEntryLineColor(Int_t, Int_t);
   void           SetEntryLineStyle(Int_t, Int_t);
   void           SetEntryLineWidth(Int_t, Int_t);
   void           SetEntryFillColor(Int_t, Int_t);
   void           SetEntryFillStyle(Int_t, Int_t);
   void           SetEntryRadiusOffset(Int_t, Double_t);
   void           SetEntryVal(Int_t, Double_t);
   void           SetFillColors(Int_t*);
   void           SetFractionFormat(const char*); // *MENU*
   void           SetHeight(Double_t val=0.08); // *MENU*
   void           SetLabel(const char *); // *MENU*
   void           SetLabelFormat(const char *); // *MENU*
   void           SetLabels(const char *[]);
   void           SetLabelsOffset(Float_t); // *MENU*
   void           SetPercentFormat(const char *); // *MENU*
   void           SetRadius(Double_t); // *MENU*
   void           SetValueFormat(const char *); // *MENU*
   void           SetX(Double_t); // *MENU*
   void           SetY(Double_t); // *MENU*

   ClassDef(TPie,1) //Pie chart graphics class
};

#endif // ROOT_TPie
