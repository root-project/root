// @(#)root/graf:$Name:  $:$Id: TPie.h,v 1.1 Exp $
// Author: Guido Volpi, 03/11/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPie
#define ROOT_TPie

#ifndef ROOT_TNamed_H
#include <TNamed.h>
#endif
#ifndef ROOT_TString_h
#include <TString.h>
#endif
#ifndef ROOT_TAttText
#include <TAttText.h>
#endif

class TPie : public TNamed , public TAttText {
private:
   void Init();

   void SliceSubdiv();

   void DrawGhost();

   Int_t fCurrent_slice; //! current slice under mouse
   Double_t fCurrent_phi1; //! phimin of the current slice
   Double_t fCurrent_phi2; //! phimax of the current slice
   Double_t fCurrent_rad;  //! current distance from the vertex of the current slice
   Double_t fCurrent_x; //! current x in the pad metric
   Double_t fCurrent_y; //! current y in the pad metric
   Double_t fCurrent_ang; //! current angular, within current_phi1 and current_phi2

   Float_t fSum; //! sum for the slice values
   Float_t *fSlices; //! subdivisions of the slices

protected:
   Double_t    fX;  //X coordinate of centre
   Double_t    fY;  //Y coordinate of centre
   Double_t    fRadius; //radius
   Double_t fOffset; //Offset angular offset for the first slice

   Float_t fLabelsOffset; //LabelsOffset offset of label

   TString fLabelFormat; //format format of the slices' label
   TString fValFormat; //vform numeric format for the value
   TString fFractionFormat; //rform numeric format for the fraction of a slice
   TString fPercentFormat; //pfrom numeric format for the percent of a slice

   Int_t fNvals; // number of elements
   Double_t *fVals; // [fNvals] vals of the elements
   TString *fLabels; // [fNvals] Labels

   Int_t *fFillColors; // [fNvals] slice colors
   Int_t *fFillStyles; // [fNvals] fill style for any pie

   Int_t *fLineColors; // [fNvals] color of line around the slice_mode
   Int_t *fLineStyles; // [fNvals] style of the line
   Int_t *fLineWidths; // [fNvals] width of the line

   Double_t *fRadOffsets; // [fNvals] distance of a slice for the center

public:

   TPie();
   TPie(const char *,const char *, Int_t);
   TPie(const char *,const char *, Int_t, Double_t *,Int_t *cols=0, const char *lbls[]=0);
   TPie(const char *,const char *, Int_t, Float_t *,Int_t *cols=0, const char *lbls[]=0);
   TPie(const TPie&);
   ~TPie();

   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   Int_t          DistancetoSlice(Int_t,Int_t);
   virtual void   Draw(Option_t *option="l");
   virtual void   ExecuteEvent(Int_t,Int_t,Int_t);
   const char*    GetEntryLabel(Int_t);
   Int_t          GetEntryFillColor(Int_t);
   Int_t          GetEntryFillStyle(Int_t);
   Double_t       GetEntryRadOffset(Int_t);
   Double_t       GetEntryVal(Int_t);
   const char    *GetFractionFormat() { return fFractionFormat.Data(); }
   const char    *GetLabelFormat() { return fLabelFormat.Data(); }
   Float_t        GetLabelsOffset() { return fLabelsOffset; }
   Double_t       GetOffset() { return fOffset; }
   const char    *GetPercentFormat() { return fPercentFormat.Data(); }
   Double_t       GetRadius() { return fRadius;}
   const char    *GetValFormat() { return fValFormat.Data(); }
   Double_t       GetX() { return fX; }
   Double_t       GetY() { return fY; }
   virtual void   Paint(Option_t *);
   void           SavePrimitive(ostream &out, Option_t *opts="");
   void           SetCircle(Double_t x=.5, Double_t y=.5, Double_t rad=.4);
   void           SetEntryLabel(Int_t, const char *text="Slice");
   void           SetEntryLineColor(Int_t, Int_t);
   void           SetEntryLineStyle(Int_t, Int_t);
   void           SetEntryLineWidth(Int_t, Int_t);
   void           SetEntryFillColor(Int_t, Int_t);
   void           SetEntryFillStyle(Int_t, Int_t);
   void           SetEntryRadOffset(Int_t, Double_t);
   void           SetEntryVal(Int_t, Double_t);
   void           SetFillColors(Int_t*);
   void           SetFractionFormat(const char*); // *MENU*
   void           SetLabel(const char *); // *MENU*
   void           SetLabelFormat(const char *); // *MENU*
   void           SetLabels(const char *[]);
   void           SetLabelsOffset(Float_t); // *MENU*
   void           SetOffset(Double_t); // *MENU*
   void           SetPercentFormat(const char *); // *MENU*
   void           SetRadius(Double_t); // *MENU*
   void           SetValFormat(const char *); // *MENU*
   void           SetX(Double_t); // *MENU*
   void           SetY(Double_t); // *MENU*

   ClassDef(TPie,1) //Pie chart graphics class
};

#endif // ROOT_TPie
