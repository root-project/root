// @(#)root/hist:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAxis
#define ROOT_TAxis


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAxis                                                                //
//                                                                      //
// Axis class.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TAttAxis
#include "TAttAxis.h"
#endif
#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif

class THashList;

class TAxis : public TNamed, public TAttAxis {

private:
   Int_t        fNbins;          //Number of bins
   Double_t     fXmin;           //low edge of first bin
   Double_t     fXmax;           //upper edge of last bin
   TArrayD      fXbins;          //Bin edges array in X
   Int_t        fFirst;          //first bin to display
   Int_t        fLast;           //last bin to display
   UShort_t     fBits2;          //second bit status word
   Bool_t       fTimeDisplay;    //on/off displaying time values instead of numerics
   TString      fTimeFormat;     //Date&time format, ex: 09/12/99 12:34:00
   TObject     *fParent;         //!Object owning this axis
   THashList   *fLabels;         //List of labels

public:
   // TAxis status bits
   enum { kTickPlus      = BIT(9),
          kTickMinus     = BIT(10),
          kAxisRange     = BIT(11),
          kCenterTitle   = BIT(12),
          kCenterLabels  = BIT(14), //bit 13 is used by TObject
          kRotateTitle   = BIT(15),
          kPalette       = BIT(16),
          kNoExponent    = BIT(17),
          kLabelsHori    = BIT(18),
          kLabelsVert    = BIT(19),
          kLabelsDown    = BIT(20),
          kLabelsUp      = BIT(21),
          kIsInteger     = BIT(22),
          kMoreLogLabels = BIT(23),
          kDecimals      = BIT(11)}; //in fBits2

   TAxis();
   TAxis(Int_t nbins, Double_t xmin, Double_t xmax);
   TAxis(Int_t nbins, const Double_t *xbins);
   TAxis(const TAxis &axis);
   virtual ~TAxis();
   TAxis& operator=(const TAxis&);

   virtual void       CenterLabels(Bool_t center=kTRUE);  // *TOGGLE* *GETTER=GetCenterLabels
   virtual void       CenterTitle(Bool_t center=kTRUE);  // *TOGGLE* *GETTER=GetCenterTitle
   const char        *ChooseTimeFormat(Double_t axislength=0);
   virtual void       Copy(TObject &axis) const;
   virtual void       Delete(Option_t * /*option*/ ="") { }
   virtual Int_t      DistancetoPrimitive(Int_t px, Int_t py);
   virtual TObject   *DrawClone(Option_t * /*option*/ ="") const {return 0;}
   virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Int_t      FindBin(Double_t x);
   virtual Int_t      FindBin(const char *label);
   virtual Int_t      FindFixBin(Double_t x) const;
   virtual Double_t   GetBinCenter(Int_t bin) const;
   virtual Double_t   GetBinCenterLog(Int_t bin) const;
   const char        *GetBinLabel(Int_t bin) const;
   virtual Double_t   GetBinLowEdge(Int_t bin) const;
   virtual Double_t   GetBinUpEdge(Int_t bin) const;
   virtual Double_t   GetBinWidth(Int_t bin) const;
   virtual void       GetCenter(Double_t *center) const;
           Bool_t     GetCenterLabels() const;
           Bool_t     GetCenterTitle() const;
   THashList         *GetLabels() {return fLabels;}
   virtual void       GetLowEdge(Double_t *edge) const;
           Bool_t     GetMoreLogLabels() const;
           Int_t      GetNbins() const { return fNbins; }
           Bool_t     GetNoExponent() const;
           Bool_t     GetDecimals() const;
   virtual TObject   *GetParent() const {return fParent;}
           Bool_t     GetRotateTitle() const;
   virtual const char *GetTicks() const;
   virtual Bool_t     GetTimeDisplay() const {return fTimeDisplay;}
   virtual const char *GetTimeFormat() const {return fTimeFormat.Data();}
   virtual const char *GetTimeFormatOnly() const;
   const char        *GetTitle() const {return fTitle.Data();}
   const TArrayD     *GetXbins() const {return &fXbins;}
           Int_t      GetFirst() const;
           Int_t      GetLast() const;
           Double_t   GetXmin() const {return fXmin;}
           Double_t   GetXmax() const {return fXmax;}
   virtual void       ImportAttributes(const TAxis *axis);
   Bool_t             IsVariableBinSize() const {
                         // true if axis has variable bin sizes, false otherwise
                         return (fXbins.GetSize() != 0);
                      }
   virtual void       LabelsOption(Option_t *option="h");  // *MENU*
   virtual void       RotateTitle(Bool_t rotate=kTRUE); // *TOGGLE* *GETTER=GetRotateTitle
   virtual void       SaveAttributes(std::ostream &out, const char *name, const char *subname);
   virtual void       Set(Int_t nbins, Double_t xmin, Double_t xmax);
   virtual void       Set(Int_t nbins, const Float_t *xbins);
   virtual void       Set(Int_t nbins, const Double_t *xbins);
   virtual void       SetBinLabel(Int_t bin, const char *label);
   virtual void       SetDefaults();
   virtual void       SetDrawOption(Option_t * /*option*/ ="") { }
   virtual void       SetLimits(Double_t xmin, Double_t xmax);
   virtual void       SetMoreLogLabels(Bool_t more=kTRUE);  // *TOGGLE* *GETTER=GetMoreLogLabels
   virtual void       SetNoExponent(Bool_t noExponent=kTRUE);  // *TOGGLE* *GETTER=GetNoExponent
   virtual void       SetDecimals(Bool_t dot=kTRUE);  // *TOGGLE* *GETTER=GetDecimals
   virtual void       SetParent(TObject *obj) {fParent = obj;}
   virtual void       SetRange(Int_t first=0, Int_t last=0);  // *MENU*
   virtual void       SetRangeUser(Double_t ufirst, Double_t ulast);  // *MENU*
   virtual void       SetTicks(Option_t *option="+"); // *MENU*
   virtual void       SetTimeDisplay(Int_t value) {fTimeDisplay = (value != 0);} // *TOGGLE*
   virtual void       SetTimeFormat(const char *format="");  // *MENU*
   virtual void       SetTimeOffset(Double_t toffset, Option_t *option="local");
   virtual void       UnZoom();  // *MENU*
   virtual void       ZoomOut(Double_t factor=0, Double_t offset=0);  // *MENU*

   ClassDef(TAxis,9)  //Axis class
};

#endif

