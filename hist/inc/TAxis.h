// @(#)root/hist:$Name:  $:$Id: TAxis.h,v 1.12 2001/10/29 06:37:02 brun Exp $
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


class TAxis : public TNamed, public TAttAxis {

private:
        Int_t        fNbins;          //Number of bins
        Axis_t       fXmin;           //low edge of first bin
        Axis_t       fXmax;           //upper edge of last bin
        TArrayD      fXbins;          //Bin edges array in X
        Int_t        fFirst;          //first bin to display
        Int_t        fLast;           //last bin to display
        Bool_t       fTimeDisplay;    //on/off displaying time values instead of numerics
        TString      fTimeFormat;     //Date&time format, ex: 09/12/99 12:34:00
        TObject     *fParent;         //!Object owning this axis

public:
        // TAxis status bits
        enum { kAxisRange   = BIT(11), 
               kCenterTitle = BIT(12),
               kRotateTitle = BIT(15),
               kPalette     = BIT(16),
               kNoExponent  = BIT(17)};

        TAxis();
        TAxis(Int_t nbins, Axis_t xmin, Axis_t xmax);
        TAxis(Int_t nbins, const Axis_t *xbins);
        TAxis(const TAxis &axis);
        virtual ~TAxis();
        virtual void     CenterTitle(Bool_t center=kTRUE);  // *MENU*
        const char      *ChooseTimeFormat(Double_t axislength=0);
        virtual void     Copy(TObject &axis);
        virtual void     Delete(Option_t *option="") {;}
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual TObject *DrawClone(Option_t *option="") const {return 0;}
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual Int_t    FindBin(Axis_t x);
        virtual Int_t    FindFixBin(Axis_t x) const;
        virtual Axis_t   GetBinCenter(Int_t bin) const;
        virtual Axis_t   GetBinLowEdge(Int_t bin) const;
        virtual Axis_t   GetBinUpEdge(Int_t bin) const;
        virtual Axis_t   GetBinWidth(Int_t bin) const;
        virtual void     GetCenter(Axis_t *center);
        virtual void     GetLowEdge(Axis_t *edge);
                Int_t    GetNbins() const { return fNbins; }
        virtual TObject *GetParent() const {return fParent;}
        virtual Bool_t   GetTimeDisplay() const {return fTimeDisplay;}
        virtual const char  *GetTimeFormat() const {return fTimeFormat.Data();}
         const char     *GetTitle() const {return fTitle.Data();}
              TArrayD   *GetXbins() {return &fXbins;}
                 Int_t   GetFirst() const;
                 Int_t   GetLast() const;
                Axis_t   GetXmin() const {return fXmin;}
                Axis_t   GetXmax() const {return fXmax;}
        virtual void     RotateTitle(Bool_t rotate=kTRUE); // *MENU*
        virtual void     SaveAttributes(ofstream &out, const char *name, const char *subname);
        virtual void     Set(Int_t nbins, Axis_t xmin, Axis_t xmax);
        virtual void     Set(Int_t nbins, const Float_t *xbins);
        virtual void     Set(Int_t nbins, const Axis_t *xbins);
        virtual void     SetDrawOption(Option_t *option="") {;}
        virtual void     SetLimits(Axis_t xmin, Axis_t xmax);
        virtual void     SetNoExponent(Bool_t noExponent=kTRUE);  // *MENU*
        virtual void     SetParent(TObject *obj) {fParent = obj;}
        virtual void     SetRange(Int_t first=0, Int_t last=0);  // *MENU*
        virtual void     SetTimeDisplay(Int_t value) {fTimeDisplay = value;} // *TOGGLE*
        virtual void     SetTimeFormat(const char *format="");  // *MENU*
        virtual void     UnZoom();  // *MENU*

        ClassDef(TAxis,6)  //Axis class
};

#endif

