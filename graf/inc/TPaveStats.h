// @(#)root/graf:$Name:  $:$Id: TPaveStats.h,v 1.4 2000/11/21 20:23:31 brun Exp $
// Author: Rene Brun   15/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPaveStats
#define ROOT_TPaveStats


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaveStats                                                           //
//                                                                      //
// a special TPaveText to draw histogram statistics                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPaveText
#include "TPaveText.h"
#endif


class TPaveStats : public TPaveText {

protected:
        Int_t         fOptFit;            //option Fit
        Int_t         fOptStat;           //option Stat
        TString       fFitFormat;         //Printing format for fit parameters
        TString       fStatFormat;        //Printing format for stats

public:
        TPaveStats();
        TPaveStats(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, Option_t *option="br");
        virtual ~TPaveStats();
        virtual TBox    *AddBox(Double_t , Double_t , Double_t , Double_t) {return 0;}
        virtual TLine   *AddLine(Double_t , Double_t , Double_t, Double_t) {return 0;}
        virtual void     DeleteText() {;}
        virtual void     EditText() {;}
        virtual const char  *GetFitFormat()  const {return fFitFormat.Data();}
        virtual const char  *GetStatFormat() const {return fStatFormat.Data();}
        Int_t            GetOptFit() const {return fOptFit;}
        Int_t            GetOptStat() const {return fOptStat;}
        virtual void     InsertText(const char *) {;}
        virtual void     ReadFile(const char *, Option_t *, Int_t, Int_t) {;}
        virtual void     SaveStyle(); // *MENU*
        virtual void     SetAllWith(const char *, Option_t *, Double_t) {;}
        virtual void     SetMargin(Float_t) {;}
        virtual void     SetFitFormat(const char *format="5.4g");    // *MENU*
        virtual void     SetStatFormat(const char *format="6.4g");   // *MENU*
        void             SetOptFit(Int_t fit=1) {fOptFit = fit;}     // *MENU*
        void             SetOptStat(Int_t stat=1) {fOptStat = stat;} // *MENU*

        ClassDef(TPaveStats,3)  //a special TPaveText to draw histogram statistics.
};

#endif
