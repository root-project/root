// @(#)root/hist:$Id$
// Author: Rene Brun   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_v5_TF1DATA
#define ROOT_v5_TF1DATA



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF1Data                                                                 //
//                                                                      //
// Dummy class with same structure of v5::TF1 objects
// used only for reading the old files                                   //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include "v5/TFormula.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"

namespace ROOT {

   namespace v5 {

struct TF1Data : public ROOT::v5::TFormula, public TAttLine, public TAttFill, public TAttMarker {

   Double_t    fXmin;        //Lower bounds for the range
   Double_t    fXmax;        //Upper bounds for the range
   Int_t       fNpx;         //Number of points used for the graphical representation
   Int_t       fType;        //(=0 for standard functions, 1 if pointer to function)
   Int_t       fNpfits;      //Number of points used in the fit
   Int_t       fNDF;         //Number of degrees of freedom in the fit
   Int_t       fNsave;       //Number of points used to fill array fSave
   Double_t    fChisquare;   //Function fit chisquare
   Double_t    *fParErrors;  //[fNpar] Array of errors of the fNpar parameters
   Double_t    *fParMin;     //[fNpar] Array of lower limits of the fNpar parameters
   Double_t    *fParMax;     //[fNpar] Array of upper limits of the fNpar parameters
   Double_t    *fSave;       //[fNsave] Array of fNsave function values
   Double_t     fMaximum;    //Maximum value for plotting
   Double_t     fMinimum;    //Minimum value for plotting



   TF1Data();
     ~TF1Data() override;
   void Streamer(TBuffer &b, Int_t version, UInt_t start, UInt_t count, const TClass *onfile_class = nullptr);

   ClassDefOverride(TF1Data,7)  //The Parametric 1-D function data structure  of v5::TF1
};

   }  // end namespace v5
}   // end namespace ROOT

#endif
