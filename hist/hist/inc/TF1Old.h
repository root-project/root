// @(#)root/hist:$Id$
// Author: Rene Brun   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- F1.h

#ifndef ROOT_TF1OLD
#define ROOT_TF1OLD



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF1Old                                                                  //
//                                                                      //
// Dummy class with same structure of old TF1 objects 
// used olny for reading the old files                                   //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#ifndef ROOT_TFormulaOld
#include "TFormulaOld.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif



struct TF1Old : public TFormulaOld, public TAttLine, public TAttFill, public TAttMarker {

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



   TF1Old();
   virtual   ~TF1Old();
   void Streamer(TBuffer &b, Int_t version, UInt_t start, UInt_t count, const TClass *onfile_class = 0);
   
   ClassDef(TF1Old,7)  //The Old Parametric 1-D function
};


#endif
