// @(#)root/proof:$Name:  $:$Id: TDrawInfo.h,v 1.3 2005/03/10 19:15:22 rdm Exp $
// Author: Marek Biskup   24/01/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDrawInfo
#define ROOT_TDrawInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDrawInfo                                                            //
//                                                                      //
// Class that parses all parameters for TTree::Draw().                  //
// See TTree::Draw() for the format description.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TDrawInfo : public TObject {

public:
   enum EOutputType {
      kUNKNOWN,
      kEVENTLIST,
      kPROFILE,
      kPROFILE2D,
      kGRAPH,
      kPOLYMARKER3D,
      kHISTOGRAM1D,
      kHISTOGRAM2D,
      kLISTOFGRAPHS,
      kLISTOFPOLYMARKERS3D
   };

   static Int_t fgMaxDimension;      // = 4
   static Int_t fgMaxParameters;     // = 9

protected:
   TString       fExp;        // complete variable expression
   TString       fSelection;  // selection expression
   TString       fOption;     // draw options

   Int_t         fDimension;  // dimension of the histogram/plot
   TString       fVarExp[4];  // variable expression 0 - X, 1 - Y, 2 - Z, 3 - W
                              // if dimension < fgMaxDimension then some
                              // expressions are empty

   Bool_t        fAdd;        // values should be added to an existing object
   TString       fName;       // histogram's/plot's name

   Int_t         fNoParameters;      // if dimensions of the plot was specified
   Bool_t        fParameterGiven[9]; // true if the parameter was given, otherwise false
   Double_t      fParameters[9];     // parameters in brackets

   Bool_t        fDraw;              // if to draw the plot
   TObject      *fOriginal;          // original plot (if it is to be reused)
   Bool_t        fDrawProfile;       // true if the options contain :"prof"
   EOutputType   fOutputType;        // type of the output

   Bool_t SplitVariables(TString variables);
   Bool_t ParseName(TString name);
   Bool_t ParseVarExp();
   Bool_t ParseOption();
   void   ClearPrevious();

public:
   TDrawInfo();
   ~TDrawInfo();

   Bool_t       Parse(const char *varexp, const char *selection, Option_t *option);
   TString      GetProofSelectorName() const;
   Double_t     GetIfSpecified(Int_t num, Double_t def) const;
   Double_t     GetParameter(int num) const;
   TString      GetObjectName() const { return fName; }
   TString      GetObjectTitle() const { return Form("%s {%s}",
                                         GetVarExp().Data(), fSelection.Data()); }
   Bool_t       GetAdd() const { return fAdd; }
   Int_t        GetDimension() const { return fDimension; }
   TString      GetSelection() const { return fSelection; }
   Bool_t       IsSpecified(int num) const;
   TString      GetVarExp(Int_t num) const;
   TString      GetVarExp() const;
   TString      GetExp() const { return fExp; }
   TObject     *GetOriginal() const { return fOriginal; }
   Int_t        GetNoParameters() const { return fNoParameters; }
   Bool_t       GetDraw() const { return fDraw; }
   void         SetObjectName(const char *s) { fName = s; }
   void         SetOriginal(TObject *o) { fOriginal = o; }

   TDrawInfo::EOutputType DefineType();

   ClassDef(TDrawInfo,0)
};

#endif
