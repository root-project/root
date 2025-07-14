// @(#)root/treeplayer:$Id$
// Author: Marek Biskup   24/01/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeDrawArgsParser
#define ROOT_TTreeDrawArgsParser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeDrawArgsParser                                                  //
//                                                                      //
// A class that parses all parameters for TTree::Draw().                //
// See TTree::Draw() for the format description.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

#include "TString.h"


class TTreeDrawArgsParser : public TObject {

public:
   enum EOutputType {
      kUNKNOWN,
      kEVENTLIST,
      kENTRYLIST,
      kPROFILE,
      kPROFILE2D,
      kGRAPH,
      kPOLYMARKER3D,
      kHISTOGRAM1D,
      kHISTOGRAM2D,
      kLISTOFGRAPHS,
      kLISTOFPOLYMARKERS3D,
      kHISTOGRAM3D
   };

   static Int_t   fgMaxDimension;     ///< = 4
   static Int_t   fgMaxParameters;    ///< = 9

protected:
   TString        fExp;               ///< Complete variable expression
   TString        fSelection;         ///< Selection expression
   TString        fOption;            ///< Draw options

   Int_t          fDimension;         ///< Dimension of the histogram/plot
   TString        fVarExp[4];         ///< Variable expression 0 - X, 1 - Y, 2 - Z, 3 - W
                                      ///< If dimension < fgMaxDimension then some
                                      ///< Expressions are empty

   bool           fAdd;               ///< Values should be added to an existing object
   TString        fName;              ///< Histogram's/plot's name

   Int_t          fNoParameters;      ///< If dimensions of the plot was specified
   bool           fParameterGiven[9]; ///< True if the parameter was given, otherwise false
   Double_t       fParameters[9];     ///< Parameters in brackets

   bool           fShouldDraw;        ///< If to draw the plot
   bool           fOptionSame;        ///< If option contained "same"
   bool           fEntryList;         ///< If fill a TEntryList
   TObject       *fOriginal;          ///< Original plot (if it is to be reused)
   bool           fDrawProfile;       ///< True if the options contain :"prof"
   EOutputType    fOutputType;        ///< Type of the output

   void           ClearPrevious();
   TTreeDrawArgsParser::EOutputType DefineType();
   bool           SplitVariables(TString variables);
   bool           ParseName(TString name);
   bool           ParseOption();
   bool           ParseVarExp();

public:
   TTreeDrawArgsParser();
   ~TTreeDrawArgsParser() override;

   bool           Parse(const char *varexp, const char *selection, Option_t *option);
   bool           GetAdd() const { return fAdd; }
   Int_t          GetDimension() const { return fDimension; }
   bool           GetShouldDraw() const { return fShouldDraw; }
   TString const& GetExp() const { return fExp; }
   Double_t       GetIfSpecified(Int_t num, Double_t def) const;
   Int_t          GetNoParameters() const { return fNoParameters; }
   Double_t       GetParameter(int num) const;
   TString const& GetObjectName() const { return fName; }
   TString        GetObjectTitle() const;
   bool           GetOptionSame() const { return fOptionSame; }
   TObject       *GetOriginal() const { return fOriginal; }
   TString const& GetSelection() const { return fSelection; }
   TString        GetVarExp(Int_t num) const;
   TString        GetVarExp() const;
   bool           IsSpecified(int num) const;
   void           SetObjectName(const char *s) { fName = s; }
   void           SetOriginal(TObject *o) { fOriginal = o; }
   static Int_t   GetMaxDimension();

   ClassDefOverride(TTreeDrawArgsParser,0); // Helper class to parse the argument to TTree::Draw
};

#endif

