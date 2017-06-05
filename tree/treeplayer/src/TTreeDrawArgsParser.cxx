// @(#)root/treeplayer:$Id$
// Author: Marek Biskup   24/01/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTreeDrawArgsParser
A class that parses all parameters for TTree::Draw().
See TTree::Draw() for the format description.
*/

#include "TTreeDrawArgsParser.h"
#include "TDirectory.h"


Int_t TTreeDrawArgsParser::fgMaxDimension = 4;
Int_t TTreeDrawArgsParser::fgMaxParameters = 9;


ClassImp(TTreeDrawArgsParser);

////////////////////////////////////////////////////////////////////////////////
/// Constructor - cleans all the class variables.

TTreeDrawArgsParser::TTreeDrawArgsParser()
{
   ClearPrevious();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TTreeDrawArgsParser::~TTreeDrawArgsParser()
{
}

////////////////////////////////////////////////////////////////////////////////
/// return fgMaxDimension (cannot be inline)

Int_t TTreeDrawArgsParser::GetMaxDimension()
{
   return fgMaxDimension;
}

////////////////////////////////////////////////////////////////////////////////
/// Resets all the variables of the class.

void TTreeDrawArgsParser::ClearPrevious()
{
   fExp = "";
   fSelection = "";
   fOption = "";
   fDimension = -1;
   int i;
   for (i = 0; i < fgMaxDimension; i++) {
      fVarExp[i] = "";
   }
   fAdd = kFALSE;
   fName = "";
   fNoParameters = 0;
   for (i = 0; i < fgMaxParameters; i++) {
      fParameterGiven[i] = kFALSE;
      fParameters[i] = 0;
   }
   fShouldDraw = kTRUE;
   fOriginal = 0;
   fDrawProfile = kFALSE;
   fOptionSame = kFALSE;
   fEntryList = kFALSE;
   fOutputType = kUNKNOWN;
}

////////////////////////////////////////////////////////////////////////////////
/// Parse expression [var1 [:var2 [:var3] ...]],
/// number of variables cannot be greater than fgMaxDimension.
///
/// A colon which is followed by (or that follows) another semicolon
/// is not regarded as a separator.
///
/// If there are more separating : than fgMaxDimension - 1 then
/// all characters after (fgMaxDimension - 1)th colon is put into
/// the last variable.
///
///  - `fDimension := <number of variables>`
///  - `fVarExp[0] := <first variable string>`
///  - `fVarExp[1] := <second variable string>`
/// ..
/// Returns kFALSE in case of an error.

Bool_t TTreeDrawArgsParser::SplitVariables(TString variables)
{
   fDimension = 0;
   if (variables.Length() == 0)
      return kTRUE;

   int prev = 0;
   int i;
   for (i = 0; i < variables.Length() && fDimension < fgMaxDimension; i++) {
      if (variables[i] == ':'
          && !( (i > 0 && variables[i - 1] == ':')
                || (i + 1 < variables.Length() && variables[i + 1] == ':') ) ) {
         fVarExp[fDimension] = variables(prev, i - prev);
         prev = i+1;
         fDimension++;
      }
   }
   if (fDimension < fgMaxDimension && i != prev)
      fVarExp[fDimension++] = variables(prev, i - prev);
   else
      return kFALSE;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Syntax:
///
///     [' '*][[\+][' '*]name[(num1 [, [num2] ] [, [num3] ] ...)]]
///
/// num's are floating point numbers
/// sets the fileds fNoParameters, fParameterGiven, fParameters, fAdd, fName
/// to appropriate values.
///
/// Returns kFALSE in case of an error.

Bool_t TTreeDrawArgsParser::ParseName(TString name)
{
   name.ReplaceAll(" ", "");

   if (name.Length() != 0 && name[0] == '+') {
      fAdd = kTRUE;
      name = name (1, name.Length() - 1);
   }
   else
      fAdd = kFALSE;
   Bool_t result = kTRUE;

   fNoParameters = 0;
   for (int i = 0; i < fgMaxParameters; i++)
      fParameterGiven[i] = kFALSE;

   if (char *p = (char*)strstr(name.Data(), "(")) {
      fName = name(0, p - name.Data());
      p++;
      char* end = p + strlen(p);

      for (int i = 0; i < fgMaxParameters; i++) {
         char* q = p;
         while (p < end && *p != ',' && *p != ')')
            p++;
         TString s(q, p - q);
         if (sscanf(s.Data(), "%lf", &fParameters[i]) == 1) {
            fParameterGiven[i] = kTRUE;
            fNoParameters++;
         }
         if (p == end) {
            Error("ParseName", "expected \')\'");
            result = kFALSE;
            break;
         }
         else if (*p == ')')
            break;
         else if (*p == ',')
            p++;
         else {
            Error("ParseName", "impossible value for *q!");
            result = kFALSE;
            break;
         }
      }
   }
   else { // if (char *p = strstr(name.Data(), "("))
      fName = name;
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Split variables and parse name and parameters in brackets.

Bool_t TTreeDrawArgsParser::ParseVarExp()
{
   char* gg = (char*)strstr(fExp.Data(), ">>");
   TString variables;
   TString name;

   if (gg) {
      variables = fExp(0, gg - fExp.Data());
      name = fExp(gg+2 - fExp.Data(), fExp.Length() - (gg + 2 - fExp.Data()));
   }
   else {
      variables = fExp;
      name = "";
   }
   Bool_t result = SplitVariables(variables) && ParseName(name);
   if (!result) {
      Error("ParseVarExp", "error parsing variable expression");
      return kFALSE;
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if options contain some data important for choosing the type of the
/// drawn object.

Bool_t TTreeDrawArgsParser::ParseOption()
{
   fOption.ToLower();

   if (fOption.Contains("goff")) {
      fShouldDraw = kFALSE;
   }
   if (fOption.Contains("prof")) {
      fDrawProfile = kTRUE;
   }
   if (fOption.Contains("same")) {
      fOptionSame = kTRUE;
   }
   if (fOption.Contains("entrylist")){
      fEntryList = kTRUE;
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Parses parameters from TTree::Draw().
///  - varexp - Variable expression; see TTree::Draw()
///  - selection - selection expression; see TTree::Draw()
///  - option - Drawing option; see TTree::Draw

Bool_t TTreeDrawArgsParser::Parse(const char *varexp, const char *selection, Option_t *option)
{
   ClearPrevious();

   // read the data provided and fill class fields
   fSelection = selection;
   fExp = varexp;
   fOption = option;
   Bool_t success = ParseVarExp();
   success &= ParseOption();

   if (!success)
      return kFALSE;

   // if the name was specified find the existing histogram
   if (fName != "") {
      fOriginal = gDirectory->Get(fName);
   }
   else
      fOriginal = 0;

   DefineType();

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Put the type of the draw result into fOutputType and return it.

TTreeDrawArgsParser::EOutputType TTreeDrawArgsParser::DefineType()
{
   if (fDimension == 0){
      if (fEntryList)
         return fOutputType = kENTRYLIST;
      else
         return fOutputType = kEVENTLIST;
   }
   if (fDimension == 2 && fDrawProfile)
      return fOutputType = kPROFILE;
   if (fDimension == 3 && fDrawProfile)
      return fOutputType = kPROFILE2D;

   if (fDimension == 2) {
      Bool_t graph = kFALSE;
// GG 9Mar2014: fixing ROOT-5337; should understand why it was like this, but we move to TSelectorDraw
//              and this will disappear
//      Int_t l = fOption.Length();
//      if (l == 0 || fOption.Contains("same")) graph = kTRUE;
      if (fOption.Contains("same")) graph = kTRUE;
      if (fOption.Contains("p")     || fOption.Contains("*")    || fOption.Contains("l"))    graph = kTRUE;
      if (fOption.Contains("surf")  || fOption.Contains("lego") || fOption.Contains("cont")) graph = kFALSE;
      if (fOption.Contains("col")   || fOption.Contains("hist") || fOption.Contains("scat")) graph = kFALSE;
      if (fOption.Contains("box"))                                                   graph = kFALSE;
      if (graph)
         return fOutputType = kGRAPH;
      else
         return fOutputType = kHISTOGRAM2D;
   }
   if (fDimension == 3) {
      if (fOption.Contains("col"))
         return fOutputType = kLISTOFGRAPHS;
      else
         return fOutputType = kHISTOGRAM3D;
// GG 9Mar2014: fixing ROOT-5337; should understand why it was like this, but we move to TSelectorDraw
//              and this will disappear
//         return fOutputType = kPOLYMARKER3D;
   }
   if (fDimension == 1)
      return fOutputType = kHISTOGRAM1D;
   if (fDimension == 4)
      return fOutputType = kLISTOFPOLYMARKERS3D;
   return kUNKNOWN;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns appropriate TSelector class name for proof for the object that is to be drawn
/// assumes that Parse() method has been called before.

TString TTreeDrawArgsParser::GetProofSelectorName() const
{
   switch (fOutputType) {
      case kUNKNOWN:
         return "";
      case kEVENTLIST:
         return "TProofDrawEventList";
      case kENTRYLIST:
         return "TProofDrawEntryList";
      case kPROFILE:
         return "TProofDrawProfile";
      case kPROFILE2D:
         return "TProofDrawProfile2D";
      case kGRAPH:
         return "TProofDrawGraph";
      case kPOLYMARKER3D:
         return "TProofDrawPolyMarker3D";
      case kLISTOFGRAPHS:
         return "TProofDrawListOfGraphs";
      case kHISTOGRAM1D:
      case kHISTOGRAM2D:
      case kHISTOGRAM3D:
         return "TProofDrawHist";
      case kLISTOFPOLYMARKERS3D:
         return "TProofDrawListOfPolyMarkers3D";
      default:
         return "";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// returns *num*-th parameter from brackets in the expression
/// in case of an error (wrong number) returns 0.0
/// num - number of parameter (counted from 0)

Double_t TTreeDrawArgsParser::GetParameter(Int_t num) const
{
   if (num >= 0 && num <= fgMaxParameters && fParameterGiven[num])
      return fParameters[num];
   else {
      Error("GetParameter","wrong arguments");
      return 0.0;
   }
}

////////////////////////////////////////////////////////////////////////////////
///  - num - parameter number
///  - def - default value of the parameter
/// returns the value of *num*-th parameter from the brackets in the variable expression
/// if the parameter of that number wasn't specified returns *def*.

Double_t TTreeDrawArgsParser::GetIfSpecified(Int_t num, Double_t def) const
{
   if (num >= 0 && num <= fgMaxParameters && fParameterGiven[num])
      return fParameters[num];
   else
      return def;
}

////////////////////////////////////////////////////////////////////////////////
/// returns kTRUE if the *num*-th parameter was specified
/// otherwise returns fFALSE
/// in case of an error (wrong num) prints an error message and
/// returns kFALSE.

Bool_t TTreeDrawArgsParser::IsSpecified(int num) const
{
   if (num >= 0 && num <= fgMaxParameters)
      return fParameterGiven[num];
   else
      Error("Specified", "wrong parameter %d; fgMaxParameters: %d", num, fgMaxParameters);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the *num*-th variable string
/// in case of an error prints an error message and returns an empty string.

TString TTreeDrawArgsParser::GetVarExp(Int_t num) const
{
   if (num >= 0 && num < fDimension)
      return fVarExp[num];
   else
      Error("GetVarExp", "wrong Parameters %d; fDimension = %d", num, fDimension);
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the variable string, i.e. [var1[:var2[:var2[:var4]]]].

TString TTreeDrawArgsParser::GetVarExp() const
{
   if (fDimension <= 0)
      return "";
   TString exp = fVarExp[0];
   for (int i = 1; i < fDimension; i++) {
      exp += ":";
      exp += fVarExp[i];
   }
   return exp;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the desired plot title.

TString TTreeDrawArgsParser::GetObjectTitle() const
{
   if (fSelection != "")
      return Form("%s {%s}", GetVarExp().Data(), fSelection.Data());
   else
      return GetVarExp();
}

