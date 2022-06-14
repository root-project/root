// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableInfo                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

/*! \class TMVA::VariableInfo
\ingroup TMVA
Class for type info of MVA input variable
*/

#include "TMVA/VariableInfo.h"
#include "TMVA/DataSetInfo.h"

#include "TMVA/Tools.h"

#include "TMath.h"
#include "TNamed.h"

#include <iomanip>
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::VariableInfo::VariableInfo( const TString& expression, const TString& title, const TString& unit,
                                  Int_t varCounter,
                                  char varType, void* external,
                                  Double_t min, Double_t max, Bool_t normalized )
     : TNamed(title.Data(),title.Data()),
     fExpression   ( expression ),
     fUnit         ( unit ),
     fVarType      ( varType ),
     fXmeanNorm    ( 0 ),
     fXrmsNorm     ( 0 ),
     fXvarianceNorm( 0 ),
     fNormalized   ( normalized ),
     fExternalData ( external ),
     fVarCounter   ( varCounter )
{
   if ( TMath::Abs(max - min) <= FLT_MIN ) {
      fXminNorm =  FLT_MAX;
      fXmaxNorm = -FLT_MAX;
   }
   else {
      fXminNorm =  min;
      fXmaxNorm =  max;
   }
   // if a label is set, than retrieve the label and the
   if (expression.Contains(":=")) {
      Ssiz_t index  = expression.Index(":=");
      fExpression   = expression(index+2,expression.Sizeof()-index-2);
      fLabel        = expression(0,index);
      fLabel        = fLabel.ReplaceAll(" ","");
   }
   else fLabel = GetExpression();

   if (fTitle == "") fTitle = fLabel;
   fInternalName = gTools().ReplaceRegularExpressions( fLabel, "_" );
}

////////////////////////////////////////////////////////////////////////////////
/// default constructor

TMVA::VariableInfo::VariableInfo()
   : TNamed(),
     fExpression   (""),
     fVarType      ('\0'),
     fXmeanNorm    ( 0 ),
     fXrmsNorm     ( 0 ),
     fXvarianceNorm( 0 ),
     fNormalized   ( kFALSE ),
     fExternalData ( 0 ),
     fVarCounter   ( 0 )
{
   fXminNorm     =  1e30;
   fXmaxNorm     = -1e30;
   fLabel        = GetExpression();
   fTitle        = fLabel;
   fName         = fTitle;
   fUnit         = "";
   fInternalName = gTools().ReplaceRegularExpressions( fLabel, "_" );
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor

TMVA::VariableInfo::VariableInfo( const VariableInfo& other )
   : TNamed(other),
     fExpression   ( other.fExpression ),
     fInternalName ( other.fInternalName ),
     fLabel        ( other.fLabel ),
     fUnit         ( other.fUnit ),
     fVarType      ( other.fVarType ),
     fXminNorm     ( other.fXminNorm ),
     fXmaxNorm     ( other.fXmaxNorm ),
     fXmeanNorm    ( other.fXmeanNorm ),
     fXrmsNorm     ( other.fXrmsNorm ),
     fXvarianceNorm( other.fXvarianceNorm ),
     fNormalized   ( other.fNormalized ),
     fExternalData ( other.fExternalData ),
     fVarCounter   ( other.fVarCounter )
{
}

////////////////////////////////////////////////////////////////////////////////
/// comparison operator

TMVA::VariableInfo& TMVA::VariableInfo::operator=(const VariableInfo& rhs)
{
   if (this !=& rhs) {
      fExpression       = rhs.fExpression;
      fInternalName     = rhs.fInternalName;
      fVarType          = rhs.fVarType;
      fXminNorm         = rhs.fXminNorm;
      fXmaxNorm         = rhs.fXmaxNorm;
      fTitle            = rhs.fTitle;
      fName             = rhs.fName;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// write VariableInfo to stream

void TMVA::VariableInfo::WriteToStream( std::ostream& o ) const
{
   UInt_t nc = TMath::Max( 30, TMath::Max( GetExpression().Length()+1, GetInternalName().Length()+1 ) );
   TString expBr(Form("\'%s\'",GetExpression().Data()));
   o << std::setw(nc) << GetExpression();
   o << std::setw(nc) << GetInternalName();
   o << std::setw(nc) << GetLabel();
   o << std::setw(nc) << GetTitle();
   o << std::setw(nc) << GetUnit();
   o << "    \'" << fVarType << "\'    ";
   o << "[" << std::setprecision(12) << GetMin() << "," << std::setprecision(12) << GetMax() << "]" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// read VariableInfo from stream

void TMVA::VariableInfo::ReadFromStream( std::istream& istr )
{
   // PLEASE do not modify this, it does not have to correspond to WriteToStream
   // this is needed to stay like this in 397 for backward compatibility
   TString exp, varname, vartype, minmax, minstr, maxstr;
   istr >> exp >> varname >> vartype >> minmax;
   exp.Strip(TString::kBoth, '\'');
   minmax = minmax.Strip(TString::kLeading, '[');
   minmax = minmax.Strip(TString::kTrailing, ']');
   minstr = minmax(0,minmax.First(','));
   maxstr = minmax(1+minmax.First(','),minmax.Length());
   Double_t min, max;
   std::stringstream strmin(minstr.Data());
   std::stringstream strmax(maxstr.Data());
   strmin >> min;
   strmax >> max;
   SetExpression     ( exp );
   SetInternalVarName( varname );
   SetLabel          ( varname );
   SetTitle          ( varname );
   SetUnit           ( "" );
   SetVarType        ( vartype[1] );
   SetMin            ( min );
   SetMax            ( max );
}

////////////////////////////////////////////////////////////////////////////////
/// write class to XML

void TMVA::VariableInfo::AddToXML( void* varnode )
{
   gTools().AddAttr( varnode, "Expression", GetExpression() );
   gTools().AddAttr( varnode, "Label",      GetLabel() );
   gTools().AddAttr( varnode, "Title",      GetTitle() );
   gTools().AddAttr( varnode, "Unit",       GetUnit() );
   gTools().AddAttr( varnode, "Internal",   GetInternalName() );

   TString typeStr(" ");
   typeStr[0] = GetVarType();
   // in case of array variables, add "[]" to the type string: e.g. F[]
   if (TestBit(DataSetInfo::kIsArrayVariable))
      typeStr += "[]"; 
   gTools().AddAttr(varnode, "Type", typeStr);
   gTools().AddAttr( varnode, "Min", gTools().StringFromDouble(GetMin()) );
   gTools().AddAttr( varnode, "Max", gTools().StringFromDouble(GetMax()) );
}

////////////////////////////////////////////////////////////////////////////////
/// read VariableInfo from stream

void TMVA::VariableInfo::ReadFromXML( void* varnode )
{
   TString type;
   gTools().ReadAttr( varnode, "Expression", fExpression );
   gTools().ReadAttr( varnode, "Label",      fLabel );
   gTools().ReadAttr( varnode, "Title",      fTitle );
   gTools().ReadAttr( varnode, "Unit",       fUnit );
   gTools().ReadAttr( varnode, "Internal",   fInternalName );
   gTools().ReadAttr( varnode, "Type",       type );
   gTools().ReadAttr( varnode, "Min",        fXminNorm );
   gTools().ReadAttr( varnode, "Max",        fXmaxNorm );

   SetVarType(type[0]);
   // detect variables from array 
   if (type.Contains("[]"))
      SetBit(DataSetInfo::kIsArrayVariable);
}
