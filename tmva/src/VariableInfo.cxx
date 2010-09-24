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

#include <sstream>
#include <iomanip>

#include "TMVA/VariableInfo.h"
#include "TMVA/Tools.h"

#include "TMath.h"

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo( const TString& expression, const TString& title, const TString& unit,
                                  Int_t varCounter,
                                  char varType, void* external,
                                  Double_t min, Double_t max, Bool_t normalized )
   : fExpression  ( expression ),
     fTitle       ( title ),
     fUnit        ( unit ),
     fVarType     ( varType ),
     fXmeanNorm   ( 0 ),
     fXrmsNorm    ( 0 ),
     fNormalized  ( normalized ),
     fExternalData( external ),
     fVarCounter  ( varCounter )
{
   // constructor

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

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo() 
   : fExpression  (""),
     fVarType     ('\0'),
     fXmeanNorm   ( 0 ),
     fXrmsNorm    ( 0 ),
     fNormalized  ( kFALSE ),
     fExternalData( 0 ),
     fVarCounter  ( 0 )
{
   // default constructor
   fXminNorm     =  1e30;
   fXmaxNorm     = -1e30;
   fLabel        = GetExpression();
   fTitle        = fLabel;
   fUnit         = "";
   fInternalName = gTools().ReplaceRegularExpressions( fLabel, "_" );   
}

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo( const VariableInfo& other ) 
   : fExpression  ( other.fExpression ),
     fInternalName( other.fInternalName ),
     fLabel       ( other.fLabel ),
     fTitle       ( other.fTitle ),
     fUnit        ( other.fUnit ),
     fVarType     ( other.fVarType ),
     fXminNorm    ( other.fXminNorm ),
     fXmaxNorm    ( other.fXmaxNorm ),
     fXmeanNorm   ( other.fXmeanNorm ),
     fXrmsNorm    ( other.fXrmsNorm ),
     fNormalized  ( other.fNormalized ),
     fExternalData( other.fExternalData ),
     fVarCounter  ( other.fVarCounter )
{
   // copy constructor
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::VariableInfo::operator=(const VariableInfo& rhs)
{
   // comparison operator
   if (this !=& rhs) {
      fExpression       = rhs.fExpression;
      fInternalName     = rhs.fInternalName;
      fVarType          = rhs.fVarType;
      fXminNorm         = rhs.fXminNorm;
      fXmaxNorm         = rhs.fXmaxNorm;
   }
   return *this;
}

//_______________________________________________________________________
void TMVA::VariableInfo::WriteToStream( std::ostream& o ) const
{
   // write VariableInfo to stream 
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

//_______________________________________________________________________
void TMVA::VariableInfo::ReadFromStream( std::istream& istr )
{
   // read VariableInfo from stream

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

//_______________________________________________________________________
void TMVA::VariableInfo::AddToXML( void* varnode )
{
   // write class to XML
   gTools().AddAttr( varnode, "Expression", GetExpression() );
   gTools().AddAttr( varnode, "Label",      GetLabel() );
   gTools().AddAttr( varnode, "Title",      GetTitle() );
   gTools().AddAttr( varnode, "Unit",       GetUnit() );
   gTools().AddAttr( varnode, "Internal",   GetInternalName() );

   TString typeStr(" ");
   typeStr[0] = GetVarType();
   gTools().AddAttr( varnode, "Type", typeStr );
   gTools().AddAttr( varnode, "Min", gTools().StringFromDouble(GetMin()) );
   gTools().AddAttr( varnode, "Max", gTools().StringFromDouble(GetMax()) );
}

//_______________________________________________________________________
void TMVA::VariableInfo::ReadFromXML( void* varnode ) 
{
   // read VariableInfo from stream
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
}
