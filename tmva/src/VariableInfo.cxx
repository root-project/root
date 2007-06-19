// @(#)root/tmva $Id: VariableInfo.cxx,v 1.7 2007/04/19 06:53:02 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

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

#include "Riostream.h"
#include "TMVA/VariableInfo.h"
#include "TMVA/Tools.h"

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo( const TString& expression, int varCounter, 
                                  char varType, void* external, 
                                  Double_t min, Double_t max ) 
   : fExpression     ( expression ),
     fVarType        ( varType ),
     fExternalData   ( external ),
     fVarCounter     ( varCounter )
{
   // constructor

   if (min == max) {
      fXminNorm =  1e30;
      fXmaxNorm = -1e30;
   } 
   else {
      fXminNorm =  min;
      fXmaxNorm =  max;
   }
   fInternalVarName = Tools::ReplaceRegularExpressions( GetExpression(), "_" );   
}

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo() 
   : fExpression(""),
     fVarType('\0'),
     fExternalData(0)
{
   // default constructor
   fXminNorm =  1e30;
   fXmaxNorm = -1e30;
   fInternalVarName = Tools::ReplaceRegularExpressions( GetExpression(), "_" );   
}

TMVA::VariableInfo::VariableInfo( const VariableInfo& other ) 
   : fExpression( other.fExpression ),
     fInternalVarName( other.fInternalVarName ),
     fVarType( other.fVarType ),
     fXminNorm( other.fXminNorm ),
     fXmaxNorm( other.fXmaxNorm ),
     fXmeanNorm( other.fXmeanNorm ),
     fXrmsNorm( other.fXrmsNorm ),
     fExternalData( other.fExternalData ),
     fVarCounter( other.fVarCounter )
{
   // copy constructor
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::VariableInfo::operator=(const VariableInfo& rhs)
{
   // comparison operator
   if (this!=&rhs) {
      fExpression       = rhs.fExpression;
      fInternalVarName  = rhs.fInternalVarName;
      fVarType          = rhs.fVarType;
      fXminNorm         = rhs.fXminNorm;
      fXmaxNorm         = rhs.fXmaxNorm;
   }
   return *this;
}

//_______________________________________________________________________
void TMVA::VariableInfo::WriteToStream(std::ostream& o) const
{
   // write VariableInfo to stream   
   UInt_t nc = TMath::Max( 30, TMath::Max( GetExpression().Length()+1, GetInternalVarName().Length()+1 ) );
   TString expBr(Form("\'%s\'",GetExpression().Data()));
   o << setw(nc) << GetExpression();
   o << setw(nc) << GetInternalVarName();
   o << "    \'" << fVarType << "\'    ";
   o << "[" << setprecision(12) << GetMin() << "," << setprecision(12) << GetMax() << "]" << std::endl;
}

//_______________________________________________________________________
void TMVA::VariableInfo::ReadFromStream(std::istream& istr)
{
   // write VariableInfo to stream
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
   SetExpression(exp);
   SetInternalVarName(varname);
   SetVarType(vartype[1]);
   SetMin(min);
   SetMax(max);
}
