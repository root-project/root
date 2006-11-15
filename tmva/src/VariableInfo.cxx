// @(#)root/tmva $Id: VariableInfo.cxx,v 1.12 2006/10/18 01:24:53 armske Exp $   
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
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
                                  char varTypeOriginal, void* external ) 
   : fExpression     ( expression ),
     fVarTypeOriginal( varTypeOriginal ),
     fExternalData   ( external ),
     fVarCounter     ( varCounter )
{
   // constructor
   fVarType = 'F'; 

   fXminNorm[0] = fXminNorm[1] = fXminNorm[2] =  1e30;
   fXmaxNorm[0] = fXmaxNorm[1] = fXmaxNorm[2] = -1e30;
   fInternalVarName = TMVA::Tools::ReplaceRegularExpressions( GetExpression(), "_" );   
}

//_______________________________________________________________________
TMVA::VariableInfo::VariableInfo()
   : fExpression(""),
     fVarType('\0'),
     fExternalData(0)
{
   // constructor
   fXminNorm[0] = fXminNorm[1] = fXminNorm[2] =  1e30;
   fXmaxNorm[0] = fXmaxNorm[1] = fXmaxNorm[2] = -1e30;
   fInternalVarName = TMVA::Tools::ReplaceRegularExpressions( GetExpression(), "_" );   
}

//_______________________________________________________________________
TMVA::VariableInfo& TMVA::VariableInfo::operator=(const TMVA::VariableInfo& rhs)
{
   // comparison operator
   if(this!=&rhs) {
      fExpression       = rhs.fExpression;
      fInternalVarName  = rhs.fInternalVarName;
      fVarType          = rhs.fVarType;
      fXminNorm[0]      = rhs.fXminNorm[0];
      fXminNorm[1]      = rhs.fXminNorm[1];
      fXminNorm[2]      = rhs.fXminNorm[2];
      fXmaxNorm[0]      = rhs.fXmaxNorm[0];
      fXmaxNorm[1]      = rhs.fXmaxNorm[1];
      fXmaxNorm[2]      = rhs.fXmaxNorm[2];
   }
   return *this;
}

//_______________________________________________________________________
void TMVA::VariableInfo::WriteToStream(std::ostream& o, Types::PreprocessingMethod corr) const
{
   // write VariableInfo to stream   
   UInt_t nc = TMath::Max( 30, TMath::Max( GetExpression().Length()+1, GetInternalVarName().Length()+1 ) );
   TString expBr(Form("\'%s\'",GetExpression().Data()));
   o << setw(nc) << GetExpression();
   o << setw(nc) << GetInternalVarName();
   o << "    \'" << VarType() << "\'    ";
   o << "[" << GetMin(corr) << "," << GetMax(corr) << "]" << std::endl;
}

//_______________________________________________________________________
void TMVA::VariableInfo::ReadFromStream(std::istream& istr, Types::PreprocessingMethod corr)
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
   SetMin(min,corr);
   SetMax(max,corr);
}
