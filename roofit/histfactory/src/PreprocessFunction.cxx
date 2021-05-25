// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, George Lewis 
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/** 
 * \ingroup HistFactory
 */

#include "RooStats/HistFactory/PreprocessFunction.h"


RooStats::HistFactory::PreprocessFunction::PreprocessFunction() {;}

RooStats::HistFactory::PreprocessFunction::PreprocessFunction(std::string Name, std::string Expression, std::string Dependents) : 
  fName(Name), fExpression(Expression), fDependents(Dependents) {
  fCommand = GetCommand(Name, Expression, Dependents);
}

std::string RooStats::HistFactory::PreprocessFunction::GetCommand(std::string Name, std::string Expression, std::string Dependents) {
  std::string command = "expr::"+Name+"('"+Expression+"',{"+Dependents+"})";
  return command;
}


void RooStats::HistFactory::PreprocessFunction::Print( std::ostream& stream ) {
  
  stream << "\t \t Name: " << fName
	 << "\t \t Expression: " << fExpression
	 << "\t \t Dependents: " << fDependents
	 << std::endl;  
  
}

void RooStats::HistFactory::PreprocessFunction::PrintXML( std::ostream& xml ) {
  xml << "<Function Name=\"" << GetName() << "\" "
      << "Expression=\""     << GetExpression() << "\" "
      << "Dependents=\""     << GetDependents() << "\" "
      << "/>" << std::endl;
}
