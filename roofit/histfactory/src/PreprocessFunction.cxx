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

#include <RooStats/HistFactory/PreprocessFunction.h>

#include <sstream>

namespace {

/// Replaces the XML special characters with their escape codes.
std::string escapeXML(const std::string &src)
{
   std::stringstream dst;
   for (char ch : src) {
      switch (ch) {
      case '&': dst << "&amp;"; break;
      case '\'': dst << "&apos;"; break;
      case '"': dst << "&quot;"; break;
      case '<': dst << "&lt;"; break;
      case '>': dst << "&gt;"; break;
      default: dst << ch; break;
      }
   }
   return dst.str();
}

} // namespace

RooStats::HistFactory::PreprocessFunction::PreprocessFunction(std::string const &name, std::string const &expression,
                                                              std::string const &dependents)
   : fName(name), fExpression(expression), fDependents(dependents)
{
}

std::string RooStats::HistFactory::PreprocessFunction::GetCommand() const
{
   return "expr::" + fName + "('" + fExpression + "',{" + fDependents + "})";
}

void RooStats::HistFactory::PreprocessFunction::Print(std::ostream &stream) const
{
   stream << "\t \t Name: " << fName << "\t \t Expression: " << fExpression << "\t \t Dependents: " << fDependents
          << std::endl;
}

void RooStats::HistFactory::PreprocessFunction::PrintXML(std::ostream &xml) const
{
   xml << "<Function Name=\"" << fName << "\" "
       << "Expression=\"" << escapeXML(fExpression) << "\" "
       << "Dependents=\"" << fDependents << "\" "
       << "/>\n";
}
