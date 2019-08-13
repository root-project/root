/// \file RFieldVisitor.cxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ROOT/RField.hxx"
#include "ROOT/RFieldVisitor.hxx"
#include "ROOT/RNTuple.hxx"


//---------------------------- RPrintVisitor ------------------------------------

void ROOT::Experimental::RPrintVisitor::SetDeepestLevel(int d) {
   fDeepestLevel = d;
   fFlagForVerticalLines.resize(d - 1);
}

void ROOT::Experimental::RPrintVisitor::SetNumFields(int n) {
   fNumFields = n;
   SetAvailableSpaceForStrings();
}

std::string ROOT::Experimental::RPrintVisitor::MakeKeyString(const Detail::RFieldBase &field, int level)
{
   std::string result{""};
   if (level==1) {
      result += "Field ";
      result += std::to_string(field.GetLevelInfo().GetOrder());
   } else {
      if (field.GetLevelInfo().GetOrder() == field.GetLevelInfo().GetNumSiblings()) { fFlagForVerticalLines.at(level-2) = false;
      } else {
         fFlagForVerticalLines.at(level-2) = true;
      }
      for(int i = 0; i < level-2; ++i) {
         if (fFlagForVerticalLines.at(i)) {
            result+= "| ";
         } else {
            result += "  ";
         }
      }
      result += "|__Field ";
      result += RNTupleFormatter::HierarchialFieldOrder(field);
   }
   return result;
}

std::string ROOT::Experimental::RPrintVisitor::MakeValueString(const Detail::RFieldBase &field)
{
   std::string nameAndType{field.GetName() + " (" + field.GetType() + ")"};
   return nameAndType;
}

// Entire function only prints 1 Line, when if statement is disregarded.
void ROOT::Experimental::RPrintVisitor::VisitField(const Detail::RFieldBase &field, int level)
{
   if (level == 1)
   {
      for (int i = 0; i < fWidth; ++i) {
         fOutput << fFrameSymbol;
      }
      fOutput << std::endl;
   }
   fOutput << fFrameSymbol << ' ';
   fOutput << RNTupleFormatter::FitString(MakeKeyString(field, level), fAvailableSpaceKeyString);
   fOutput << " : ";
   fOutput << RNTupleFormatter::FitString(MakeValueString(field), fAvailableSpaceValueString);
   fOutput << fFrameSymbol << std::endl;
}

//---------------------- RPrepareVisitor -------------------------------


void ROOT::Experimental::RPrepareVisitor::VisitField(const Detail::RFieldBase &/*field*/, int level)
{
   ++fNumFields;
   if (level > fDeepestLevel)
      fDeepestLevel = level;
}

//------------------------ RNTupleFormatter -----------------------------

//E.g. ("ExampleString" , space= 8) => "Examp..."
std::string ROOT::Experimental::RNTupleFormatter::FitString(const std::string &str, int availableSpace) {
   int strSize{static_cast<int>(str.size())};
   if (strSize <= availableSpace)
      return str + std::string(availableSpace - strSize, ' ');
   else if (availableSpace < 3)
      return std::string(availableSpace, '.');
   return std::string(str, 0, availableSpace - 3) + "...";
}

// Returns std::string of form "1" or "2.1.1"
std::string ROOT::Experimental::RNTupleFormatter::HierarchialFieldOrder(const ROOT::Experimental::Detail::RFieldBase &field)
{
   std::string hierarchialOrder{std::to_string(field.GetLevelInfo().GetOrder())};
   const ROOT::Experimental::Detail::RFieldBase* parentPtr{field.GetParent()};
   // To avoid having the index of the RootField (-1) in the return value, it is checked if the grandparent is a nullptr (in that case RootField is parent)
   while (parentPtr && (parentPtr->GetLevelInfo().GetOrder() != -1)) {
      hierarchialOrder = std::to_string(parentPtr->GetLevelInfo().GetOrder()) + "." + hierarchialOrder;
      parentPtr = parentPtr->GetParent();
   }
   return hierarchialOrder;
}
