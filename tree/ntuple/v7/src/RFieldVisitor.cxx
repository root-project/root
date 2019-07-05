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

#include "ROOT/RFieldVisitor.hxx"
#include "ROOT/RField.hxx"
#include "ROOT/RNTuple.hxx"

static std::string HierarchialFieldOrder(const ROOT::Experimental::Detail::RFieldBase &field);

//---------------------------- RPrintVisitor ------------------------------------

std::string ROOT::Experimental::RPrintVisitor::KeyString(const ROOT::Experimental::Detail::RFieldBase &field, int level)
{
   std::string goesOut{""};
   if(level==1) {
      goesOut += "Field ";
      goesOut += std::to_string(field.GetIndex());
      goesOut += std::string(std::max(0, fAvailableSpaceKeyString-6-static_cast<int>(std::to_string(field.GetIndex()).size())), ' ');
   } else {
      if(field.IsLastInParentSubField()) fFlagforVerticalLines.at(level-2) = false;
      else fFlagforVerticalLines.at(level-2) = true;
      for(int i = 0; i < level-2; ++i) {
         if(fFlagforVerticalLines.at(i)) goesOut+= "| "; else goesOut += "  ";
      }
      goesOut += "|__Field ";
      goesOut += HierarchialFieldOrder(field);
      goesOut += std::string(std::max(0, fAvailableSpaceKeyString-2*(level-2)-static_cast<int>(HierarchialFieldOrder(field).size())-9), ' ');
   }
   return goesOut;
}

std::string ROOT::Experimental::RPrintVisitor::ValueString(const ROOT::Experimental::Detail::RFieldBase &field)
{
   std::string nameAndType{field.GetName() + " (" + field.GetType() + ")"};
   nameAndType += std::string(std::max(0, fAvailableSpaceValueString-static_cast<int>(nameAndType.size())),' '); //adding whitespaces
   return nameAndType;
}

// Entire function only prints 1 Line, when if statement is disregarded.
void ROOT::Experimental::RPrintVisitor::visitField(const ROOT::Experimental::Detail::RFieldBase &field, int level)
{
   if(level==1) {
      for(int i = 0; i < fWidth; ++i) fOutput << fFrameSymbol; fOutput << '\n';
   }
   fOutput << fFrameSymbol << ' ';
   fOutput << CutStringAndAddEllipsisIfNeeded(KeyString(field, level), fAvailableSpaceKeyString);
   fOutput << " : ";
   fOutput << CutStringAndAddEllipsisIfNeeded(ValueString(field), fAvailableSpaceValueString);
   fOutput << fFrameSymbol << '\n';
}

//---------------------- RPrepareVisitor -------------------------------


void ROOT::Experimental::RPrepareVisitor::visitField(const ROOT::Experimental::Detail::RFieldBase &/*field*/, int level)
{
   ++fNumFields;
   if(level > fDeepestLevel) fDeepestLevel = level;
}

//------------------------ Helper functions -----------------------------

//E.g. ("ExampleString" , space= 8) => "Examp..."
std::string ROOT::Experimental::CutStringAndAddEllipsisIfNeeded(const std::string &toCut, int maxAvailableSpace)
{
   if (maxAvailableSpace < 3) return "";
   if (static_cast<int>(toCut.size()) > maxAvailableSpace) {
      return std::string(toCut, 0, maxAvailableSpace - 3) + "...";
   }
   return toCut;
}

//Returns string of form "1" or "2.1.1"
std::string HierarchialFieldOrder(const ROOT::Experimental::Detail::RFieldBase &field)
{
   std::string hierarchialOrder{std::to_string(field.GetIndex())};
   const ROOT::Experimental::Detail::RFieldBase* ParentPtr{field.GetParent()};
   // To avoid having the index of the RootField (-1) in the return value, it is checked if the grandparent is a nullptr (in that case RootField is parent)
   while(ParentPtr && ParentPtr->GetParent()) {
      hierarchialOrder = std::to_string(ParentPtr->GetIndex()) + "." + hierarchialOrder;
      ParentPtr = ParentPtr->GetParent();
   }
   return hierarchialOrder;
}
