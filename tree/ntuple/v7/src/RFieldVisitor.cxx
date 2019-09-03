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

#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>

#include <TInterpreter.h>
#include <TROOT.h>
#include <TString.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

//------------------------ RValueVisitor --------------------------------

void ROOT::Experimental::RValueVisitor::VisitField(const Detail::RFieldBase &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   // A custom object (has ENTupleStructure::kRecord) should let its subfields display its entries.
   // So the field of the custom object should only display its name.
   if (field.GetStructure() == ENTupleStructure::kRecord)
      return;
   
   // gROOT->ProcessLine doesn't like templates, so the below doesn't work...
   //gROOT->ProcessLine(TString::Format("auto view = fReader->GetView<std::vector<float>>(RNTupleFormatter::FieldHierarchy(field));"));
   //gROOT->ProcessLine(TString::Format("fOutput << gInterpreter->ToString(field.GetType().c_str(), view.GetRawPtr(fIndex))"));
   
   Detail::RFieldBase* fieldPtr = const_cast<Detail::RFieldBase*>(&field);
   Detail::RFieldValue value{fieldPtr->GenerateValue()};
   fieldPtr->Read(fIndex, &value);
   fOutput << gInterpreter->ToString(field.GetType().c_str(), value.GetRawPtr());
   
   if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
      fOutput << ',';
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitBoolField(const RField<bool> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<bool>(RNTupleFormatter::FieldHierarchy(field));
      if (view(fIndex) == 0) {
         fOutput << "false";
      } else {
         fOutput << "true";
      }
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitDoubleField(const RField<double> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<double>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fIndex);
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitFloatField(const RField<float> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<float>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fIndex);
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitIntField(const RField<int> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<int>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fIndex);
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitStringField(const RField<std::string> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<std::string>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << "\"" << view(fIndex) << "\"";
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUIntField(const RField<std::uint32_t> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<std::uint32_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fIndex);
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUInt64Field(const RField<std::uint64_t> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<std::uint64_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << view(fIndex);
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

void ROOT::Experimental::RValueVisitor::VisitUInt8Field(const RField<std::uint8_t> &field, int level)
{
   for (int i = 0; i < level; ++i) fOutput << "  ";
   fOutput << "\"" << field.GetName() << "\": ";
   
   if (field.GetStructure() != ENTupleStructure::kRecord) {
      auto view = fReader->GetView<std::uint8_t>(RNTupleFormatter::FieldHierarchy(field));
      fOutput << "'" << view(fIndex) << "'";
      if (field.GetLevelInfo().GetNumSiblings() != field.GetLevelInfo().GetOrder())
         fOutput << ',';
   }
   fOutput << std::endl;
}

//------------------------ RNTupleFormatter -----------------------------

// Returns std::string of form "SubFieldofRootFieldName. ... .ParentFieldName.FieldName"
std::string ROOT::Experimental::RNTupleFormatter::FieldHierarchy(const Detail::RFieldBase &field)
{
   std::string str{field.GetName()};
   if (!field.GetParent())
      return str;
   const Detail::RFieldBase* parentField{field.GetParent()};
   for (int i = field.GetLevelInfo().GetLevel(); i > 1; --i) {
      str = parentField->GetName() + "." + str;
      if (parentField->GetParent())
         parentField = parentField->GetParent();
   }
   return str;
}

// E.g. ("ExampleString" , space= 8) => "Examp..."
std::string ROOT::Experimental::RNTupleFormatter::FitString(const std::string &str, int availableSpace)
{
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
   const Detail::RFieldBase* parentPtr{field.GetParent()};
   // To avoid having the index of the RootField (-1) in the return value, it is checked if the grandparent is a nullptr (in that case RootField is parent)
   while (parentPtr && (parentPtr->GetLevelInfo().GetOrder() != -1)) {
      hierarchialOrder = std::to_string(parentPtr->GetLevelInfo().GetOrder()) + "." + hierarchialOrder;
      parentPtr = parentPtr->GetParent();
   }
   return hierarchialOrder;
}
