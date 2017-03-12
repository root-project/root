// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDataFrameInterface.hxx"
#include "TBranch.h"
#include "TBranchElement.h"

namespace ROOT {
namespace Internal {

std::string ColumnName2ColumnTypeName(const std::string &colName, ROOT::Detail::TDataFrameImpl &df)
{
   auto tree = df.GetTree();
   if (auto branch = tree->GetBranch(colName.c_str())) {
      static const TClassRef tbranchelRef("TBranchElement");
      if (branch->InheritsFrom(tbranchelRef)) {
         return static_cast<TBranchElement*>(branch)->GetClassName();
      } else { // Try the fundamental type
         auto title = branch->GetTitle();
         auto typeCode = title[strlen(title) - 1];
         if (typeCode == 'B') return "char";
         else if (typeCode == 'b') return "unsigned char";
         else if (typeCode == 'I') return "int";
         else if (typeCode == 'i') return "unsigned int";
         else if (typeCode == 'S') return "short";
         else if (typeCode == 's') return "unsigned short";
         else if (typeCode == 'D') return "double";
         else if (typeCode == 'F') return "float";
         else if (typeCode == 'L') return "Long64_t";
         else if (typeCode == 'l') return "ULong64_t";
         else if (typeCode == 'O') return "bool";
      }
   } else {
      const auto &type_id = df.GetBookedBranch(colName.c_str()).GetTypeId();
      if (auto c = TClass::GetClass(type_id)) {
         return c->GetName();
      } else if (type_id == typeid(char)) return "char";
      else if (type_id == typeid(unsigned char)) return "unsigned char";
      else if (type_id == typeid(int)) return "int";
      else if (type_id == typeid(unsigned int)) return "unsigned int";
      else if (type_id == typeid(short)) return "short";
      else if (type_id == typeid(unsigned short)) return "unsigned short";
      else if (type_id == typeid(double)) return "double";
      else if (type_id == typeid(float)) return "float";
      else if (type_id == typeid(Long64_t)) return "Long64_t";
      else if (type_id == typeid(ULong64_t)) return "ULong64_t";
      else if (type_id == typeid(bool)) return "bool";

   }
   return "";
}

} // namespace Internal

namespace Experimental {

// extern templates
template class TDataFrameInterface<ROOT::Detail::TDataFrameImpl>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>;

} // namespace Experimental
} // namespace ROOT
