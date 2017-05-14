// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h"      // R__USE_IMT
#include "ROOT/TDFNodes.hxx" // ColumnName2ColumnTypeName requires TCustomColumnBase
#include "ROOT/TDFUtils.hxx"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TClassRef.h"
#include "TROOT.h" // IsImplicitMTEnabled, GetImplicitMTPoolSize

#include <stdexcept>
#include <string>
class TTree;
using namespace ROOT::Detail::TDF;

namespace ROOT {
namespace Internal {
namespace TDF {

/// Return a string containing the type of the given branch. Works both with real TTree branches and with temporary
/// column created by Define.
std::string ColumnName2ColumnTypeName(const std::string &colName, TTree &tree, TCustomColumnBase *tmpBranch)
{
   if (auto branch = tree.GetBranch(colName.c_str())) {
      // this must be a real TTree branch
      static const TClassRef tbranchelRef("TBranchElement");
      if (branch->InheritsFrom(tbranchelRef)) {
         return static_cast<TBranchElement *>(branch)->GetClassName();
      } else { // Try the fundamental type
         auto title = branch->GetTitle();
         auto typeCode = title[strlen(title) - 1];
         if (typeCode == 'B')
            return "char";
         else if (typeCode == 'b')
            return "unsigned char";
         else if (typeCode == 'I')
            return "int";
         else if (typeCode == 'i')
            return "unsigned int";
         else if (typeCode == 'S')
            return "short";
         else if (typeCode == 's')
            return "unsigned short";
         else if (typeCode == 'D')
            return "double";
         else if (typeCode == 'F')
            return "float";
         else if (typeCode == 'L')
            return "Long64_t";
         else if (typeCode == 'l')
            return "ULong64_t";
         else if (typeCode == 'O')
            return "bool";
      }
   } else {
      // this must be a temporary branch
      const auto &type_id = tmpBranch->GetTypeId();
      if (auto c = TClass::GetClass(type_id)) {
         return c->GetName();
      } else if (type_id == typeid(char))
         return "char";
      else if (type_id == typeid(unsigned char))
         return "unsigned char";
      else if (type_id == typeid(int))
         return "int";
      else if (type_id == typeid(unsigned int))
         return "unsigned int";
      else if (type_id == typeid(short))
         return "short";
      else if (type_id == typeid(unsigned short))
         return "unsigned short";
      else if (type_id == typeid(long))
         return "long";
      else if (type_id == typeid(unsigned long))
         return "unsigned long";
      else if (type_id == typeid(double))
         return "double";
      else if (type_id == typeid(float))
         return "float";
      else if (type_id == typeid(Long64_t))
         return "Long64_t";
      else if (type_id == typeid(ULong64_t))
         return "ULong64_t";
      else if (type_id == typeid(bool))
         return "bool";
      else {
         std::string msg("Cannot deduce type of temporary column ");
         msg += colName.c_str();
         msg += ". The typename is ";
         msg += tmpBranch->GetTypeId().name();
         msg += ".";
         throw std::runtime_error(msg);
      }
   }

   std::string msg("Cannot deduce type of column ");
   msg += colName.c_str();
   msg += ".";
   throw std::runtime_error(msg);
}

const char *ToConstCharPtr(const char *s)
{
   return s;
}

const char *ToConstCharPtr(const std::string s)
{
   return s.c_str();
}

unsigned int GetNSlots()
{
   unsigned int nSlots = 1;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) nSlots = ROOT::GetImplicitMTPoolSize();
#endif // R__USE_IMT
   return nSlots;
}

void CheckTmpBranch(const std::string &branchName, TTree *treePtr)
{
   if (treePtr != nullptr) {
      auto branch = treePtr->GetBranch(branchName.c_str());
      if (branch != nullptr) {
         auto msg = "branch \"" + branchName + "\" already present in TTree";
         throw std::runtime_error(msg);
      }
   }
}

/// Returns local BranchNames or default BranchNames according to which one should be used
const ColumnNames_t &PickBranchNames(unsigned int nArgs, const ColumnNames_t &bl, const ColumnNames_t &defBl)
{
   bool useDefBl = false;
   if (nArgs != bl.size()) {
      if (bl.size() == 0 && nArgs == defBl.size()) {
         useDefBl = true;
      } else {
         auto msg = "mismatch between number of filter/define arguments (" + std::to_string(nArgs) +
                    ") and number of columns specified (" + std::to_string(bl.size() ? bl.size() : defBl.size()) +
                    "). Please check the number of arguments of the function/lambda/functor and the number of branches "
                    "specified.";
         throw std::runtime_error(msg);
      }
   }

   return useDefBl ? defBl : bl;
}

} // end NS TDF
} // end NS Internal
} // end NS ROOT
