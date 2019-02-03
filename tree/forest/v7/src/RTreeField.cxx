/// \file RTreeField.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RColumn.hxx"
#include "ROOT/RColumnModel.hxx"
#include "ROOT/RTreeField.hxx"
#include "ROOT/RTreeValue.hxx"

#include "TError.h"

#include <utility>

ROOT::Experimental::Detail::RTreeFieldBase::RTreeFieldBase(std::string_view name, std::string_view type, bool isSimple)
   : fName(name), fType(type), fIsSimple(isSimple), fParent(nullptr), fPrincipalColumn(nullptr)
{
}

ROOT::Experimental::Detail::RTreeFieldBase::~RTreeFieldBase()
{
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& /*value*/) {
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoRead(
   ROOT::Experimental::TreeIndex_t /*index*/,
   const RTreeValueBase& /*value*/)
{
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoReadV(
   ROOT::Experimental::TreeIndex_t /*index*/,
   ROOT::Experimental::TreeIndex_t /*count*/,
   void* /*dst*/)
{
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RTreeFieldBase::Attach(ROOT::Experimental::Detail::RTreeFieldBase* child)
{
   child->fParent = this;
   fSubFields.emplace_back(child);
}

std::string ROOT::Experimental::Detail::RTreeFieldBase::GetLeafName() const
{
   auto idx = fName.find_last_of(kLevelSeparator);
   return (idx == std::string::npos) ? fName : fName.substr(idx + 1);
}

ROOT::Experimental::Detail::RTreeFieldBase::RIterator ROOT::Experimental::Detail::RTreeFieldBase::begin()
{
   if (fSubFields.empty()) return RIterator(this, -1);
   return RIterator(this->fSubFields[0], 0);
}

ROOT::Experimental::Detail::RTreeFieldBase::RIterator ROOT::Experimental::Detail::RTreeFieldBase::end()
{
   return RIterator(this, -1);
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::Detail::RTreeFieldBase::RIterator::Advance()
{
   auto itr = fStack.rbegin();
   if (!itr->fFieldPtr->fSubFields.empty()) {
      fStack.emplace_back(Position(itr->fFieldPtr->fSubFields[0], 0));
      return;
   }

   unsigned int nextIdxInParent = ++(itr->fIdxInParent);
   while (nextIdxInParent >= itr->fFieldPtr->fParent->fSubFields.size()) {
      if (fStack.size() == 1) {
         itr->fFieldPtr = itr->fFieldPtr->fParent;
         itr->fIdxInParent = -1;
         return;
      }
      fStack.pop_back();
      itr = fStack.rbegin();
      nextIdxInParent = itr->fIdxInParent++;
   }
   itr->fFieldPtr = itr->fFieldPtr->fParent->fSubFields[nextIdxInParent];
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::RTreeFieldRoot::GenerateColumns(ROOT::Experimental::Detail::RPageStorage* /*storage*/)
{
   R__ASSERT(false);
}


ROOT::Experimental::Detail::RTreeValueBase* ROOT::Experimental::RTreeFieldRoot::GenerateValue()
{
   R__ASSERT(false);
   return nullptr;
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RTreeField<float>::GenerateColumns(ROOT::Experimental::Detail::RPageStorage* pageStorage)
{
   RColumnModel model(GetName(), EColumnType::kReal32, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model, pageStorage));
   fPrincipalColumn = fColumns[0].get();
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RTreeField<std::string>::GenerateColumns(ROOT::Experimental::Detail::RPageStorage* pageStorage)
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex, pageStorage));

   std::string columnChars(GetName());
   columnChars.push_back(kLevelSeparator);
   columnChars.append(GetLeafName());
   RColumnModel modelChars(columnChars, EColumnType::kByte, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelChars, pageStorage));
   fPrincipalColumn = fColumns[0].get();
}

