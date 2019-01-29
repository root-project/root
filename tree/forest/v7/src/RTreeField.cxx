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

#include "ROOT/RTreeField.hxx"
#include "ROOT/RTreeValue.hxx"

#include "TError.h"

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


ROOT::Experimental::Detail::RTreeFieldBase::const_iterator ROOT::Experimental::Detail::RTreeFieldBase::begin() const
{
   if (fSubFields.empty()) return const_iterator(this, -1);
   return const_iterator(this->fSubFields[0], 0);
}

ROOT::Experimental::Detail::RTreeFieldBase::const_iterator ROOT::Experimental::Detail::RTreeFieldBase::end() const
{
   return const_iterator(this, -1);
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::Detail::RTreeFieldBase::const_iterator::Advance()
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


void ROOT::Experimental::RTreeFieldCollection::Attach(ROOT::Experimental::Detail::RTreeFieldBase* child)
{
   child->fParent = this;
   fSubFields.emplace_back(child);
}


ROOT::Experimental::RTreeFieldCollection::RTreeFieldCollection(std::string_view name)
   : ROOT::Experimental::Detail::RTreeFieldBase(name, "", false /* isSimple */)
{
}


ROOT::Experimental::RTreeFieldCollection::~RTreeFieldCollection()
{
}


void ROOT::Experimental::RTreeFieldCollection::GenerateColumns(ROOT::Experimental::Detail::RPageStorage& /*storage*/)
{
}


ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeFieldCollection::GenerateValue()
{
   R__ASSERT(false);
   //return nullptr;
}

void ROOT::Experimental::RTreeFieldCollection::DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& /*value*/)
{
}

void ROOT::Experimental::RTreeFieldCollection::DoRead(TreeIndex_t /*index*/, const ROOT::Experimental::Detail::RTreeValueBase& /*value*/)
{
}

void ROOT::Experimental::RTreeFieldCollection::DoReadV(TreeIndex_t /*index*/, TreeIndex_t /*count*/, void* /*dst*/)
{
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::RTreeField<float>::GenerateColumns(ROOT::Experimental::Detail::RPageStorage& /*storage*/)
{
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeField<float>::GenerateValue()
{
   return ROOT::Experimental::RTreeValue<float>();
}
