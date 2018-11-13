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

ROOT::Experimental::Detail::RTreeFieldBase::RTreeFieldBase(std::string_view /*name*/)
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


//-----------------------------------------------------------------------------


void ROOT::Experimental::RTreeFieldCollection::Attach(ROOT::Experimental::Detail::RTreeFieldBase* child)
{
}


ROOT::Experimental::RTreeFieldCollection::RTreeFieldCollection(std::string_view name)
   : ROOT::Experimental::Detail::RTreeFieldBase(name)
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
