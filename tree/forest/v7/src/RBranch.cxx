/// \file RBranch.cxx
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

#include "ROOT/RBranch.hxx"

ROOT::Experimental::Detail::RBranchBase::RBranchBase(std::string_view /*name*/)
{
}

ROOT::Experimental::Detail::RBranchBase::~RBranchBase()
{
}


ROOT::Experimental::RBranchCollection::RBranchCollection(std::string_view name)
   : ROOT::Experimental::Detail::RBranchBase(name)
{
}


ROOT::Experimental::RBranchCollection::~RBranchCollection()
{
}


void ROOT::Experimental::RBranchCollection::GenerateColumns(ROOT::Experimental::Detail::RPageStorage& /*storage*/)
{
}


std::unique_ptr<ROOT::Experimental::Detail::RTreeValueBase> ROOT::Experimental::RBranchCollection::GenerateValue()
{
   return nullptr;
}

void ROOT::Experimental::RBranchCollection::DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& /*value*/)
{
}

void ROOT::Experimental::RBranchCollection::DoRead(TreeIndex_t /*index*/, const ROOT::Experimental::Detail::RTreeValueBase& /*value*/)
{
}

void ROOT::Experimental::RBranchCollection::DoReadV(TreeIndex_t /*index*/, TreeIndex_t /*count*/, void* /*dst*/)
{
}
