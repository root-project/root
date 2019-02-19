/// \file RTreeModel.cxx
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

#include "ROOT/RTreeModel.hxx"

#include <TError.h>

#include <cstdlib>
#include <utility>

void ROOT::Experimental::RTreeModel::AddField(std::unique_ptr<Detail::RTreeFieldBase> field)
{
   fDefaultEntry.AddValue(field->GenerateValue());
   fRootField.Attach(std::move(field));
}

