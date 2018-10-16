/// \file ROOT/RTreeModel.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTreeModel
#define ROOT7_RTreeModel

#include <ROOT/RBranch.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeEntry.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RTreeModel
\ingroup Forest
\brief The RTreeModel encapulates the schema of a tree.

The tree model comprises a collection of hierarchically organized branches. From a frozen model, "entries"
can be extracted. As a convenience, the model provides a default entry. Models have a unique model identifier
that faciliates checking whether entries are compatible with it (i.e.: have been extracted from that model).
A model needs to be frozen before it can be used to create an RTree.
*/
// clang-format on
class RTreeModel {
  RBranchSubtree fRootBranch;
  RTreeEntry fDefaultEntry;

public:
   RTreeModel();

   /// Creates a new branch and corresponding cargo object
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> Branch(std::string_view branchName, ArgsT&&... args) {
     RBranch<T> *branch = new RBranch<T>(branchName);
     fRootBranch.Attach(branch);

     return fDefaultEntry.AddCargo<T>(branch, std::forward<ArgsT>(args)...);
   }

   /// Mounts an existing model as a sub tree, which allows for composing of tree models
   std::shared_ptr<RCargoSubtree> BranchCollection(std::string_view branchName, std::shared_ptr<RTreeModel> subModel);
};

} // namespace Exerimental
} // namespace ROOT

#endif
