/// \file ROOT/RTreeView.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-05
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTreeView
#define ROOT7_RTreeView

#include <ROOT/RBranch.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeUtil.hxx>
#include <ROOT/RTreeView.hxx>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RTreeView
\ingroup Forest
\brief An RTreeView provides read-only access to a single branch

(NB(jblomer): The tree view is very close to TTreeReader. Do we simply want to teach TTreeReader to deal with Forest?)

The view owns a branch and its underlying columns in order to fill a tree value object with values from the
given branch. Branch elements can be accessed by index. For top level branches, the index refers to the entry number.
Branches that are part of nested collections have global index numbers that are derived from their parent indexes.

The RTreeView object is an iterable. That means, all branch elements can be sequentially read from begin() to end().

For simple types, template specializations let the reading become a pure mapping into a page buffer.
*/
// clang-format on
template <typename T>
class RTreeView {
private:
   std::unique_ptr<RBranch<T>> fBranch;
   RTreeValue<T> fTreeValue;

public:
   RTreeView(std::unique_ptr<RBranch<T>> branch);

   /// Use the branch to read the referenced element into the tree value object. To be inlined
   const T& operator ()(TreeIndex_t index);
};


// clang-format off
/**
\class ROOT::Experimental::RTreeView<TreeIndex_t>
\ingroup Forest
\brief An RTreeView specialization for an index branch without its sub branches
*/
// clang-format on
template <>
class RTreeView<TreeIndex_t> {
protected:
   std::unique_ptr<RBranch<TreeIndex_t>> fBranch;
   RTreeValue<TreeIndex_t> fTreeValue;

public:
   RTreeView(std::unique_ptr<RBranch<TreeIndex_t>> branch);

   /// Use the branch to read the referenced element into the tree value object. To be inlined
   TreeIndex_t operator ()(TreeIndex_t index);
};


// clang-format off
/**
\class ROOT::Experimental::RTreeViewCollection
\ingroup Forest
\brief A tree view for a collection, that can itself generate new tree views for sub branches.
*/
// clang-format on
class RTreeViewCollection : public RTreeView<TreeIndex_t> {
public:
   template <typename T>
   RTreeView<T> GetView(std::string_view branchName) {
      auto branch = std::make_unique<RBranch<T>>(branchName, fBranch->GetSource());
      // ...
      return RTreeView<T>(std::move(branch));
   }

   RTreeViewCollection GetViewCollection(std::string_view branchName);
};

} // namespace Experimental
} // namespace ROOT

#endif
