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
#include <ROOT/RCargo.hxx>
#include <ROOT/RTreeUtil.hxx>

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

The view owns a branch and its underlying columns in order to fill a cargo object with values from the
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
   //RCargo<T> fCargo;

public:
   RTreeView(std::unique_ptr<RBranch<T>> branch)
      : fBranch(std::move(branch))
      //, fCargo(fBranch.get())
   { }

   /// Use the branch to read the referenced element into the cargo object
   const T& operator ()(TreeIndex_t index);
};


//template <>
//class RTreeView<double> {
//private:
//  RColumn *fColumn;
//  RColumnElement<double> fColumnElement;
//
//public:
//  RTreeView(RBranch<double> *branch)
//    : fColumn(branch->GetPrincipalColumn())
//    , fColumnElement(nullptr)
//  {
//    std::cout << "Using optimized reading for double" << std::endl;
//  }
//
//  const double& operator ()(const RColumnPointer &p) {
//    fColumn->Map(p.GetIndex(), &fColumnElement);
//    return *(fColumnElement.GetPtr());
//  }
//
//  void ReadBulk(std::uint64_t start, std::uint64_t num, double* buf) {
//    fColumn->ReadV(start, num, buf);
//  }
//};
//
//
//template <>
//class RTreeView<OffsetColumn_t> {
//protected:
//  RBranch<OffsetColumn_t> *fBranch;
//
//private:
//  // For offset columns, read both the index and the one before to
//  // get the size TODO
//  OffsetColumn_t fOffsetPair[2];
//  RCargo<OffsetColumn_t> fCargo;
//
//public:
//  RTreeView(RBranch<OffsetColumn_t> *branch)
//    : fBranch(branch)
//    , fCargo(fBranch)
//  { }
//
//  RColumnRange GetRange(const RColumnPointer &p) {
//    if (p.GetIndex() == 0) {
//      fBranch->Read(0, &fCargo);
//      return RColumnRange(0, *fCargo.Get());
//    }
//    fBranch->Read(p.GetIndex() - 1, &fCargo);
//    OffsetColumn_t lower = *fCargo.Get();
//    fBranch->Read(p.GetIndex(), &fCargo);
//    return RColumnRange(lower, *fCargo.Get());
//  }
//
//  OffsetColumn_t operator ()(const RColumnPointer &p) {
//    return GetRange(p).GetSize();
//  }
//
//  void ReadBulk(std::uint64_t start, std::uint64_t num, OffsetColumn_t *buf) {
//    fBranch->ReadV(start, num, buf);
//  }
//};
//
//
//class RTreeViewCollection : public RTreeView<TreeIndex_t> {
//public:
//   template <typename T>
//   RTreeView<T> GetView(std::string_view name) {
//     // TODO not with raw pointer
//     auto branch = new RBranch<T>(fBranch->GetName() + "/" + std::string(name));
//     branch->GenerateColumns(fSource, nullptr);
//     return RTreeView<T>(branch);
//   }
//
//   RTreeViewCollection GetViewCollection(std::string_view branchName);
//};

} // namespace Experimental
} // namespace ROOT

#endif
