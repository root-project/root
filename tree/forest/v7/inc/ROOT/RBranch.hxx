/// \file ROOT/RBranch.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBranch
#define ROOT7_RBranch

#include <ROOT/RCargo.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeUtil.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

class RTreeSink;
class RTreeSource;

namespace Detail {

class RCargoBase;

// clang-format off
/**
\class ROOT::Experimental::RBranchBase
\ingroup Forest
\brief An RBranchBase translates read and write calls from/to underlying columns to/from cargo objects

The RBranchBase and its type-safe descendants provide the object to column mapper. They map C++ objects
to primitive columns, where the mapping is trivial for simple types such as 'double'.
*/
// clang-format on
class RBranchBase {
private:
   /// A branch on a trivial type that maps as-is to a single column
   bool fIsSimple;
   /// All branches have a main column. For nested branches, the main column is the index branch. Points into fColumns.
   RColumn *fPrincipalColumn;
   /// The columns are connected either to a sink or to a source (not to both)
   RColumnCollection_t fColumns;

protected:
   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual void DoAppend(const RCargoBase &cargo) { }
   virtual void DoRead(TreeIndex_t index, const RCargoBase &cargo) { };
   virtual void DoReadV(TreeIndex_t index, TreeIndex_t count, void *dst) { };

public:
   /// The constructor creates the underlying column objects and connects them to either a sink or a source.
   RBranchBase(std::string_view name, RTreeSink &sink);
   RBranchBase(std::string_view name, RTreeSource &source);
   virtual ~RBranchBase();

   /// Generates a cargo object of the branch type.
   virtual std::unique_ptr<RCargoBase> GenerateCargo() = 0;

   /// Write the value stored in cargo to a tree. The cargo object has to be of the same type as the branch.
   void Append(const RCargoBase &cargo) {
     if (!fIsSimple) {
        DoAppend(cargo);
        return;
     }
     fPrincipalColumn->Append(*(cargo.fPrincipalElement));
   }

   /// Populate a single cargo object with data from the tree, which needs to be of the fitting type.
   /// Reading copies data into the memory given by cargo.
   void Read(TreeIndex_t index, RCargoBase &cargo) {
      if (!fIsSimple) {
         DoRead(index, cargo);
         return;
      }
      fPrincipalColumn->Read(index, cargo.fPrincipalElement);
   }

   /// Type unsafe bulk read interface; dst must point to a vector of objects of the branch type.
   /// TODO(jblomer): can this be type safe?
   void ReadV(TreeIndex_t index, TreeIndex_t count, void *dst)
   {
      if (!fIsSimple) {
         DoReadV(index, count, dst);
         return;
      }
      fPrincipalColumn->ReadV(index, count, dst);
   }

   /// Only for simple types, let the memory enclosed in cargo simply point into the page buffer.
   /// The resulting cargo object may only be used for as long as no request to another item of this branch is made
   /// because another index might trigger a swap of the page buffer.
   /// The dst location must be an object of the branch type.
   void Map(TreeIndex_t index, void **dst) {
      if (!fIsSimple) {
         // TODO(jblomer)
      }
      fPrincipalColumn->Map(index, dst);
   }

   /// The number of elements in the principal column. For top level branches, the number of entries.
   TreeIndex_t GetNItems();

   /// Ensure that all written items are written from page buffers to the storage.
   void Flush();
};

} // namespace Detail


/// Supported types are implemented as template specializations
template <typename T>
class RBranch : public Detail::RBranchBase {
};

/// TODO(jblomer): template specializations

} // namespace Experimental
} // namespace ROOT

#endif
