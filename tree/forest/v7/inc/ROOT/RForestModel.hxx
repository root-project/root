/// \file ROOT/RForestModel.hxx
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

#ifndef ROOT7_RForestModel
#define ROOT7_RForestModel

#include <ROOT/RField.hxx>
#include <ROOT/RForestEntry.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeValue.hxx>

#include <TError.h>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RForestModel
\ingroup Forest
\brief The RForestModel encapulates the schema of a tree.

The tree model comprises a collection of hierarchically organized fields. From a frozen model, "entries"
can be extracted. For convenience, the model provides a default entry. Models have a unique model identifier
that faciliates checking whether entries are compatible with it (i.e.: have been extracted from that model).
A model needs to be frozen before it can be used to create an RTree.
*/
// clang-format on
class RForestModel {
   /// Hierarchy of fields consiting of simple types and collections (sub trees)
   RFieldRoot fRootField;
   /// Contains tree values corresponding to the created fields
   RForestEntry fDefaultEntry;

public:
   /// Adds a field whose type is not known at compile time.  Thus there is no shared pointer returned.
   void AddField(std::unique_ptr<Detail::RFieldBase> field);

   /// Creates a new field and a corresponding tree value that is managed by a shared pointer.
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> AddField(std::string_view fieldName, ArgsT&&... args) {
      auto field = std::make_unique<RField<T>>(fieldName);
      auto ptr = fDefaultEntry.AddValue<T>(field.get(), std::forward<ArgsT>(args)...);
      fRootField.Attach(std::move(field));
      return ptr;
   }

   template <typename T, typename... ArgsT>
   T& AddFieldRef(std::string_view fieldName, ArgsT&&... args) {
      return *AddField<T>(fieldName, std::forward<ArgsT>(args)...);
   }

   template <typename T>
   void CaptureField(std::string_view fieldName, T* fromWhere) {
      auto field = std::make_unique<RField<T>>(fieldName);
      fDefaultEntry.CaptureValue(field->CaptureValue(fromWhere));
      fRootField.Attach(std::move(field));
   }

   void AddCollection(std::string_view fieldName, std::shared_ptr<RForestModel> collectionModel);

   RFieldRoot* GetRootField() { return &fRootField; }
   RForestEntry* GetDefaultEntry() { return &fDefaultEntry; }
};

} // namespace Exerimental
} // namespace ROOT

#endif
