/// \file ROOT/RNTupleModel.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleModel
#define ROOT7_RNTupleModel

#include <ROOT/REntry.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

class RCollectionForest;

// clang-format off
/**
\class ROOT::Experimental::RForestModel
\ingroup NTuple
\brief The RForestModel encapulates the schema of a forest.

The forest model comprises a collection of hierarchically organized fields. From a frozen model, "entries"
can be extracted. For convenience, the model provides a default entry. Models have a unique model identifier
that faciliates checking whether entries are compatible with it (i.e.: have been extracted from that model).
A model needs to be frozen before it can be used to create an RForest.
*/
// clang-format on
class RForestModel {
   /// Hierarchy of fields consisting of simple types and collections (sub trees)
   std::unique_ptr<RFieldRoot> fRootField;
   /// Contains field values corresponding to the created top-level fields
   std::unique_ptr<REntry> fDefaultEntry;

public:
   RForestModel();
   RForestModel(const RForestModel&) = delete;
   RForestModel& operator =(const RForestModel&) = delete;
   ~RForestModel() = default;

   RForestModel* Clone();
   static std::unique_ptr<RForestModel> Create() { return std::make_unique<RForestModel>(); }

   /// Creates a new field and a corresponding tree value that is managed by a shared pointer.
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> MakeField(std::string_view fieldName, ArgsT&&... args) {
      auto field = std::make_unique<RField<T>>(fieldName);
      auto ptr = fDefaultEntry->AddValue<T>(field.get(), std::forward<ArgsT>(args)...);
      fRootField->Attach(std::move(field));
      return ptr;
   }

   /// Adds a field whose type is not known at compile time.  Thus there is no shared pointer returned.
   void AddField(std::unique_ptr<Detail::RFieldBase> field);

   template <typename T>
   void AddField(std::string_view fieldName, T* fromWhere) {
      auto field = std::make_unique<RField<T>>(fieldName);
      fDefaultEntry->CaptureValue(field->CaptureValue(fromWhere));
      fRootField->Attach(std::move(field));
   }

   template <typename T>
   T* Get(std::string_view fieldName) {
      return fDefaultEntry->Get<T>(fieldName);
   }

   /// Ingests a model for a sub collection and attaches it to the current model
   std::shared_ptr<RCollectionForest> MakeCollection(
      std::string_view fieldName,
      std::unique_ptr<RForestModel> collectionModel);

   RFieldRoot* GetRootField() { return fRootField.get(); }
   REntry* GetDefaultEntry() { return fDefaultEntry.get(); }
   std::unique_ptr<REntry> CreateEntry();
};

} // namespace Exerimental
} // namespace ROOT

#endif
