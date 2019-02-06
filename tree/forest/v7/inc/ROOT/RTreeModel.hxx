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

#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeEntry.hxx>
#include <ROOT/RTreeField.hxx>
#include <ROOT/RTreeValue.hxx>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RTreeModel
\ingroup Forest
\brief The RTreeModel encapulates the schema of a tree.

The tree model comprises a collection of hierarchically organized fields. From a frozen model, "entries"
can be extracted. For convenience, the model provides a default entry. Models have a unique model identifier
that faciliates checking whether entries are compatible with it (i.e.: have been extracted from that model).
A model needs to be frozen before it can be used to create an RTree.
*/
// clang-format on
class RTreeModel {
   /// Hierarchy of fields consiting of simple types and collections (sub trees)
   RTreeFieldRoot fRootField;
   /// Contains tree values corresponding to the created fields
   RTreeEntry fDefaultEntry;

public:
   /// Creates a new field and a corresponding tree value.
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> AddField(std::string_view fieldName, ArgsT&&... args) {
      auto field = std::make_unique<RTreeField<T>>(fieldName);
      RTreeValue<T>* value = static_cast<RTreeValue<T>*>(field->GenerateValue(std::forward<ArgsT>(args)...));
      fRootField.Attach(std::move(field));

      auto valuePtr = value->GetSharedPtr();
      fDefaultEntry.TakeValue(value);
      return valuePtr;
   }

   /// Mounts an existing model as a sub tree, which allows for composing of tree models
   std::shared_ptr<RTreeValueCollection> TreeFieldCollection(std::string_view fieldName, std::shared_ptr<RTreeModel> subModel);

   RTreeFieldRoot* GetRootField() { return &fRootField; }
   RTreeEntry* GetDefaultEntry() { return &fDefaultEntry; }
};

} // namespace Exerimental
} // namespace ROOT

#endif
