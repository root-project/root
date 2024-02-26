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
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <string_view>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ROOT {
namespace Experimental {

class RNTupleCollectionWriter;
class RNTupleModel;
class RNTupleWriter;

namespace Internal {
class RPageSinkBuf;

// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleModelChangeset
\ingroup NTuple
\brief The incremental changes to a `RNTupleModel`

Represents a set of alterations to a `RNTupleModel` that happened after the model is used to initialize a `RPageSink`
instance. This object can be used to communicate metadata updates to a `RPageSink`.
You will not normally use this directly; see `RNTupleModel::RUpdater` instead.
*/
// clang-format on
struct RNTupleModelChangeset {
   RNTupleModel &fModel;
   /// Points to the fields in fModel that were added as part of an updater transaction
   std::vector<RFieldBase *> fAddedFields;
   /// Points to the projected fields in fModel that were added as part of an updater transaction
   std::vector<RFieldBase *> fAddedProjectedFields;

   RNTupleModelChangeset(RNTupleModel &model) : fModel(model) {}
   bool IsEmpty() const { return fAddedFields.empty() && fAddedProjectedFields.empty(); }
};

} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleModel
\ingroup NTuple
\brief The RNTupleModel encapulates the schema of an ntuple.

The ntuple model comprises a collection of hierarchically organized fields. From a model, "entries"
can be extracted. For convenience, the model provides a default entry unless it is created as a "bare model".
Models have a unique model identifier that faciliates checking whether entries are compatible with it
(i.e.: have been extracted from that model).

A model is subject to a state transition during its lifetime: it starts in a building state, in which fields can be
added and modified.  Once the schema is finalized, the model gets frozen.  Only frozen models can create entries.
*/
// clang-format on
class RNTupleModel {
public:
   /// A wrapper over a field name and an optional description; used in `AddField()` and `RUpdater::AddField()`
   struct NameWithDescription_t {
      NameWithDescription_t(const char *name) : fName(name) {}
      NameWithDescription_t(const std::string &name) : fName(name) {}
      NameWithDescription_t(std::string_view name) : fName(name) {}
      NameWithDescription_t(std::string_view name, std::string_view descr) : fName(name), fDescription(descr) {}

      std::string_view fName;
      std::string_view fDescription = "";
   };

   /// Projected fields are fields whose columns are reused from existing fields. Projected fields are not attached
   /// to the models zero field.  Only the real source fields are written to, projected fields are stored as meta-data
   /// (header) information only.  Only top-level projected fields are supported because otherwise the layout of types
   /// could be altered in unexpected ways.
   /// All projected fields and the source fields used to back them are kept in this class.
   class RProjectedFields {
   public:
      /// The map keys are the projected target fields, the map values are the backing source fields
      /// Note that sub fields are treated individually and indepently of their parent field
      using FieldMap_t = std::unordered_map<const RFieldBase *, const RFieldBase *>;

   private:
      explicit RProjectedFields(std::unique_ptr<RFieldZero> fieldZero) : fFieldZero(std::move(fieldZero)) {}
      /// The projected fields are attached to this zero field
      std::unique_ptr<RFieldZero> fFieldZero;
      /// Maps the source fields from fModel to the target projected fields attached to fFieldZero
      FieldMap_t fFieldMap;
      /// The model this set of projected fields belongs to
      const RNTupleModel *fModel;

      /// Asserts that the passed field is a valid target of the source field provided in the field map.
      /// Checks the field without looking into sub fields.
      RResult<void> EnsureValidMapping(const RFieldBase *target, const FieldMap_t &fieldMap);

   public:
      explicit RProjectedFields(const RNTupleModel *model) : fFieldZero(std::make_unique<RFieldZero>()), fModel(model)
      {
      }
      RProjectedFields(const RProjectedFields &) = delete;
      RProjectedFields(RProjectedFields &&) = default;
      RProjectedFields &operator=(const RProjectedFields &) = delete;
      RProjectedFields &operator=(RProjectedFields &&) = default;
      ~RProjectedFields() = default;

      /// The new model needs to be a clone of fModel
      std::unique_ptr<RProjectedFields> Clone(const RNTupleModel *newModel) const;

      RFieldZero *GetFieldZero() const { return fFieldZero.get(); }
      const RFieldBase *GetSourceField(const RFieldBase *target) const;
      /// Adds a new projected field. The field map needs to provide valid source fields of fModel for 'field'
      /// and each of its sub fields.
      RResult<void> Add(std::unique_ptr<RFieldBase> field, const FieldMap_t &fieldMap);
      bool IsEmpty() const { return fFieldZero->begin() == fFieldZero->end(); }
   };

   /// A model is usually immutable after passing it to an `RNTupleWriter`. However, for the rare
   /// cases that require changing the model after the fact, `RUpdater` provides limited support for
   /// incremental updates, e.g. addition of new fields.
   ///
   /// See `RNTupleWriter::CreateModelUpdater()` for an example.
   class RUpdater {
   private:
      RNTupleWriter &fWriter;
      Internal::RNTupleModelChangeset fOpenChangeset;
      std::uint64_t fNewModelId = 0; ///< The model ID after committing

   public:
      explicit RUpdater(RNTupleWriter &writer);
      ~RUpdater() { CommitUpdate(); }
      /// Begin a new set of alterations to the underlying model. As a side effect, all `REntry` instances related to
      /// the model are invalidated.
      void BeginUpdate();
      /// Commit changes since the last call to `BeginUpdate()`. All the invalidated `REntry`s remain invalid.
      /// `CreateEntry()` or `CreateBareEntry()` can be used to create an `REntry` that matching the new model.
      /// Upon completion, `BeginUpdate()` can be called again to begin a new set of changes.
      void CommitUpdate();

      template <typename T, typename... ArgsT>
      std::shared_ptr<T> MakeField(const NameWithDescription_t &fieldNameDesc, ArgsT &&...args)
      {
         auto objPtr = fOpenChangeset.fModel.MakeField<T>(fieldNameDesc, std::forward<ArgsT>(args)...);
         auto fieldZero = fOpenChangeset.fModel.fFieldZero.get();
         auto it = std::find_if(fieldZero->begin(), fieldZero->end(),
                                [&](const auto &f) { return f.GetFieldName() == fieldNameDesc.fName; });
         R__ASSERT(it != fieldZero->end());
         fOpenChangeset.fAddedFields.emplace_back(&(*it));
         return objPtr;
      }

      void AddField(std::unique_ptr<RFieldBase> field);

      RResult<void>
      AddProjectedField(std::unique_ptr<RFieldBase> field, std::function<std::string(const std::string &)> mapping);
   };

private:
   /// Hierarchy of fields consisting of simple types and collections (sub trees)
   std::unique_ptr<RFieldZero> fFieldZero;
   /// Contains field values corresponding to the created top-level fields
   std::unique_ptr<REntry> fDefaultEntry;
   /// Keeps track of which field names are taken, including projected field names.
   std::unordered_set<std::string> fFieldNames;
   /// Free text set by the user
   std::string fDescription;
   /// The set of projected top-level fields
   std::unique_ptr<RProjectedFields> fProjectedFields;
   /// Every model has a unique ID to distinguish it from other models. Entries are linked to models via the ID.
   /// Cloned models get a new model ID.
   std::uint64_t fModelId = 0;
   /// Changed by Freeze() / Unfreeze() and by the RUpdater.
   bool fIsFrozen = false;

   /// Checks that user-provided field names are valid in the context
   /// of this NTuple model. Throws an RException for invalid names.
   void EnsureValidFieldName(std::string_view fieldName);

   /// Throws an RException if fFrozen is true
   void EnsureNotFrozen() const;

   /// Throws an RException if fDefaultEntry is nullptr
   void EnsureNotBare() const;

   /// The field name can be a top-level field or a nested field. Returns nullptr if the field is not in the model.
   RFieldBase *FindField(std::string_view fieldName) const;

   RNTupleModel(std::unique_ptr<RFieldZero> fieldZero);

public:
   RNTupleModel(const RNTupleModel&) = delete;
   RNTupleModel& operator =(const RNTupleModel&) = delete;
   ~RNTupleModel() = default;

   std::unique_ptr<RNTupleModel> Clone() const;
   static std::unique_ptr<RNTupleModel> Create();
   static std::unique_ptr<RNTupleModel> Create(std::unique_ptr<RFieldZero> fieldZero);
   /// A bare model has no default entry
   static std::unique_ptr<RNTupleModel> CreateBare();
   static std::unique_ptr<RNTupleModel> CreateBare(std::unique_ptr<RFieldZero> fieldZero);

   /// Creates a new field given a `name` or `{name, description}` pair and a
   /// corresponding value that is managed by a shared pointer.
   ///
   /// **Example: create some fields and fill an %RNTuple**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleModel.hxx>
   /// #include <ROOT/RNTupleWriter.hxx>
   /// using ROOT::Experimental::RNTupleModel;
   /// using ROOT::Experimental::RNTupleWriter;
   ///
   /// #include <vector>
   ///
   /// auto model = RNTupleModel::Create();
   /// auto pt = model->MakeField<float>("pt");
   /// auto vec = model->MakeField<std::vector<int>>("vec");
   ///
   /// // The RNTuple is written to disk when the RNTupleWriter goes out of scope
   /// {
   ///    auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", "myFile.root");
   ///    for (int i = 0; i < 100; i++) {
   ///       *pt = static_cast<float>(i);
   ///       *vec = {i, i+1, i+2};
   ///       writer->Fill();
   ///    }
   /// }
   /// ~~~
   ///
   /// **Example: create a field with an initial value**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleModel.hxx>
   /// using ROOT::Experimental::RNTupleModel;
   ///
   /// auto model = RNTupleModel::Create();
   /// // pt's initial value is 42.0
   /// auto pt = model->MakeField<float>("pt", 42.0);
   /// ~~~
   /// **Example: create a field with a description**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleModel.hxx>
   /// using ROOT::Experimental::RNTupleModel;
   ///
   /// auto model = RNTupleModel::Create();
   /// auto hadronFlavour = model->MakeField<float>({
   ///    "hadronFlavour", "flavour from hadron ghost clustering"
   /// });
   /// ~~~
   template <typename T, typename... ArgsT>
   std::shared_ptr<T> MakeField(const NameWithDescription_t &fieldNameDesc, ArgsT &&...args)
   {
      EnsureNotFrozen();
      EnsureValidFieldName(fieldNameDesc.fName);
      auto field = std::make_unique<RField<T>>(fieldNameDesc.fName);
      field->SetDescription(fieldNameDesc.fDescription);
      std::shared_ptr<T> ptr;
      if (fDefaultEntry)
         ptr = fDefaultEntry->AddValue<T>(*field, std::forward<ArgsT>(args)...);
      fFieldZero->Attach(std::move(field));
      return ptr;
   }

   /// Adds a field whose type is not known at compile time.  Thus there is no shared pointer returned.
   ///
   /// Throws an exception if the field is null.
   void AddField(std::unique_ptr<RFieldBase> field);

   /// Adds a top-level field based on existing fields. The mapping function is called with the qualified field names
   /// of the provided field and the subfields.  It should return the qualified field names used as a mapping source.
   /// Projected fields can only be used for models used to write data.
   RResult<void>
   AddProjectedField(std::unique_ptr<RFieldBase> field, std::function<std::string(const std::string &)> mapping);
   const RProjectedFields &GetProjectedFields() const { return *fProjectedFields; }

   void Freeze();
   void Unfreeze();
   bool IsFrozen() const { return fIsFrozen; }
   std::uint64_t GetModelId() const { return fModelId; }

   /// Ingests a model for a sub collection and attaches it to the current model
   ///
   /// Throws an exception if collectionModel is null.
   std::shared_ptr<RNTupleCollectionWriter>
   MakeCollection(std::string_view fieldName, std::unique_ptr<RNTupleModel> collectionModel);

   std::unique_ptr<REntry> CreateEntry() const;
   /// In a bare entry, all values point to nullptr. The resulting entry shall use BindValue() in order
   /// set memory addresses to be serialized / deserialized
   std::unique_ptr<REntry> CreateBareEntry() const;
   /// Calls the given field's CreateBulk() method. Throws an exception if no field with the given name exists.
   RFieldBase::RBulk CreateBulk(std::string_view fieldName) const;

   REntry &GetDefaultEntry();
   const REntry &GetDefaultEntry() const;

   /// Non-const access to the root field is used to commit clusters during writing
   /// and to set the on-disk field IDs when connecting a model to a page source or sink.
   RFieldZero &GetFieldZero();
   const RFieldZero &GetFieldZero() const { return *fFieldZero; }
   const RFieldBase &GetField(std::string_view fieldName) const;

   std::string GetDescription() const { return fDescription; }
   void SetDescription(std::string_view description);
};

} // namespace Experimental
} // namespace ROOT

#endif
