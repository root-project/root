/// \file ROOT/RNTupleModel.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleModel
#define ROOT_RNTupleModel

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldToken.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <string_view>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ROOT {

class RNTupleWriteOptions;
class RNTupleModel;
class RNTupleWriter;

namespace Experimental {
namespace Detail {
class RRawPtrWriteEntry;
} // namespace Detail
} // namespace Experimental

namespace Internal {
class RProjectedFields;

ROOT::RFieldZero &GetFieldZeroOfModel(RNTupleModel &model);
RProjectedFields &GetProjectedFieldsOfModel(RNTupleModel &model);

// clang-format off
/**
\class ROOT::Internal::RProjectedFields
\ingroup NTuple
\brief Container for the projected fields of an RNTupleModel

Projected fields are fields whose columns are reused from existing fields. Projected fields are not attached
to the model's zero field but form a separate hierarchy with their own zero field (which is stored in this class).
Only the real source fields are written to: projected fields are stored as metadata
(header) information only. Only top-level projected fields are supported because otherwise the layout of types
could be altered in unexpected ways.
This class owns the hierarchy of projected fields and keeps the mapping between them and their backing source fields.
*/
// clang-format on
class RProjectedFields {
public:
   /// The map keys are the projected target fields, the map values are the backing source fields
   /// Note that sub fields are treated individually and indepently of their parent field
   using FieldMap_t = std::unordered_map<const ROOT::RFieldBase *, const ROOT::RFieldBase *>;

private:
   explicit RProjectedFields(std::unique_ptr<ROOT::RFieldZero> fieldZero) : fFieldZero(std::move(fieldZero)) {}
   /// The projected fields are attached to this zero field
   std::unique_ptr<ROOT::RFieldZero> fFieldZero;
   /// Maps the source fields from fModel to the target projected fields attached to fFieldZero
   FieldMap_t fFieldMap;
   /// The model this set of projected fields belongs to
   const RNTupleModel *fModel;

   /// Asserts that the passed field is a valid target of the source field provided in the field map.
   /// Checks the field without looking into sub fields.
   RResult<void> EnsureValidMapping(const ROOT::RFieldBase *target, const FieldMap_t &fieldMap);

public:
   explicit RProjectedFields(const RNTupleModel &model)
      : fFieldZero(std::make_unique<ROOT::RFieldZero>()), fModel(&model)
   {
   }
   RProjectedFields(const RProjectedFields &) = delete;
   RProjectedFields(RProjectedFields &&) = default;
   RProjectedFields &operator=(const RProjectedFields &) = delete;
   RProjectedFields &operator=(RProjectedFields &&) = default;
   ~RProjectedFields() = default;

   /// Clones this container and all the projected fields it owns. `newModel` must be a clone of the model
   /// that this RProjectedFields was constructed with.
   std::unique_ptr<RProjectedFields> Clone(const RNTupleModel &newModel) const;

   ROOT::RFieldZero &GetFieldZero() { return *fFieldZero; }
   const ROOT::RFieldBase *GetSourceField(const ROOT::RFieldBase *target) const;
   /// Adds a new projected field. The field map needs to provide valid source fields of fModel for 'field'
   /// and each of its sub fields.
   RResult<void> Add(std::unique_ptr<ROOT::RFieldBase> field, const FieldMap_t &fieldMap);
   bool IsEmpty() const { return fFieldZero->begin() == fFieldZero->end(); }
};

} // namespace Internal

// clang-format off
/**
\class ROOT::RNTupleModel
\ingroup NTuple
\brief The RNTupleModel encapulates the schema of an RNTuple.

The RNTupleModel comprises a collection of hierarchically organized fields. From a model, "entries"
can be extracted or created. For convenience, the RNTupleModel provides a default entry unless it is created as a "bare model".
Models have a unique model identifier that facilitates checking whether entries are compatible with it
(i.e.: have been extracted from that model).

A model is subject to state transitions during its lifetime: it starts in a *building* state, in which fields can be
added and modified.  Once the schema is finalized, the model gets *frozen*.  Only frozen models can create entries.
From frozen, models move into an *expired* state. In this state, the model is only partially usable: it can be cloned
and queried, but it can't be unfrozen anymore and no new entries can be created.  This state is used for models
that were used for writing and are no longer connected to a page sink.

```
(Model gets created)
     |
     |       (passed to a Sink            (detached from
 ____v______  or explicitly    __________  Sink after     ___________
|           | frozen)         |          | writing)      |           |
| Building  |---------------->|  Frozen  |-------------->|  Expired  |
|___________|<----------------|__________|               |___________|
             (explicitly
              unfrozen)
```

*/
// clang-format on
class RNTupleModel {
   friend ROOT::RFieldZero &Internal::GetFieldZeroOfModel(RNTupleModel &);
   friend Internal::RProjectedFields &Internal::GetProjectedFieldsOfModel(RNTupleModel &);

public:
   /// User-provided function that describes the mapping of existing source fields to projected fields in terms
   /// of fully qualified field names. The mapping function is called with the qualified field names of the provided
   /// field and the subfields. It should return the qualified field names used as a mapping source.
   /// See AddProjectedFields() for more details.
   using FieldMappingFunc_t = std::function<std::string(const std::string &)>;

   class RUpdater;

private:
   /// The states a model can be in. Possible transitions are between kBuilding and kFrozen
   /// and from kFrozen to kExpired.
   /// See RNTupleModel for the state transition graph.
   enum class EState {
      kBuilding,
      kFrozen,
      kExpired
   };

   /// Hierarchy of fields consisting of simple types and collections (sub trees)
   std::unique_ptr<ROOT::RFieldZero> fFieldZero;
   /// Contains field values corresponding to the created top-level fields, as well as registered subfields
   std::unique_ptr<ROOT::REntry> fDefaultEntry;
   /// Keeps track of which field names are taken, including projected field names.
   std::unordered_set<std::string> fFieldNames;
   /// Free text set by the user
   std::string fDescription;
   /// The set of projected top-level fields
   std::unique_ptr<Internal::RProjectedFields> fProjectedFields;
   /// Keeps track of which subfields have been registered to be included in entries belonging to this model.
   std::unordered_set<std::string> fRegisteredSubfields;
   /// Every model has a unique ID to distinguish it from other models. Entries are linked to models via the ID.
   /// Cloned models get a new model ID. Expired models are cloned into frozen models.
   std::uint64_t fModelId = 0;
   /// Models have a separate schema ID to remember that the clone of a frozen model still has the same schema.
   std::uint64_t fSchemaId = 0;
   /// Changed by Freeze() / Unfreeze() and by the RUpdater.
   EState fModelState = EState::kBuilding;

   /// Checks that user-provided field names are valid in the context of this RNTupleModel.
   /// Throws an RException for invalid names, empty names (which is reserved for the zero field) and duplicate field
   /// names.
   void EnsureValidFieldName(std::string_view fieldName);

   /// Throws an RException if fFrozen is true
   void EnsureNotFrozen() const;

   /// Throws an RException if fDefaultEntry is nullptr
   void EnsureNotBare() const;

   /// The field name can be a top-level field or a nested field. Returns nullptr if the field is not in the model.
   ROOT::RFieldBase *FindField(std::string_view fieldName) const;

   /// Add a subfield to the provided entry. If `initializeValue` is false, a nullptr will be bound to the entry value
   /// (used in bare models).
   void AddSubfield(std::string_view fieldName, ROOT::REntry &entry, bool initializeValue = true) const;

   RNTupleModel(std::unique_ptr<ROOT::RFieldZero> fieldZero);

public:
   RNTupleModel(const RNTupleModel &) = delete;
   RNTupleModel &operator=(const RNTupleModel &) = delete;
   ~RNTupleModel() = default;

   std::unique_ptr<RNTupleModel> Clone() const;
   static std::unique_ptr<RNTupleModel> Create();
   static std::unique_ptr<RNTupleModel> Create(std::unique_ptr<ROOT::RFieldZero> fieldZero);
   /// Creates a "bare model", i.e. an RNTupleModel with no default entry
   static std::unique_ptr<RNTupleModel> CreateBare();
   /// Creates a "bare model", i.e. an RNTupleModel with no default entry, with the given field zero.
   static std::unique_ptr<RNTupleModel> CreateBare(std::unique_ptr<ROOT::RFieldZero> fieldZero);

   /// Creates a new field given a `name` or `{name, description}` pair and a
   /// corresponding, default-constructed value that is managed by a shared pointer.
   ///
   /// **Example: create some fields and fill an %RNTuple**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleModel.hxx>
   /// #include <ROOT/RNTupleWriter.hxx>
   /// using ROOT::RNTupleWriter;
   ///
   /// #include <vector>
   ///
   /// auto model = ROOT::RNTupleModel::Create();
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
   /// **Example: create a field with a description**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleModel.hxx>
   ///
   /// auto model = ROOT::RNTupleModel::Create();
   /// auto hadronFlavour = model->MakeField<float>(
   ///    "hadronFlavour", "flavour from hadron ghost clustering"
   /// );
   /// ~~~
   template <typename T>
   std::shared_ptr<T> MakeField(std::string_view name, std::string_view description = "")
   {
      EnsureNotFrozen();
      EnsureValidFieldName(name);
      auto field = std::make_unique<ROOT::RField<T>>(name);
      field->SetDescription(description);
      std::shared_ptr<T> ptr;
      if (fDefaultEntry)
         ptr = fDefaultEntry->AddValue<T>(*field);
      fFieldNames.insert(field->GetFieldName());
      fFieldZero->Attach(std::move(field));
      return ptr;
   }

   /// Adds a field whose type is not known at compile time. No shared pointer is returned in this case:
   /// pointers should be retrieved or bound via REntry.
   ///
   /// Throws an RException if the field is null.
   void AddField(std::unique_ptr<ROOT::RFieldBase> field);

   /// Register a subfield so it can be accessed directly from entries belonging to the model. Because registering a
   /// subfield does not fundamentally change the model, previously created entries will not be invalidated, nor
   /// modified in any way; a registered subfield is merely an accessor added to the default entry (if present) and any
   /// entries created afterwards. Note that previously created entries won't have this subfield added to them.
   ///
   /// Using models with registered subfields for writing is not allowed. Attempting to do so will result in an
   /// exception.
   ///
   /// Throws an RException if the provided subfield could not be found in the model.
   void RegisterSubfield(std::string_view qualifiedFieldName);

   /// Adds a top-level field based on existing fields.
   ///
   /// The mapping function takes one argument, which is a string containing the name of the projected field. The return
   /// value of the mapping function should be the name of the (existing) field onto which the projection is made.
   /// **Example**
   /// ~~~ {.cpp}
   /// auto model = RNTupleModel::Create();
   /// model->MakeField<float>("met");
   /// auto metProjection = ROOT::RFieldBase::Create("missingE", "float").Unwrap();
   /// model->AddProjectedField(std::move(metProjection), [](const std::string &) { return "met"; });
   /// ~~~
   ///
   /// Adding projections for collection fields is also possible, as long as they follow the same schema structure. For
   /// example, a projection of a collection of structs onto a collection of scalars is possible, but a projection of a
   /// collection of a collection of scalars onto a collection of scalars is not.
   ///
   /// In the case of projections for nested fields, the mapping function must provide a mapping for every nesting
   /// level.
   /// **Example**
   /// ~~~ {.cpp}
   /// struct P { int x, y; };
   ///
   /// auto model = RNTupleModel::Create();
   /// model->MakeField<std::vector<P>>("points");
   /// auto pxProjection = ROOT::RFieldBase::Create("pxs", "std::vector<int>").Unwrap();
   /// model->AddProjectedField(std::move(pxProjection), [](const std::string &fieldName) {
   ///   if (fieldName == "pxs")
   ///     return "points";
   ///   else
   ///     return "points._0.x";
   /// });
   /// ~~~
   ///
   /// Creating projections for fields containing `std::variant` or fixed-size arrays is unsupported.
   RResult<void> AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field, FieldMappingFunc_t mapping);

   /// Transitions an RNTupleModel from the *building* state to the *frozen* state, disabling adding additional fields
   /// and enabling creating entries from it. Freezing an already-frozen model is a no-op. Throws an RException if the
   /// model is in the *expired* state. See RNTupleModel for more detailed explanation on the state transitions.
   void Freeze();
   /// Transitions an RNTupleModel from the *frozen* state back to the *building* state, invalidating all previously
   /// created entries, re-enabling adding additional fields and disabling creating entries from it. Unfreezing a model
   /// that is already in the *building* state is a no-op. Throws an RException if the model is in the *expired* state.
   /// See RNTupleModel for a more detailed explanation on the state transitions.
   void Unfreeze();
   /// Transitions an RNTupleModel from the *frozen* state to the *expired* state, invalidating all previously created
   /// entries, disabling creating new entries from it and disabling further state transitions. Expiring a model that is
   /// already expired is a no-op. Throws an RException if the model is in the *building* state. See RNTupleModel for a
   /// more detailed explanation on the state transitions.
   void Expire();
   /// \see Expire()
   bool IsExpired() const { return fModelState == EState::kExpired; }
   /// \see Freeze()
   bool IsFrozen() const { return (fModelState == EState::kFrozen) || (fModelState == EState::kExpired); }
   /// \see CreateBare()
   bool IsBare() const { return !fDefaultEntry; }
   std::uint64_t GetModelId() const { return fModelId; }
   std::uint64_t GetSchemaId() const { return fSchemaId; }

   /// Creates a new entry with default values for each field.
   std::unique_ptr<REntry> CreateEntry() const;
   /// Creates a "bare entry", i.e. a entry with all null values. The user needs to explicitly call BindValue() or
   /// BindRawPtr() to set memory addresses before serializing / deserializing the entry.
   std::unique_ptr<REntry> CreateBareEntry() const;
   std::unique_ptr<Experimental::Detail::RRawPtrWriteEntry> CreateRawPtrWriteEntry() const;
   /// Creates a token to be used in REntry methods to address a field present in the entry
   ROOT::RFieldToken GetToken(std::string_view fieldName) const;
   /// Calls the given field's CreateBulk() method. Throws an RException if no field with the given name exists.
   ROOT::RFieldBase::RBulkValues CreateBulk(std::string_view fieldName) const;

   /// Retrieves the default entry of this model.
   /// Throws an RException if this is a bare model (i.e. if it was created with CreateBare()).
   REntry &GetDefaultEntry();
   /// \see GetDefaultEntry()
   const REntry &GetDefaultEntry() const;

   /// Retrieves the field zero of this model, i.e. the root of the field hierarchy.
   /// This may be used to make adjustments on the field hierarchy before the model is frozen.
   ROOT::RFieldZero &GetMutableFieldZero();
   /// Retrieves the field zero of this model, i.e. the root of the field hierarchy.
   const ROOT::RFieldZero &GetConstFieldZero() const { return *fFieldZero; }
   /// Retrieves the field with fully-qualified name `fieldName`.
   /// Dot-separated names are used to walk down the field hierarchy: e.g. `"parent.child"` should
   /// be used to retrieve a field with name `"child"` whose parent is the top-level field with name `"parent"`.
   /// Throws an RException if no field is found with the given name.
   ROOT::RFieldBase &GetMutableField(std::string_view fieldName);
   /// \see GetMutableField()
   const ROOT::RFieldBase &GetConstField(std::string_view fieldName) const;

   const std::string &GetDescription() const { return fDescription; }
   void SetDescription(std::string_view description);

   /// Get the names of the fields currently present in the model, including projected fields. Registered subfields
   /// are not included, use GetRegisteredSubfieldnames() for this.
   const std::unordered_set<std::string> &GetFieldNames() const { return fFieldNames; }
   /// Get the (qualified) names of subfields that have been registered (via RegisterSubfield()) to be included in
   /// entries from this model.
   const std::unordered_set<std::string> &GetRegisteredSubfieldNames() const { return fRegisteredSubfields; }

   /// Estimate the memory usage for this model during writing
   ///
   /// This will return an estimate in bytes for the internal page and compression buffers. The value should be
   /// understood per sequential RNTupleWriter or per RNTupleFillContext created for an RNTupleParallelWriter
   /// constructed with this model.
   std::size_t EstimateWriteMemoryUsage(const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions()) const;
};

namespace Internal {

// clang-format off
/**
\class ROOT::Internal::RNTupleModelChangeset
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
   std::vector<ROOT::RFieldBase *> fAddedFields;
   /// Points to the projected fields in fModel that were added as part of an updater transaction
   std::vector<ROOT::RFieldBase *> fAddedProjectedFields;

   RNTupleModelChangeset(RNTupleModel &model) : fModel(model) {}
   bool IsEmpty() const { return fAddedFields.empty() && fAddedProjectedFields.empty(); }

   void AddField(std::unique_ptr<ROOT::RFieldBase> field);

   /// \see RNTupleModel::AddProjectedField()
   ROOT::RResult<void>
   AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field, RNTupleModel::FieldMappingFunc_t mapping);
};

} // namespace Internal

/// A model is usually immutable after passing it to an `RNTupleWriter`. However, for the rare
/// cases that require changing the model after the fact, `RUpdater` provides limited support for
/// incremental updates, e.g. addition of new fields.
///
/// See `RNTupleWriter::CreateModelUpdater()` for an example.
class RNTupleModel::RUpdater {
private:
   ROOT::RNTupleWriter &fWriter;
   Internal::RNTupleModelChangeset fOpenChangeset;
   std::uint64_t fNewModelId = 0; ///< The model ID after committing

public:
   explicit RUpdater(ROOT::RNTupleWriter &writer);
   ~RUpdater() { CommitUpdate(); }
   /// Begin a new set of alterations to the underlying model. As a side effect, all REntry
   /// instances related to the model are invalidated.
   void BeginUpdate();
   /// Commit changes since the last call to `BeginUpdate()`. All the invalidated REntries remain
   /// invalid. `CreateEntry()` or `CreateBareEntry()` can be used to create an REntry that
   /// matches the new model. Upon completion, `BeginUpdate()` can be called again to begin a new set of changes.
   void CommitUpdate();

   template <typename T>
   std::shared_ptr<T> MakeField(std::string_view name, std::string_view description = "")
   {
      auto objPtr = fOpenChangeset.fModel.MakeField<T>(name, description);
      auto fieldZero = fOpenChangeset.fModel.fFieldZero.get();
      auto it =
         std::find_if(fieldZero->begin(), fieldZero->end(), [&](const auto &f) { return f.GetFieldName() == name; });
      R__ASSERT(it != fieldZero->end());
      fOpenChangeset.fAddedFields.emplace_back(&(*it));
      return objPtr;
   }

   void AddField(std::unique_ptr<ROOT::RFieldBase> field);

   /// \see RNTupleModel::AddProjectedField()
   RResult<void> AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field, FieldMappingFunc_t mapping);
};

} // namespace ROOT

#endif
