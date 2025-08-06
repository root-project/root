/// \file ROOT/RFieldBase.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RFieldBase
#define ROOT_RFieldBase

#include <ROOT/RColumn.hxx>
#include <ROOT/RCreateFieldOptions.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <atomic>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <typeinfo>
#include <type_traits>
#include <vector>

namespace ROOT {

class REntry;
class RFieldBase;
class RClassField;

namespace Detail {
class RFieldVisitor;
} // namespace Detail

namespace Experimental {

namespace Detail {
class RRawPtrWriteEntry;
} // namespace Detail

} // namespace Experimental

namespace Internal {

class RPageSink;
class RPageSource;
struct RFieldCallbackInjector;
struct RFieldRepresentationModifier;

// TODO(jblomer): find a better way to not have these methods in the RFieldBase public API
void CallFlushColumnsOnField(RFieldBase &);
void CallCommitClusterOnField(RFieldBase &);
void CallConnectPageSinkOnField(RFieldBase &, ROOT::Internal::RPageSink &, ROOT::NTupleSize_t firstEntry = 0);
void CallConnectPageSourceOnField(RFieldBase &, ROOT::Internal::RPageSource &);
ROOT::RResult<std::unique_ptr<ROOT::RFieldBase>>
CallFieldBaseCreate(const std::string &fieldName, const std::string &typeName, const ROOT::RCreateFieldOptions &options,
                    const ROOT::RNTupleDescriptor *desc, ROOT::DescriptorId_t fieldId);

} // namespace Internal

// clang-format off
/**
\class ROOT::RFieldBase
\ingroup NTuple
\brief A field translates read and write calls from/to underlying columns to/from tree values

A field is a serializable C++ type or a container for a collection of subfields. The RFieldBase and its
type-safe descendants provide the object to column mapper. They map C++ objects to primitive columns.  The
mapping is trivial for simple types such as 'double'. Complex types resolve to multiple primitive columns.
The field knows based on its type and the field name the type(s) and name(s) of the columns.

Note: the class hierarchy starting at RFieldBase is not meant to be extended by user-provided child classes.
This is and can only be partially enforced through C++.
*/
// clang-format on
class RFieldBase {
   friend class ROOT::RClassField;                             // to mark members as artificial
   friend class ROOT::Experimental::Detail::RRawPtrWriteEntry; // to call Append()
   friend struct ROOT::Internal::RFieldCallbackInjector;       // used for unit tests
   friend struct ROOT::Internal::RFieldRepresentationModifier; // used for unit tests
   friend void Internal::CallFlushColumnsOnField(RFieldBase &);
   friend void Internal::CallCommitClusterOnField(RFieldBase &);
   friend void Internal::CallConnectPageSinkOnField(RFieldBase &, ROOT::Internal::RPageSink &, ROOT::NTupleSize_t);
   friend void Internal::CallConnectPageSourceOnField(RFieldBase &, ROOT::Internal::RPageSource &);
   friend ROOT::RResult<std::unique_ptr<ROOT::RFieldBase>>
   Internal::CallFieldBaseCreate(const std::string &fieldName, const std::string &typeName,
                                 const ROOT::RCreateFieldOptions &options, const ROOT::RNTupleDescriptor *desc,
                                 ROOT::DescriptorId_t fieldId);

   using ReadCallback_t = std::function<void(void *)>;

protected:
   /// A functor to release the memory acquired by CreateValue() (memory and constructor).
   /// This implementation works for types with a trivial destructor. More complex fields implement a derived deleter.
   /// The deleter is operational without the field object and thus can be used to destruct/release a value after
   /// the field has been destructed.
   class RDeleter {
   public:
      virtual ~RDeleter() = default;
      virtual void operator()(void *objPtr, bool dtorOnly)
      {
         if (!dtorOnly)
            operator delete(objPtr);
      }
   };

   /// A deleter for templated RFieldBase descendents where the value type is known.
   template <typename T>
   class RTypedDeleter : public RDeleter {
   public:
      void operator()(void *objPtr, bool dtorOnly) final
      {
         std::destroy_at(static_cast<T *>(objPtr));
         RDeleter::operator()(objPtr, dtorOnly);
      }
   };

   // We cannot directly use RFieldBase::RDeleter as a shared pointer deleter due to splicing. We use this
   // wrapper class to store a polymorphic pointer to the actual deleter.
   struct RSharedPtrDeleter {
      std::unique_ptr<RFieldBase::RDeleter> fDeleter;
      void operator()(void *objPtr) { fDeleter->operator()(objPtr, false /* dtorOnly*/); }
      explicit RSharedPtrDeleter(std::unique_ptr<RFieldBase::RDeleter> deleter) : fDeleter(std::move(deleter)) {}
   };

public:
   static constexpr std::uint32_t kInvalidTypeVersion = -1U;
   enum {
      /// No constructor needs to be called, i.e. any bit pattern in the allocated memory represents a valid type
      /// A trivially constructible field has a no-op ConstructValue() implementation
      kTraitTriviallyConstructible = 0x01,
      /// The type is cleaned up just by freeing its memory. I.e. the destructor performs a no-op.
      kTraitTriviallyDestructible = 0x02,
      /// A field of a fundamental type that can be directly mapped via RField<T>::Map(), i.e. maps as-is to a single
      /// column
      kTraitMappable = 0x04,
      /// The TClass checksum is set and valid
      kTraitTypeChecksum = 0x08,
      /// This field is an instance of RInvalidField and can be safely `static_cast` to it
      kTraitInvalidField = 0x10,
      /// This field is a user defined type that was missing dictionaries and was reconstructed from the on-disk
      /// information
      kTraitEmulatedField = 0x20,

      /// Shorthand for types that are both trivially constructible and destructible
      kTraitTrivialType = kTraitTriviallyConstructible | kTraitTriviallyDestructible
   };

   using ColumnRepresentation_t = std::vector<ROOT::ENTupleColumnType>;

   /// During its lifetime, a field undergoes the following possible state transitions:
   ///
   ///  [*] --> Unconnected --> ConnectedToSink ----
   ///               |      |                      |
   ///               |      --> ConnectedToSource ---> [*]
   ///               |                             |
   ///               -------------------------------
   enum class EState {
      kUnconnected,
      kConnectedToSink,
      kConnectedToSource
   };

   // clang-format off
   /**
   \class ROOT::RFieldBase::RColumnRepresentations
   \ingroup NTuple
   \brief The list of column representations a field can have.

   Some fields have multiple possible column representations, e.g. with or without split encoding.
   All column representations supported for writing also need to be supported for reading. In addition,
   fields can support extra column representations for reading only, e.g. a 64bit integer reading from a
   32bit column.
   The defined column representations must be supported by corresponding column packing/unpacking implementations,
   i.e. for the example above, the unpacking of 32bit ints to 64bit pages must be implemented in RColumnElement.hxx
   */
   // clang-format on
   class RColumnRepresentations {
   public:
      /// A list of column representations
      using Selection_t = std::vector<ColumnRepresentation_t>;

      RColumnRepresentations();
      RColumnRepresentations(const Selection_t &serializationTypes, const Selection_t &deserializationExtraTypes);

      /// The first column list from `fSerializationTypes` is the default for writing.
      const ColumnRepresentation_t &GetSerializationDefault() const { return fSerializationTypes[0]; }
      const Selection_t &GetSerializationTypes() const { return fSerializationTypes; }
      const Selection_t &GetDeserializationTypes() const { return fDeserializationTypes; }

   private:
      Selection_t fSerializationTypes;
      /// The union of the serialization types and the deserialization extra types passed during construction.
      /// Duplicates the serialization types list but the benefit is that GetDeserializationTypes() does not need to
      /// compile the list.
      Selection_t fDeserializationTypes;
   }; // class RColumnRepresentations

   class RValue;
   class RBulkValues;

private:
   /// The field name relative to its parent field
   std::string fName;
   /// The C++ type captured by this field
   std::string fType;
   /// The role of this field in the data model structure
   ROOT::ENTupleStructure fStructure;
   /// For fixed sized arrays, the array length
   std::size_t fNRepetitions;
   /// A field qualifies as simple if it is mappable (which implies it has a single principal column),
   /// and it is not an artificial field and has no post-read callback
   bool fIsSimple;
   /// A field that is not backed on disk but computed, e.g. a default-constructed missing field or
   /// a field whose data is created by I/O customization rules. Subfields of artificial fields are
   /// artificial, too.
   bool fIsArtificial = false;
   /// When the columns are connected to a page source or page sink, the field represents a field id in the
   /// corresponding RNTuple descriptor. This on-disk ID is set in RPageSink::Create() for writing and by
   /// RFieldDescriptor::CreateField() when recreating a field / model from the stored descriptor.
   ROOT::DescriptorId_t fOnDiskId = ROOT::kInvalidDescriptorId;
   /// Free text set by the user
   std::string fDescription;
   /// Changed by ConnectTo[Sink,Source], reset by Clone()
   EState fState = EState::kUnconnected;

   void InvokeReadCallbacks(void *target)
   {
      for (const auto &func : fReadCallbacks)
         func(target);
   }

   /// Translate an entry index to a column element index of the principal column and vice versa. These functions
   /// take into account the role and number of repetitions on each level of the field hierarchy as follows:
   /// - Top level fields: element index == entry index
   /// - Record fields propagate their principal column index to the principal columns of direct descendant fields
   /// - Collection and variant fields set the principal column index of their children to 0
   ///
   /// The column element index also depends on the number of repetitions of each field in the hierarchy, e.g., given a
   /// field with type `std::array<std::array<float, 4>, 2>`, this function returns 8 for the innermost field.
   ROOT::NTupleSize_t EntryToColumnElementIndex(ROOT::NTupleSize_t globalIndex) const;

   /// Flushes data from active columns
   void FlushColumns();
   /// Flushes data from active columns to disk and calls CommitClusterImpl()
   void CommitCluster();
   /// Fields and their columns live in the void until connected to a physical page storage.  Only once connected, data
   /// can be read or written.  In order to find the field in the page storage, the field's on-disk ID has to be set.
   /// \param firstEntry The global index of the first entry with on-disk data for the connected field
   void ConnectPageSink(ROOT::Internal::RPageSink &pageSink, ROOT::NTupleSize_t firstEntry = 0);
   /// Connects the field and its subfield tree to the given page source. Once connected, data can be read.
   /// Only unconnected fields may be connected, i.e. the method is not idempotent. The field ID has to be set prior to
   /// calling this function. For subfields, a field ID may or may not be set. If the field ID is unset, it will be
   /// determined using the page source descriptor, based on the parent field ID and the subfield name.
   void ConnectPageSource(ROOT::Internal::RPageSource &pageSource);

   void SetArtificial()
   {
      fIsSimple = false;
      fIsArtificial = true;
      for (auto &field : fSubfields) {
         field->SetArtificial();
      }
   }

protected:
   struct RBulkSpec;

   /// Collections and classes own subfields
   std::vector<std::unique_ptr<RFieldBase>> fSubfields;
   /// Subfields point to their mother field
   RFieldBase *fParent;
   /// All fields that have columns have a distinct main column. E.g., for simple fields (`float`, `int`, ...), the
   /// principal column corresponds to the field type. For collection fields except fixed-sized arrays,
   /// the main column is the offset field.  Class fields have no column of their own.
   /// When reading, points to any column of the column team of the active representation. Usually, this is just
   /// the first column.
   /// When writing, points to the first column index of the currently active (not suppressed) column representation.
   ROOT::Internal::RColumn *fPrincipalColumn = nullptr;
   /// Some fields have a second column in its column representation. In this case, `fAuxiliaryColumn` points into
   /// `fAvailableColumns` to the column that immediately follows the column `fPrincipalColumn` points to.
   ROOT::Internal::RColumn *fAuxiliaryColumn = nullptr;
   /// The columns are connected either to a sink or to a source (not to both); they are owned by the field.
   /// Contains all columns of all representations in order of representation and column index.
   std::vector<std::unique_ptr<ROOT::Internal::RColumn>> fAvailableColumns;
   /// Properties of the type that allow for optimizations of collections of that type
   std::uint32_t fTraits = 0;
   /// A typedef or using name that was used when creating the field
   std::string fTypeAlias;
   /// List of functions to be called after reading a value
   std::vector<ReadCallback_t> fReadCallbacks;
   /// C++ type version cached from the descriptor after a call to ConnectPageSource()
   std::uint32_t fOnDiskTypeVersion = kInvalidTypeVersion;
   /// TClass checksum cached from the descriptor after a call to ConnectPageSource(). Only set
   /// for classes with dictionaries.
   std::uint32_t fOnDiskTypeChecksum = 0;
   /// Pointers into the static vector returned by RColumnRepresentations::GetSerializationTypes() when
   /// SetColumnRepresentatives() is called. Otherwise (if empty) GetColumnRepresentatives() returns a vector
   /// with a single element, the default representation.  Always empty for artificial fields.
   std::vector<std::reference_wrapper<const ColumnRepresentation_t>> fColumnRepresentatives;

   /// Factory method for the field's type. The caller owns the returned pointer
   void *CreateObjectRawPtr() const;

   /// Helpers for generating columns. We use the fact that most fields have the same C++/memory types
   /// for all their column representations.
   /// Where possible, we call the helpers not from the header to reduce compilation time.
   template <std::uint32_t ColumnIndexT, typename HeadT, typename... TailTs>
   void GenerateColumnsImpl(const ColumnRepresentation_t &representation, std::uint16_t representationIndex)
   {
      assert(ColumnIndexT < representation.size());
      auto &column = fAvailableColumns.emplace_back(
         ROOT::Internal::RColumn::Create<HeadT>(representation[ColumnIndexT], ColumnIndexT, representationIndex));

      // Initially, the first two columns become the active column representation
      if (representationIndex == 0 && !fPrincipalColumn) {
         fPrincipalColumn = column.get();
      } else if (representationIndex == 0 && !fAuxiliaryColumn) {
         fAuxiliaryColumn = column.get();
      } else {
         // We currently have no fields with more than 2 columns in its column representation
         R__ASSERT(representationIndex > 0);
      }

      if constexpr (sizeof...(TailTs))
         GenerateColumnsImpl<ColumnIndexT + 1, TailTs...>(representation, representationIndex);
   }

   /// For writing, use the currently set column representative
   template <typename... ColumnCppTs>
   void GenerateColumnsImpl()
   {
      if (fColumnRepresentatives.empty()) {
         fAvailableColumns.reserve(sizeof...(ColumnCppTs));
         GenerateColumnsImpl<0, ColumnCppTs...>(GetColumnRepresentations().GetSerializationDefault(), 0);
      } else {
         const auto N = fColumnRepresentatives.size();
         fAvailableColumns.reserve(N * sizeof...(ColumnCppTs));
         for (unsigned i = 0; i < N; ++i) {
            GenerateColumnsImpl<0, ColumnCppTs...>(fColumnRepresentatives[i].get(), i);
         }
      }
   }

   /// For reading, use the on-disk column list
   template <typename... ColumnCppTs>
   void GenerateColumnsImpl(const ROOT::RNTupleDescriptor &desc)
   {
      std::uint16_t representationIndex = 0;
      do {
         const auto &onDiskTypes = EnsureCompatibleColumnTypes(desc, representationIndex);
         if (onDiskTypes.empty())
            break;
         GenerateColumnsImpl<0, ColumnCppTs...>(onDiskTypes, representationIndex);
         fColumnRepresentatives.emplace_back(onDiskTypes);
         if (representationIndex > 0) {
            for (std::size_t i = 0; i < sizeof...(ColumnCppTs); ++i) {
               fAvailableColumns[i]->MergeTeams(
                  *fAvailableColumns[representationIndex * sizeof...(ColumnCppTs) + i].get());
            }
         }
         representationIndex++;
      } while (true);
   }

   /// Implementations in derived classes should return a static RColumnRepresentations object. The default
   /// implementation does not attach any columns to the field.
   virtual const RColumnRepresentations &GetColumnRepresentations() const;
   /// Implementations in derived classes should create the backing columns corresponding to the field type for
   /// writing. The default implementation does not attach any columns to the field.
   virtual void GenerateColumns() {}
   /// Implementations in derived classes should create the backing columns corresponding to the field type for reading.
   /// The default implementation does not attach any columns to the field. The method should check, using the page
   /// source and `fOnDiskId`, if the column types match and throw if they don't.
   virtual void GenerateColumns(const ROOT::RNTupleDescriptor & /*desc*/) {}
   /// Returns the on-disk column types found in the provided descriptor for `fOnDiskId` and the given
   /// representation index. If there are no columns for the given representation index, return an empty
   /// ColumnRepresentation_t list. Otherwise, the returned reference points into the static array returned by
   /// GetColumnRepresentations().
   /// Throws an exception if the types on disk don't match any of the deserialization types from
   /// GetColumnRepresentations().
   const ColumnRepresentation_t &
   EnsureCompatibleColumnTypes(const ROOT::RNTupleDescriptor &desc, std::uint16_t representationIndex) const;
   /// When connecting a field to a page sink, the field's default column representation is subject
   /// to adjustment according to the write options. E.g., if compression is turned off, encoded columns
   /// are changed to their unencoded counterparts.
   void AutoAdjustColumnTypes(const ROOT::RNTupleWriteOptions &options);

   /// Called by Clone(), which additionally copies the on-disk ID
   virtual std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const = 0;

   /// Constructs value in a given location of size at least GetValueSize(). Called by the base class' CreateValue().
   virtual void ConstructValue(void *where) const = 0;
   virtual std::unique_ptr<RDeleter> GetDeleter() const { return std::make_unique<RDeleter>(); }
   /// Allow derived classes to call ConstructValue(void *) and GetDeleter() on other (sub)fields.
   static void CallConstructValueOn(const RFieldBase &other, void *where) { other.ConstructValue(where); }
   static std::unique_ptr<RDeleter> GetDeleterOf(const RFieldBase &other) { return other.GetDeleter(); }

   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual std::size_t AppendImpl(const void *from);
   virtual void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to);
   virtual void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to);

   /// Write the given value into columns. The value object has to be of the same type as the field.
   /// Returns the number of uncompressed bytes written.
   std::size_t Append(const void *from);

   /// Populate a single value with data from the field. The memory location pointed to by to needs to be of the
   /// fitting type. The fast path is conditioned by the field qualifying as simple, i.e. maps as-is
   /// to a single column and has no read callback.
   void Read(ROOT::NTupleSize_t globalIndex, void *to)
   {
      if (fIsSimple)
         return (void)fPrincipalColumn->Read(globalIndex, to);

      if (!fIsArtificial) {
         if (fTraits & kTraitMappable)
            fPrincipalColumn->Read(globalIndex, to);
         else
            ReadGlobalImpl(globalIndex, to);
      }
      if (R__unlikely(!fReadCallbacks.empty()))
         InvokeReadCallbacks(to);
   }

   /// Populate a single value with data from the field. The memory location pointed to by to needs to be of the
   /// fitting type. The fast path is conditioned by the field qualifying as simple, i.e. maps as-is
   /// to a single column and has no read callback.
   void Read(RNTupleLocalIndex localIndex, void *to)
   {
      if (fIsSimple)
         return (void)fPrincipalColumn->Read(localIndex, to);

      if (!fIsArtificial) {
         if (fTraits & kTraitMappable)
            fPrincipalColumn->Read(localIndex, to);
         else
            ReadInClusterImpl(localIndex, to);
      }
      if (R__unlikely(!fReadCallbacks.empty()))
         InvokeReadCallbacks(to);
   }

   /// General implementation of bulk read. Loop over the required range and read values that are required
   /// and not already present. Derived classes may implement more optimized versions of this method.
   /// See ReadBulk() for the return value.
   virtual std::size_t ReadBulkImpl(const RBulkSpec &bulkSpec);

   /// Returns the number of newly available values, that is the number of bools in `bulkSpec.fMaskAvail` that
   /// flipped from false to true. As a special return value, `kAllSet` can be used if all values are read
   /// independent from the masks.
   std::size_t ReadBulk(const RBulkSpec &bulkSpec);

   /// Allow derived classes to call Append() and Read() on other (sub)fields.
   static std::size_t CallAppendOn(RFieldBase &other, const void *from) { return other.Append(from); }
   static void CallReadOn(RFieldBase &other, RNTupleLocalIndex localIndex, void *to) { other.Read(localIndex, to); }
   static void CallReadOn(RFieldBase &other, ROOT::NTupleSize_t globalIndex, void *to) { other.Read(globalIndex, to); }
   static void *CallCreateObjectRawPtrOn(RFieldBase &other) { return other.CreateObjectRawPtr(); }

   /// Fields may need direct access to the principal column of their subfields, e.g. in RRVecField::ReadBulk()
   static ROOT::Internal::RColumn *GetPrincipalColumnOf(const RFieldBase &other) { return other.fPrincipalColumn; }

   /// Set a user-defined function to be called after reading a value, giving a chance to inspect and/or modify the
   /// value object.
   /// Returns an index that can be used to remove the callback.
   size_t AddReadCallback(ReadCallback_t func);
   void RemoveReadCallback(size_t idx);

   // Perform housekeeping tasks for global to cluster-local index translation
   virtual void CommitClusterImpl() {}
   // The field can indicate that it needs to register extra type information in the on-disk schema.
   // In this case, a callback from the page sink to the field will be registered on connect, so that the
   // extra type information can be collected when the dataset gets committed.
   virtual bool HasExtraTypeInfo() const { return false; }
   // The page sink's callback when the data set gets committed will call this method to get the field's extra
   // type information. This has to happen at the end of writing because the type information may change depending
   // on the data that's written, e.g. for polymorphic types in the streamer field.
   virtual ROOT::RExtraTypeInfoDescriptor GetExtraTypeInfo() const { return ROOT::RExtraTypeInfoDescriptor(); }

   /// Add a new subfield to the list of nested fields
   void Attach(std::unique_ptr<RFieldBase> child);

   /// Called by ConnectPageSource() before connecting; derived classes may override this as appropriate
   virtual void BeforeConnectPageSource(ROOT::Internal::RPageSource &) {}

   /// Called by ConnectPageSource() once connected; derived classes may override this as appropriate
   virtual void AfterConnectPageSource() {}

   /// Factory method to resurrect a field from the stored on-disk type information.  This overload takes an already
   /// normalized type name and type alias.
   /// `desc` and `fieldId` must be passed if `options.fEmulateUnknownTypes` is true, otherwise they can be left blank.
   static RResult<std::unique_ptr<RFieldBase>>
   Create(const std::string &fieldName, const std::string &typeName, const ROOT::RCreateFieldOptions &options,
          const ROOT::RNTupleDescriptor *desc, ROOT::DescriptorId_t fieldId);

public:
   template <bool IsConstT>
   class RSchemaIteratorTemplate;
   using RSchemaIterator = RSchemaIteratorTemplate<false>;
   using RConstSchemaIterator = RSchemaIteratorTemplate<true>;

   // This is used in CreateObject() and is specialized for void
   template <typename T>
   struct RCreateObjectDeleter {
      using deleter = std::default_delete<T>;
   };

   /// Used in the return value of the Check() method
   struct RCheckResult {
      std::string fFieldName; ///< Qualified field name causing the error
      std::string fTypeName;  ///< Type name corresponding to the (sub)field
      std::string fErrMsg;    ///< Cause of the failure, e.g. unsupported type
   };

   /// The constructor creates the underlying column objects and connects them to either a sink or a source.
   /// If `isSimple` is `true`, the trait `kTraitMappable` is automatically set on construction. However, the
   /// field might be demoted to non-simple if a post-read callback is set.
   RFieldBase(std::string_view name, std::string_view type, ROOT::ENTupleStructure structure, bool isSimple,
              std::size_t nRepetitions = 0);
   RFieldBase(const RFieldBase &) = delete;
   RFieldBase(RFieldBase &&) = default;
   RFieldBase &operator=(const RFieldBase &) = delete;
   RFieldBase &operator=(RFieldBase &&) = default;
   virtual ~RFieldBase() = default;

   /// Copies the field and its subfields using a possibly new name and a new, unconnected set of columns
   std::unique_ptr<RFieldBase> Clone(std::string_view newName) const;

   /// Factory method to create a field from a certain type given as string.
   /// Note that the provided type name must be a valid C++ type name. Template arguments of templated types
   /// must be type names or integers (e.g., no expressions).
   static RResult<std::unique_ptr<RFieldBase>>
   Create(const std::string &fieldName, const std::string &typeName);

   /// Checks if the given type is supported by RNTuple. In case of success, the result vector is empty.
   /// Otherwise there is an error record for each failing subfield (subtype).
   static std::vector<RCheckResult> Check(const std::string &fieldName, const std::string &typeName);

   /// Generates an object of the field type and allocates new initialized memory according to the type.
   /// Implemented at the end of this header because the implementation is using RField<T>::TypeName()
   /// The returned object can be released with `delete`, i.e. it is valid to call:
   /// ~~~{.cpp}
   ///    auto ptr = field->CreateObject();
   ///    delete ptr.release();
   /// ~~~
   ///
   /// Note that CreateObject<void>() is supported. The returned `unique_ptr` has a custom deleter that reports an error
   /// if it is called. The intended use of the returned `unique_ptr<void>` is to call `release()`. In this way, the
   /// transfer of pointer ownership is explicit.
   template <typename T>
   std::unique_ptr<T, typename RCreateObjectDeleter<T>::deleter> CreateObject() const;
   /// Generates an object of the field's type, wraps it in a shared pointer and returns it as an RValue connected to
   /// the field.
   RValue CreateValue();
   /// Creates a new, initially empty bulk.
   /// RBulkValues::ReadBulk() will construct the array of values. The memory of the value array is managed by the
   /// RBulkValues class.
   RBulkValues CreateBulk();
   /// Creates a value from a memory location with an already constructed object
   RValue BindValue(std::shared_ptr<void> objPtr);
   /// Creates the list of direct child values given an existing value for this field. E.g. a single value for the
   /// correct `std::variant` or all the elements of a collection. The default implementation assumes no subvalues
   /// and returns an empty vector.
   virtual std::vector<RValue> SplitValue(const RValue &value) const;
   /// The number of bytes taken by a value of the appropriate type
   virtual size_t GetValueSize() const = 0;
   /// As a rule of thumb, the alignment is equal to the size of the type. There are, however, various exceptions
   /// to this rule depending on OS and CPU architecture. So enforce the alignment to be explicitly spelled out.
   virtual size_t GetAlignment() const = 0;
   std::uint32_t GetTraits() const { return fTraits; }
   bool HasReadCallbacks() const { return !fReadCallbacks.empty(); }

   const std::string &GetFieldName() const { return fName; }
   /// Returns the field name and parent field names separated by dots (`grandparent.parent.child`)
   std::string GetQualifiedFieldName() const;
   const std::string &GetTypeName() const { return fType; }
   const std::string &GetTypeAlias() const { return fTypeAlias; }
   ROOT::ENTupleStructure GetStructure() const { return fStructure; }
   std::size_t GetNRepetitions() const { return fNRepetitions; }
   const RFieldBase *GetParent() const { return fParent; }
   std::vector<RFieldBase *> GetMutableSubfields();
   std::vector<const RFieldBase *> GetConstSubfields() const;
   bool IsSimple() const { return fIsSimple; }
   bool IsArtificial() const { return fIsArtificial; }
   /// Get the field's description
   const std::string &GetDescription() const { return fDescription; }
   void SetDescription(std::string_view description);
   EState GetState() const { return fState; }

   ROOT::DescriptorId_t GetOnDiskId() const { return fOnDiskId; }
   void SetOnDiskId(ROOT::DescriptorId_t id);

   /// Returns the `fColumnRepresentative` pointee or, if unset (always the case for artificial fields), the field's
   /// default representative
   RColumnRepresentations::Selection_t GetColumnRepresentatives() const;
   /// Fixes a column representative. This can only be done _before_ connecting the field to a page sink.
   /// Otherwise, or if the provided representation is not in the list of GetColumnRepresentations(),
   /// an exception is thrown
   void SetColumnRepresentatives(const RColumnRepresentations::Selection_t &representatives);
   /// Whether or not an explicit column representative was set
   bool HasDefaultColumnRepresentative() const { return fColumnRepresentatives.empty(); }

   /// Indicates an evolution of the mapping scheme from C++ type to columns
   virtual std::uint32_t GetFieldVersion() const { return 0; }
   /// Indicates an evolution of the C++ type itself
   virtual std::uint32_t GetTypeVersion() const { return 0; }
   /// Return the current TClass reported checksum of this class. Only valid if `kTraitTypeChecksum` is set.
   virtual std::uint32_t GetTypeChecksum() const { return 0; }
   /// Return the C++ type version stored in the field descriptor; only valid after a call to ConnectPageSource()
   std::uint32_t GetOnDiskTypeVersion() const { return fOnDiskTypeVersion; }
   /// Return checksum stored in the field descriptor; only valid after a call to ConnectPageSource(),
   /// if the field stored a type checksum
   std::uint32_t GetOnDiskTypeChecksum() const { return fOnDiskTypeChecksum; }

   RSchemaIterator begin();
   RSchemaIterator end();
   RConstSchemaIterator begin() const;
   RConstSchemaIterator end() const;
   RConstSchemaIterator cbegin() const;
   RConstSchemaIterator cend() const;

   virtual void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const;
}; // class RFieldBase

/// Iterates over the subtree of fields in depth-first search order
template <bool IsConstT>
class RFieldBase::RSchemaIteratorTemplate {
private:
   struct Position {
      using FieldPtr_t = std::conditional_t<IsConstT, const RFieldBase *, RFieldBase *>;
      Position() : fFieldPtr(nullptr), fIdxInParent(-1) {}
      Position(FieldPtr_t fieldPtr, int idxInParent) : fFieldPtr(fieldPtr), fIdxInParent(idxInParent) {}
      FieldPtr_t fFieldPtr;
      int fIdxInParent;
   };
   /// The stack of nodes visited when walking down the tree of fields
   std::vector<Position> fStack;

public:
   using iterator = RSchemaIteratorTemplate<IsConstT>;
   using iterator_category = std::forward_iterator_tag;
   using difference_type = std::ptrdiff_t;
   using value_type = std::conditional_t<IsConstT, const RFieldBase, RFieldBase>;
   using pointer = std::conditional_t<IsConstT, const RFieldBase *, RFieldBase *>;
   using reference = std::conditional_t<IsConstT, const RFieldBase &, RFieldBase &>;

   RSchemaIteratorTemplate() { fStack.emplace_back(Position()); }
   RSchemaIteratorTemplate(pointer val, int idxInParent) { fStack.emplace_back(Position(val, idxInParent)); }
   ~RSchemaIteratorTemplate() {}
   /// Given that the iterator points to a valid field which is not the end iterator, go to the next field
   /// in depth-first search order
   void Advance()
   {
      auto itr = fStack.rbegin();
      if (!itr->fFieldPtr->fSubfields.empty()) {
         fStack.emplace_back(Position(itr->fFieldPtr->fSubfields[0].get(), 0));
         return;
      }

      unsigned int nextIdxInParent = ++(itr->fIdxInParent);
      while (nextIdxInParent >= itr->fFieldPtr->fParent->fSubfields.size()) {
         if (fStack.size() == 1) {
            itr->fFieldPtr = itr->fFieldPtr->fParent;
            itr->fIdxInParent = -1;
            return;
         }
         fStack.pop_back();
         itr = fStack.rbegin();
         nextIdxInParent = ++(itr->fIdxInParent);
      }
      itr->fFieldPtr = itr->fFieldPtr->fParent->fSubfields[nextIdxInParent].get();
   }

   iterator operator++(int) /* postfix */
   {
      auto r = *this;
      Advance();
      return r;
   }
   iterator &operator++() /* prefix */
   {
      Advance();
      return *this;
   }
   reference operator*() const { return *fStack.back().fFieldPtr; }
   pointer operator->() const { return fStack.back().fFieldPtr; }
   bool operator==(const iterator &rh) const { return fStack.back().fFieldPtr == rh.fStack.back().fFieldPtr; }
   bool operator!=(const iterator &rh) const { return fStack.back().fFieldPtr != rh.fStack.back().fFieldPtr; }
};

/// Points to an object with RNTuple I/O support and keeps a pointer to the corresponding field.
/// Fields can create RValue objects through RFieldBase::CreateValue(), RFieldBase::BindValue()) or
/// RFieldBase::SplitValue().
class RFieldBase::RValue final {
   friend class RFieldBase;
   friend class ROOT::REntry;

private:
   RFieldBase *fField = nullptr;  ///< The field that created the RValue
   /// Set by Bind() or by RFieldBase::CreateValue(), RFieldBase::SplitValue() or RFieldBase::BindValue()
   std::shared_ptr<void> fObjPtr;
   mutable std::atomic<const std::type_info *> fTypeInfo = nullptr;

   RValue(RFieldBase *field, std::shared_ptr<void> objPtr) : fField(field), fObjPtr(objPtr) {}

public:
   RValue(const RValue &other) : fField(other.fField), fObjPtr(other.fObjPtr) {}
   RValue &operator=(const RValue &other)
   {
      fField = other.fField;
      fObjPtr = other.fObjPtr;
      // We could copy over the cached type info, or just start with a fresh state...
      fTypeInfo = nullptr;
      return *this;
   }
   RValue(RValue &&other) : fField(other.fField), fObjPtr(other.fObjPtr) {}
   RValue &operator=(RValue &&other)
   {
      fField = other.fField;
      fObjPtr = other.fObjPtr;
      // We could copy over the cached type info, or just start with a fresh state...
      fTypeInfo = nullptr;
      return *this;
   }
   ~RValue() = default;

private:
   template <typename T>
   void EnsureMatchingType() const
   {
      if constexpr (!std::is_void_v<T>) {
         const std::type_info &ti = typeid(T);
         // Fast path: if we had a matching type before, try comparing the type_info's. This may still fail in case the
         // type has a suppressed template argument that may change the typeid.
         auto *cachedTypeInfo = fTypeInfo.load();
         if (cachedTypeInfo != nullptr && *cachedTypeInfo == ti) {
            return;
         }
         std::string renormalizedTypeName = Internal::GetRenormalizedTypeName(ti);
         if (Internal::IsMatchingFieldType(fField->GetTypeName(), renormalizedTypeName, ti)) {
            fTypeInfo.store(&ti);
            return;
         }
         throw RException(R__FAIL("type mismatch for field \"" + fField->GetFieldName() + "\": expected " +
                                  fField->GetTypeName() + ", got " + renormalizedTypeName));
      }
   }

   std::size_t Append() { return fField->Append(fObjPtr.get()); }

public:
   void Read(ROOT::NTupleSize_t globalIndex) { fField->Read(globalIndex, fObjPtr.get()); }
   void Read(RNTupleLocalIndex localIndex) { fField->Read(localIndex, fObjPtr.get()); }

   void Bind(std::shared_ptr<void> objPtr) { fObjPtr = objPtr; }
   void BindRawPtr(void *rawPtr);
   /// Replace the current object pointer by a pointer to a new object constructed by the field
   void EmplaceNew() { fObjPtr = fField->CreateValue().GetPtr<void>(); }

   template <typename T>
   std::shared_ptr<T> GetPtr() const
   {
      EnsureMatchingType<T>();
      return std::static_pointer_cast<T>(fObjPtr);
   }

   template <typename T>
   const T &GetRef() const
   {
      EnsureMatchingType<T>();
      return *static_cast<T *>(fObjPtr.get());
   }

   const RFieldBase &GetField() const { return *fField; }
};

/// Input parameter to RFieldBase::ReadBulk() and RFieldBase::ReadBulkImpl().
//  See the RBulkValues class documentation for more information.
struct RFieldBase::RBulkSpec {
   /// Possible return value of ReadBulk() and ReadBulkImpl(), which indicates that the full bulk range was read
   /// independently of the provided masks.
   static const std::size_t kAllSet = std::size_t(-1);

   RNTupleLocalIndex fFirstIndex; ///< Start of the bulk range
   std::size_t fCount = 0;        ///< Size of the bulk range
   /// A bool array of size fCount, indicating the required values in the requested range
   const bool *fMaskReq = nullptr;
   bool *fMaskAvail = nullptr; ///< A bool array of size `fCount`, indicating the valid values in fValues
   /// The destination area, which has to be an array of valid objects of the correct type large enough to hold the bulk
   /// range.
   void *fValues = nullptr;
   /// Reference to memory owned by the RBulkValues class. The field implementing BulkReadImpl() may use `fAuxData` as
   /// memory that stays persistent between calls.
   std::vector<unsigned char> *fAuxData = nullptr;
};

// clang-format off
/**
\class ROOT::RFieldBase::RBulkValues
\ingroup NTuple
\brief Points to an array of objects with RNTuple I/O support, used for bulk reading.

Similar to RValue, but manages an array of consecutive values. Bulks have to come from the same cluster.
Bulk I/O works with two bit masks: the mask of all the available entries in the current bulk and the mask
of the required entries in a bulk read. The idea is that a single bulk may serve multiple read operations
on the same range, where in each read operation a different subset of values is required.
The memory of the value array is managed by the RBulkValues class.
*/
// clang-format on
class RFieldBase::RBulkValues final {
private:
   friend class RFieldBase;

   RFieldBase *fField = nullptr;                   ///< The field that created the array of values
   std::unique_ptr<RFieldBase::RDeleter> fDeleter; /// Cached deleter of fField
   void *fValues = nullptr;                        ///< Pointer to the start of the array
   std::size_t fValueSize = 0;                     ///< Cached copy of RFieldBase::GetValueSize()
   std::size_t fCapacity = 0;                      ///< The size of the array memory block in number of values
   std::size_t fSize = 0;              ///< The number of available values in the array (provided their mask is set)
   bool fIsAdopted = false;            ///< True if the user provides the memory buffer for fValues
   std::unique_ptr<bool[]> fMaskAvail; ///< Masks invalid values in the array
   std::size_t fNValidValues = 0;      ///< The sum of non-zero elements in the fMask
   RNTupleLocalIndex fFirstIndex;      ///< Index of the first value of the array
   /// Reading arrays of complex values may require additional memory, for instance for the elements of
   /// arrays of vectors. A pointer to the `fAuxData` array is passed to the field's BulkRead method.
   /// The RBulkValues class does not modify the array in-between calls to the field's BulkRead method.
   std::vector<unsigned char> fAuxData;

   void ReleaseValues();
   /// Sets a new range for the bulk. If there is enough capacity, the `fValues` array will be reused.
   /// Otherwise a new array is allocated. After reset, fMaskAvail is false for all values.
   void Reset(RNTupleLocalIndex firstIndex, std::size_t size);
   void CountValidValues();

   bool ContainsRange(RNTupleLocalIndex firstIndex, std::size_t size) const
   {
      if (firstIndex.GetClusterId() != fFirstIndex.GetClusterId())
         return false;
      return (firstIndex.GetIndexInCluster() >= fFirstIndex.GetIndexInCluster()) &&
             ((firstIndex.GetIndexInCluster() + size) <= (fFirstIndex.GetIndexInCluster() + fSize));
   }

   void *GetValuePtrAt(std::size_t idx) const { return reinterpret_cast<unsigned char *>(fValues) + idx * fValueSize; }

   explicit RBulkValues(RFieldBase *field)
      : fField(field), fDeleter(field->GetDeleter()), fValueSize(field->GetValueSize())
   {
   }

public:
   ~RBulkValues();
   RBulkValues(const RBulkValues &) = delete;
   RBulkValues &operator=(const RBulkValues &) = delete;
   RBulkValues(RBulkValues &&other);
   RBulkValues &operator=(RBulkValues &&other);

   // Sets `fValues` and `fSize`/`fCapacity` to the given values. The capacity is specified in number of values.
   // Once a buffer is adopted, an attempt to read more values then available throws an exception.
   void AdoptBuffer(void *buf, std::size_t capacity);

   /// Reads `size` values from the associated field, starting from `firstIndex`. Note that the index is given
   /// relative to a certain cluster. The return value points to the array of read objects.
   /// The `maskReq` parameter is a bool array of at least `size` elements. Only objects for which the mask is
   /// true are guaranteed to be read in the returned value array. A `nullptr` means to read all elements.
   void *ReadBulk(RNTupleLocalIndex firstIndex, const bool *maskReq, std::size_t size)
   {
      if (!ContainsRange(firstIndex, size))
         Reset(firstIndex, size);

      // We may read a subrange of the currently available range
      auto offset = firstIndex.GetIndexInCluster() - fFirstIndex.GetIndexInCluster();

      if (fNValidValues == fSize)
         return GetValuePtrAt(offset);

      RBulkSpec bulkSpec;
      bulkSpec.fFirstIndex = firstIndex;
      bulkSpec.fCount = size;
      bulkSpec.fMaskReq = maskReq;
      bulkSpec.fMaskAvail = &fMaskAvail[offset];
      bulkSpec.fValues = GetValuePtrAt(offset);
      bulkSpec.fAuxData = &fAuxData;
      auto nRead = fField->ReadBulk(bulkSpec);
      if (nRead == RBulkSpec::kAllSet) {
         if ((offset == 0) && (size == fSize)) {
            fNValidValues = fSize;
         } else {
            CountValidValues();
         }
      } else {
         fNValidValues += nRead;
      }
      return GetValuePtrAt(offset);
   }

   /// Overload to read all elements in the given cluster range.
   void *ReadBulk(ROOT::RNTupleLocalRange range) { return ReadBulk(*range.begin(), nullptr, range.size()); }
};

namespace Internal {
// At some point, RFieldBase::OnClusterCommit() may allow for a user-defined callback to change the
// column representation. For now, we inject this for testing and internal use only.
struct RFieldRepresentationModifier {
   static void SetPrimaryColumnRepresentation(RFieldBase &field, std::uint16_t newRepresentationIdx)
   {
      R__ASSERT(newRepresentationIdx < field.fColumnRepresentatives.size());
      const auto N = field.fColumnRepresentatives[0].get().size();
      R__ASSERT(N >= 1 && N <= 2);
      R__ASSERT(field.fPrincipalColumn);
      field.fPrincipalColumn = field.fAvailableColumns[newRepresentationIdx * N].get();
      if (field.fAuxiliaryColumn) {
         R__ASSERT(N == 2);
         field.fAuxiliaryColumn = field.fAvailableColumns[newRepresentationIdx * N + 1].get();
      }
   }
};
} // namespace Internal
} // namespace ROOT

#endif
