/// \file ROOT/RField.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RField
#define ROOT7_RField

#include <ROOT/RColumn.hxx>
#include <ROOT/RColumnElement.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/TypeTraits.hxx>

#include <TGenericClassInfo.h>
#include <TError.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <utility>

class TClass;

namespace ROOT {
namespace Experimental {

class RCollectionNTuple;
class REntry;
class RNTupleModel;
class RFieldCollection;

namespace Detail {

class RPageStorage;

// clang-format off
/**
\class ROOT::Experimental::RFieldBase
\ingroup NTuple
\brief A field translates read and write calls from/to underlying columns to/from tree values

A field is a serializable C++ type or a container for a collection of sub fields. The RFieldBase and its
type-safe descendants provide the object to column mapper. They map C++ objects to primitive columns.  The
mapping is trivial for simple types such as 'double'. Complex types resolve to multiple primitive columns.
The field knows based on its type and the field name the type(s) and name(s) of the columns.
*/
// clang-format on
class RFieldBase {
   friend class ROOT::Experimental::RFieldCollection; // to change the field names when collections are attached
private:
   /// The field name is a unique within a tree and also the basis for the column name(s)
   std::string fName;
   /// The C++ type captured by this field
   std::string fType;
   /// The role of this field in the data model structure
   ENTupleStructure fStructure;
   /// A field on a trivial type that maps as-is to a single column
   bool fIsSimple;

protected:
   /// Collections and classes own sub fields
   std::vector<std::unique_ptr<RFieldBase>> fSubFields;
   /// Sub fields point to their mother field
   RFieldBase* fParent;
   /// All fields have a main column. For collection fields, the main column is the index field. Points into fColumns.
   RColumn* fPrincipalColumn;
   /// The columns are connected either to a sink or to a source (not to both); they are owned by the field.
   std::vector<std::unique_ptr<RColumn>> fColumns;

   /// Creates the backing columns corresponsing to the field type and name
   virtual void DoGenerateColumns() = 0;

   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual void DoAppend(const RFieldValue& value);
   virtual void DoRead(NTupleSize_t index, RFieldValue* value);
   virtual void DoReadV(NTupleSize_t index, NTupleSize_t count, void* dst);

public:
   /// Field names convey the level of subfields; sub fields (nested collections) are separated by a dot
   static constexpr char kCollectionSeparator = '/';

   /// Iterates over the sub fields in depth-first search order
   class RIterator : public std::iterator<std::forward_iterator_tag, Detail::RFieldBase> {
   private:
      using iterator = RIterator;
      struct Position {
         Position() : fFieldPtr(nullptr), fIdxInParent(-1) { }
         Position(pointer fieldPtr, int idxInParent) : fFieldPtr(fieldPtr), fIdxInParent(idxInParent) { }
         pointer fFieldPtr;
         int fIdxInParent;
      };
      /// The stack of nodes visited when walking down the tree of fields
      std::vector<Position> fStack;
   public:
      RIterator() { fStack.emplace_back(Position()); }
      RIterator(pointer val, int idxInParent) { fStack.emplace_back(Position(val, idxInParent)); }
      ~RIterator() {}
      /// Given that the iterator points to a valid field which is not the end iterator, go to the next field
      /// in depth-first search order
      void Advance();

      iterator  operator++(int) /* postfix */        { auto r = *this; Advance(); return r; }
      iterator& operator++()    /* prefix */         { Advance(); return *this; }
      reference operator* () const                   { return *fStack.back().fFieldPtr; }
      pointer   operator->() const                   { return fStack.back().fFieldPtr; }
      bool      operator==(const iterator& rh) const { return fStack.back().fFieldPtr == rh.fStack.back().fFieldPtr; }
      bool      operator!=(const iterator& rh) const { return fStack.back().fFieldPtr != rh.fStack.back().fFieldPtr; }
   };

   /// The constructor creates the underlying column objects and connects them to either a sink or a source.
   RFieldBase(std::string_view name, std::string_view type, ENTupleStructure structure, bool isSimple);
   RFieldBase(const RFieldBase&) = delete;
   RFieldBase(RFieldBase&&) = default;
   RFieldBase& operator =(const RFieldBase&) = delete;
   RFieldBase& operator =(RFieldBase&&) = default;
   virtual ~RFieldBase();

   ///// Copies the field and its sub fields using a possibly new name and a new, unconnected set of columns
   virtual RFieldBase* Clone(std::string_view newName) = 0;

   /// Factory method to resurrect a field from the stored on-disk type information
   static RFieldBase *Create(const std::string &fieldName, const std::string &typeName);
   /// Get the tail of the field name up to the last dot
   static std::string GetLeafName(const std::string &fullName);
   /// Get the name for an item sub field that is part of a collection, e.g. the float field of std::vector<float>
   static std::string GetCollectionName(const std::string &parentName);

   /// Registeres (or re-registers) the backing columns with the physical storage
   void ConnectColumns(Detail::RPageStorage *pageStorage);
   /// Returns the number of columns generated to store data for the field; defaults to 1
   virtual unsigned int GetNColumns() const = 0;

   /// Generates a tree value of the field type and allocates new initialized memory according to the type.
   RFieldValue GenerateValue();
   /// Generates a tree value in a given location of size at least GetValueSize(). Assumes that where has been
   /// allocated by malloc().
   virtual RFieldValue GenerateValue(void *where) = 0;
   /// Releases the resources acquired during GenerateValue (memory and constructor)
   /// This implementation works for simple types but needs to be overwritten for complex ones
   virtual void DestroyValue(const RFieldValue &value, bool dtorOnly = false);
   /// Creates a value from a memory location with an already constructed object
   virtual RFieldValue CaptureValue(void *where) = 0;
   /// The number of bytes taken by a value of the appropriate type
   virtual size_t GetValueSize() const = 0;

   /// Write the given value into columns. The value object has to be of the same type as the field.
   void Append(const RFieldValue& value) {
      if (!fIsSimple) {
         DoAppend(value);
         return;
      }
      //printf("Appending simple value for %lu %s\n", *(unsigned long *)(value.GetRawPtr()), fName.c_str());
      fPrincipalColumn->Append(value.fMappedElement);
   }

   /// Populate a single value with data from the tree, which needs to be of the fitting type.
   /// Reading copies data into the memory wrapped by the tree value.
   void Read(NTupleSize_t index, RFieldValue* value) {
      if (!fIsSimple) {
         DoRead(index, value);
         return;
      }
      fPrincipalColumn->Read(index, &value->fMappedElement);
   }

   /// Type unsafe bulk read interface; dst must point to a vector of objects of the field type.
   /// TODO(jblomer): can this be type safe?
   void ReadV(NTupleSize_t index, NTupleSize_t count, void *dst)
   {
      if (!fIsSimple) {
         DoReadV(index, count, dst);
         return;
      }
      //fPrincipalColumn->ReadV(index, count, dst);
   }

   /// The number of elements in the principal column. For top level fields, the number of entries.
   NTupleSize_t GetNItems();

   /// Ensure that all received items are written from page buffers to the storage.
   void Flush() const;
   /// Perform housekeeping tasks for global to cluster-local index translation
   virtual void CommitCluster() {}

   void Attach(std::unique_ptr<Detail::RFieldBase> child);

   std::string GetName() const { return fName; }
   std::string GetType() const { return fType; }
   ENTupleStructure GetStructure() const { return fStructure; }
   const RFieldBase* GetParent() const { return fParent; }
   bool IsSimple() const { return fIsSimple; }

   /// Indicates an evolution of the mapping scheme from C++ type to columns
   virtual RNTupleVersion GetFieldVersion() const { return RNTupleVersion(); }
   /// Indicates an evolution of the C++ type itself
   virtual RNTupleVersion GetTypeVersion() const { return RNTupleVersion(); }

   RIterator begin();
   RIterator end();
};

} // namespace Detail

/// The container field for a tree model, which itself has no physical representation
class RFieldRoot : public Detail::RFieldBase {
public:
   RFieldRoot() : Detail::RFieldBase("", "", ENTupleStructure::kRecord, false /* isSimple */) {}
   RFieldBase* Clone(std::string_view newName);

   void DoGenerateColumns() final {}
   unsigned int GetNColumns() const final { return 0; }
   using Detail::RFieldBase::GenerateValue;
   Detail::RFieldValue GenerateValue(void*) { return Detail::RFieldValue(); }
   Detail::RFieldValue CaptureValue(void*) final { return Detail::RFieldValue(); }
   size_t GetValueSize() const final { return 0; }

   /// Generates managed values for the top-level sub fields
   REntry* GenerateEntry();
};

/// The field for a class with dictionary
class RFieldClass : public Detail::RFieldBase {
private:
   TClass* fClass;
protected:
   void DoAppend(const Detail::RFieldValue& value) final;
   void DoRead(NTupleSize_t index, Detail::RFieldValue* value) final;
public:
   RFieldClass(std::string_view fieldName, std::string_view className);
   RFieldClass(RFieldClass&& other) = default;
   RFieldClass& operator =(RFieldClass&& other) = default;
   ~RFieldClass() = default;
   RFieldBase* Clone(std::string_view newName) final;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final;
   using Detail::RFieldBase::GenerateValue;
   Detail::RFieldValue GenerateValue(void* where) override;
   void DestroyValue(const Detail::RFieldValue& value, bool dtorOnly = false) final;
   Detail::RFieldValue CaptureValue(void *where) final;
   size_t GetValueSize() const override;
};

/// The generic field for a (nested) std::vector<Type>
class RFieldVector : public Detail::RFieldBase {
private:
   size_t fItemSize;
   ClusterSize_t fNWritten;

protected:
   void DoAppend(const Detail::RFieldValue& value) final;
   void DoRead(NTupleSize_t index, Detail::RFieldValue* value) final;

public:
   RFieldVector(std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField);
   RFieldVector(RFieldVector&& other) = default;
   RFieldVector& operator =(RFieldVector&& other) = default;
   ~RFieldVector() = default;
   RFieldBase* Clone(std::string_view newName) final;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final;
   using Detail::RFieldBase::GenerateValue;
   Detail::RFieldValue GenerateValue(void* where) override;
   void DestroyValue(const Detail::RFieldValue& value, bool dtorOnly = false) final;
   Detail::RFieldValue CaptureValue(void *where) override;
   size_t GetValueSize() const override;
   void CommitCluster() final;
};


/// Classes with dictionaries that can be inspected by TClass
template <typename T, typename=void>
class RField : public RFieldClass {
public:
   static std::string MyTypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : RFieldClass(name, MyTypeName()) {
      static_assert(std::is_class<T>::value, "no I/O support for this basic C++ type");
   }
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(this, static_cast<T*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, T()); }
};


class RFieldCollection : public ROOT::Experimental::Detail::RFieldBase {
private:
   /// Save the link to the collection ntuple in order to reset the offset counter when committing the cluster
   std::shared_ptr<RCollectionNTuple> fCollectionNTuple;
public:
   static std::string MyTypeName() { return ":RFieldCollection:"; }
   RFieldCollection(std::string_view name,
                    std::shared_ptr<RCollectionNTuple> collectionNTuple,
                    std::unique_ptr<RNTupleModel> collectionModel);
   RFieldCollection(RFieldCollection&& other) = default;
   RFieldCollection& operator =(RFieldCollection&& other) = default;
   ~RFieldCollection() = default;
   RFieldBase* Clone(std::string_view newName) final;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   using Detail::RFieldBase::GenerateValue;
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final {
      return Detail::RFieldValue(
         Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex>(static_cast<ClusterSize_t*>(where)),
         this, static_cast<ClusterSize_t*>(where));
   }
   Detail::RFieldValue CaptureValue(void* where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex>(static_cast<ClusterSize_t*>(where)), this, where);
   }
   size_t GetValueSize() const final { return 0; }
   void CommitCluster() final;
};


/// Template specializations for concrete C++ types


template <>
class RField<ClusterSize_t> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "ROOT::Experimental::ClusterSize_t"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   ClusterSize_t* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex>::kIsMappable,
                    "(ClusterSize_t, EColumnType::kIndex) is not identical on this platform");
      return fPrincipalColumn->Map<ClusterSize_t, EColumnType::kIndex>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex>(static_cast<ClusterSize_t*>(where)),
         this, static_cast<ClusterSize_t*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex>(static_cast<ClusterSize_t*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(ClusterSize_t); }

   /// Special help for offset fields
   void GetCollectionInfo(NTupleSize_t index, NTupleSize_t* idxStart, ClusterSize_t* size) {
      fPrincipalColumn->GetCollectionInfo(index, idxStart, size);
   }
};


template <>
class RField<float> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "float"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   float* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<float, EColumnType::kReal32>::kIsMappable,
                    "(float, EColumnType::kReal32) is not identical on this platform");
      return fPrincipalColumn->Map<float, EColumnType::kReal32>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<float, EColumnType::kReal32>(static_cast<float*>(where)),
         this, static_cast<float*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0.0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<float, EColumnType::kReal32>(static_cast<float*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(float); }
};


template <>
class RField<double> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "double"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   double* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<double, EColumnType::kReal64>::kIsMappable,
                    "(double, EColumnType::kReal64) is not identical on this platform");
      return fPrincipalColumn->Map<double, EColumnType::kReal64>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<double, EColumnType::kReal64>(static_cast<double*>(where)),
         this, static_cast<double*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0.0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<double, EColumnType::kReal64>(static_cast<double*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(double); }
};

template <>
class RField<std::int32_t> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "std::int32_t"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   std::int32_t* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<std::int32_t, EColumnType::kInt32>::kIsMappable,
                    "(std::int32_t, EColumnType::kInt32) is not identical on this platform");
      return fPrincipalColumn->Map<std::int32_t, EColumnType::kInt32>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<std::int32_t, EColumnType::kInt32>(static_cast<std::int32_t*>(where)),
         this, static_cast<std::int32_t*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<std::int32_t, EColumnType::kInt32>(static_cast<std::int32_t*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(std::int32_t); }
};

template <>
class RField<std::uint32_t> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "std::uint32_t"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   std::uint32_t* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<std::uint32_t, EColumnType::kInt32>::kIsMappable,
                    "(std::uint32_t, EColumnType::kInt32) is not identical on this platform");
      return fPrincipalColumn->Map<std::uint32_t, EColumnType::kInt32>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<std::uint32_t, EColumnType::kInt32>(static_cast<std::uint32_t*>(where)),
         this, static_cast<std::uint32_t*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<std::uint32_t, EColumnType::kInt32>(static_cast<std::uint32_t*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(std::uint32_t); }
};

template <>
class RField<std::uint64_t> : public Detail::RFieldBase {
public:
   static std::string MyTypeName() { return "std::uint64_t"; }
   explicit RField(std::string_view name)
     : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, true /* isSimple */) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   std::uint64_t* Map(NTupleSize_t index) {
      static_assert(Detail::RColumnElement<std::uint64_t, EColumnType::kInt64>::kIsMappable,
                    "(std::uint64_t, EColumnType::kInt64) is not identical on this platform");
      return fPrincipalColumn->Map<std::uint64_t, EColumnType::kInt64>(index, nullptr);
   }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(
         Detail::RColumnElement<std::uint64_t, EColumnType::kInt64>(static_cast<std::uint64_t*>(where)),
         this, static_cast<std::uint64_t*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, 0); }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */,
         Detail::RColumnElement<std::uint64_t, EColumnType::kInt64>(static_cast<std::uint64_t*>(where)), this, where);
   }
   size_t GetValueSize() const final { return sizeof(std::uint64_t); }
};


template <>
class RField<std::string> : public Detail::RFieldBase {
private:
   ClusterSize_t fIndex;
   Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex> fElemIndex;

   void DoAppend(const ROOT::Experimental::Detail::RFieldValue& value) final;
   void DoRead(ROOT::Experimental::NTupleSize_t index, ROOT::Experimental::Detail::RFieldValue* value) final;

public:
   static std::string MyTypeName() { return "std::string"; }
   explicit RField(std::string_view name)
      : Detail::RFieldBase(name, MyTypeName(), ENTupleStructure::kLeaf, false /* isSimple */)
      , fIndex(0), fElemIndex(&fIndex) {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final { return new RField(newName); }

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 2; }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(this, static_cast<std::string*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final { return GenerateValue(where, ""); }
   void DestroyValue(const Detail::RFieldValue& value, bool dtorOnly = false) {
      auto str = value.Get<std::string>();
      str->~basic_string(); // TODO(jblomer) C++17 std::destroy_at
      if (!dtorOnly)
         free(str);
   }
   Detail::RFieldValue CaptureValue(void *where) {
      return Detail::RFieldValue(true /* captureFlag */, this, where);
   }
   size_t GetValueSize() const final { return sizeof(std::string); }
   void CommitCluster() final;
};


template <typename ItemT>
class RField<std::vector<ItemT>> : public RFieldVector {
   using ContainerT = typename std::vector<ItemT>;
public:
   static std::string MyTypeName() { return "std::vector<" + RField<ItemT>::MyTypeName() + ">"; }
   explicit RField(std::string_view name)
      : RFieldVector(name, std::make_unique<RField<ItemT>>(GetCollectionName(std::string(name))))
   {}
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(this, static_cast<ContainerT*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final {
      return GenerateValue(where, ContainerT());
   }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */, this, where);
   }
   size_t GetValueSize() const final { return sizeof(ContainerT); }
};


/**
 * The RVec type has different layouts depending on the item type, therefore we cannot go with a generic
 * RVec implementation as we can with std::vector
 */
template <typename ItemT>
class RField<ROOT::VecOps::RVec<ItemT>> : public Detail::RFieldBase {
   using ContainerT = typename ROOT::VecOps::RVec<ItemT>;
private:
   size_t fItemSize;
   ClusterSize_t fNWritten;

protected:
   void DoAppend(const Detail::RFieldValue& value) final {
      auto typedValue = value.Get<ContainerT>();
      auto count = typedValue->size();
      for (unsigned i = 0; i < count; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(&typedValue->data()[i]);
         fSubFields[0]->Append(itemValue);
      }
      Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex> elemIndex(&fNWritten);
      fNWritten += count;
      fColumns[0]->Append(elemIndex);
   }
   void DoRead(NTupleSize_t index, Detail::RFieldValue* value) final {
      auto typedValue = value->Get<ContainerT>();
      ClusterSize_t nItems;
      NTupleSize_t idxStart;
      fPrincipalColumn->GetCollectionInfo(index, &idxStart, &nItems);
      typedValue->resize(nItems);
      for (unsigned i = 0; i < nItems; ++i) {
         auto itemValue = fSubFields[0]->GenerateValue(&typedValue->data()[i]);
         fSubFields[0]->Read(idxStart + i, &itemValue);
      }
   }

public:
   RField(std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField)
      : ROOT::Experimental::Detail::RFieldBase(
           fieldName, "ROOT::VecOps::RVec<" + itemField->GetType() + ">", ENTupleStructure::kCollection, false)
      , fItemSize(itemField->GetValueSize()), fNWritten(0)
   {
      Attach(std::move(itemField));
   }
   explicit RField(std::string_view name)
      : RField(name, std::make_unique<RField<ItemT>>(GetCollectionName(std::string(name))))
   {
   }
   RField(RField&& other) = default;
   RField& operator =(RField&& other) = default;
   ~RField() = default;
   RFieldBase* Clone(std::string_view newName) final {
      auto newItemField = fSubFields[0]->Clone(GetCollectionName(std::string(newName)));
      return new RField<ROOT::VecOps::RVec<ItemT>>(newName, std::unique_ptr<Detail::RFieldBase>(newItemField));
   }

   void DoGenerateColumns() final {
      RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
      fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));
      fPrincipalColumn = fColumns[0].get();
   }
   unsigned int GetNColumns() const final { return 1; }
   void DestroyValue(const Detail::RFieldValue& value, bool dtorOnly = false) final {
      auto vec = reinterpret_cast<ContainerT*>(value.GetRawPtr());
      auto nItems = vec->size();
      for (unsigned i = 0; i < nItems; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(vec->data() + (i * fItemSize));
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }
      vec->~RVec();
      if (!dtorOnly)
         free(vec);
   }
   void CommitCluster() final { fNWritten = 0; }

   static std::string MyTypeName() { return "ROOT::VecOps::RVec<" + RField<ItemT>::MyTypeName() + ">"; }

   using Detail::RFieldBase::GenerateValue;
   template <typename... ArgsT>
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where, ArgsT&&... args)
   {
      return Detail::RFieldValue(this, static_cast<ContainerT*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final {
      return GenerateValue(where, ContainerT());
   }
   Detail::RFieldValue CaptureValue(void *where) final {
      return Detail::RFieldValue(true /* captureFlag */, this, static_cast<ContainerT*>(where));
   }
   size_t GetValueSize() const final { return sizeof(ContainerT); }
};

} // namespace Experimental
} // namespace ROOT

#endif
