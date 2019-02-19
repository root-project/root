/// \file ROOT/RTreeField.hxx
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

#ifndef ROOT7_RTreeField
#define ROOT7_RTreeField

#include <ROOT/RColumn.hxx>
#include <ROOT/RColumnElement.hxx>
#include <ROOT/RForestUtil.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeValue.hxx>
#include <ROOT/TypeTraits.hxx>

#include <TGenericClassInfo.h>
#include <TError.h>

#include <algorithm>
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

namespace Detail {

class RPageStorage;

// clang-format off
/**
\class ROOT::Experimental::RTreeFieldBase
\ingroup Forest
\brief A field translates read and write calls from/to underlying columns to/from tree values

A field is a serializable C++ type or a container for a collection of sub fields. The RTreeFieldBase and its
type-safe descendants provide the object to column mapper. They map C++ objects to primitive columns.  The
mapping is trivial for simple types such as 'double'. Complex types resolve to multiple primitive columns.
The field knows based on its type and the field name the type(s) and name(s) of the columns.
*/
// clang-format on
class RTreeFieldBase {
private:
   /// The field name is a unique within a tree and also the basis for the column name(s)
   std::string fName;
   /// The C++ type captured by this field
   std::string fType;
   /// A field on a trivial type that maps as-is to a single column
   bool fIsSimple;

protected:
   /// Collections an classes own sub fields
   std::vector<std::unique_ptr<RTreeFieldBase>> fSubFields;
   /// Sub fields point to their mother field
   RTreeFieldBase* fParent;
   /// All fields have a main column. For collection fields, the main column is the index field. Points into fColumns.
   RColumn* fPrincipalColumn;
   /// The columns are connected either to a sink or to a source (not to both); they are owned by the field.
   std::vector<std::unique_ptr<RColumn>> fColumns;

   /// Creates the backing columns corresponsing to the field type and name
   virtual void DoGenerateColumns() = 0;

   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual void DoAppend(const RTreeValueBase& value);
   virtual void DoRead(TreeIndex_t index, RTreeValueBase* value);
   virtual void DoReadV(TreeIndex_t index, TreeIndex_t count, void* dst);

   /// Simple fields resolve directly to a column. When simple fields generate a value, they use
   /// MakeSimpleValue to connect the value's fMappedElement member to the principal column
   void MakeSimpleValue(const RColumnElementBase &mappedElement, RTreeValueBase *value) {
      value->SetMappedElement(mappedElement);
   }

public:
   /// Field names convey the level of subfields; sub fields (nested collections) are separated by a dot
   static constexpr char kCollectionSeparator = '/';

   /// Iterates over the sub fields in depth-first search order
   class RIterator : public std::iterator<std::forward_iterator_tag, Detail::RTreeFieldBase> {
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
   RTreeFieldBase(std::string_view name, std::string_view type, bool isSimple);
   RTreeFieldBase(const RTreeFieldBase&) = delete;
   RTreeFieldBase& operator =(const RTreeFieldBase&) = delete;
   virtual ~RTreeFieldBase();

   /// Factory method to resurrect a field from the stored on-disk type information
   static RTreeFieldBase *Create(const std::string &fieldName, const std::string &typeName);
   /// Get the tail of the field name up to the last dot
   static std::string GetLeafName(const std::string &fullName);
   /// Get the name for an item sub field that is part of a collection, e.g. the float field of std::vector<float>
   static std::string GetCollectionName(const std::string &parentName);

   /// Registeres (or re-registers) the backing columns with the physical storage
   void ConnectColumns(Detail::RPageStorage *pageStorage);
   /// Returns the number of columns generated to store data for the field; defaults to 1
   virtual unsigned int GetNColumns() const = 0;

   /// Generates a tree value of the field type and allocates new initialized memory according to the type.
   RTreeValueBase GenerateValue();
   /// Generates a tree value in a given location of size at least GetValueSize(). Assumes that where has been
   /// allocated by malloc().
   virtual RTreeValueBase GenerateValue(void *where) = 0;
   /// Releases the resources acquired during GenerateValue (memory and constructor)
   /// This implementation works for simple types but needs to be overwritten for complex ones
   virtual void DestroyValue(const RTreeValueBase &value, bool dtorOnly = false);
   /// Creates a value from a memory location with an already constructed object
   virtual RTreeValueBase CaptureValue(void *where) = 0;
   /// The number of bytes taken by a value of the appropriate type
   virtual size_t GetValueSize() const = 0;

   /// Write the given value to a tree. The value object has to be of the same type as the field.
   void Append(const RTreeValueBase& value) {
      if (!fIsSimple) {
         DoAppend(value);
         return;
      }
      fPrincipalColumn->Append(value.fMappedElement);
   }

   /// Populate a single value with data from the tree, which needs to be of the fitting type.
   /// Reading copies data into the memory wrapped by the tree value.
   void Read(TreeIndex_t index, RTreeValueBase* value) {
      if (!fIsSimple) {
         DoRead(index, value);
         return;
      }
      printf("Simple reading index %lu in field %s\n", index, fName.c_str());
      fPrincipalColumn->Read(index, &value->fMappedElement);
   }

   /// Type unsafe bulk read interface; dst must point to a vector of objects of the field type.
   /// TODO(jblomer): can this be type safe?
   void ReadV(TreeIndex_t index, TreeIndex_t count, void *dst)
   {
      if (!fIsSimple) {
         DoReadV(index, count, dst);
         return;
      }
      //fPrincipalColumn->ReadV(index, count, dst);
   }

   /// Only for simple types, let the pointer wrapped by the tree value simply point into the page buffer.
   /// The resulting tree value may only be used for as long as no request to another item of this field is made
   /// because another index might trigger a swap of the page buffer.
   /// The dst location must be an object of the field type.
   void Map(TreeIndex_t /*index*/, void** /*dst*/) {
      if (!fIsSimple) {
         // TODO(jblomer)
      }
      //fPrincipalColumn->Map(index, dst);
   }

   /// The number of elements in the principal column. For top level fields, the number of entries.
   TreeIndex_t GetNItems();

   /// Ensure that all received items are written from page buffers to the storage.
   void Flush() const;

   void Attach(std::unique_ptr<Detail::RTreeFieldBase> child);

   std::string GetName() const { return fName; }
   std::string GetType() const { return fType; }
   const RTreeFieldBase* GetParent() const { return fParent; }

   RIterator begin();
   RIterator end();
};

} // namespace Detail

/// The container field for a tree model, which itself has no physical representation
class RTreeFieldRoot : public Detail::RTreeFieldBase {
public:
   RTreeFieldRoot() : Detail::RTreeFieldBase("", "", false /* isSimple */) {}

   void DoGenerateColumns() final {}
   unsigned int GetNColumns() const final { return 0; }
   Detail::RTreeValueBase GenerateValue(void*) { return Detail::RTreeValueBase(); }
   Detail::RTreeValueBase CaptureValue(void*) final { return Detail::RTreeValueBase(); }
   size_t GetValueSize() const final { return 0; }
};

/// The field for a class with dictionary
class RTreeFieldClass : public Detail::RTreeFieldBase {
private:
   TClass* fClass;
protected:
   void DoAppend(const Detail::RTreeValueBase& value) final;
   void DoRead(TreeIndex_t index, Detail::RTreeValueBase* value) final;
public:
   RTreeFieldClass(std::string_view fieldName, std::string_view className);
   ~RTreeFieldClass() = default;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final;
   Detail::RTreeValueBase GenerateValue(void* where) override;
   void DestroyValue(const Detail::RTreeValueBase& value, bool dtorOnly = false) final;
   Detail::RTreeValueBase CaptureValue(void *where) final;
   size_t GetValueSize() const override;
};

/// The generic field for a (nested) vector
class RTreeFieldVector : public Detail::RTreeFieldBase {
private:
   size_t fItemSize;
   TreeIndex_t fNWritten;

protected:
   void DoAppend(const Detail::RTreeValueBase& value) final;
   void DoRead(TreeIndex_t index, Detail::RTreeValueBase* value) final;

public:
   RTreeFieldVector(std::string_view fieldName, std::unique_ptr<Detail::RTreeFieldBase> itemField);
   ~RTreeFieldVector() = default;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final;
   Detail::RTreeValueBase GenerateValue(void* where) override;
   void DestroyValue(const Detail::RTreeValueBase& value, bool dtorOnly = false) final;
   Detail::RTreeValueBase CaptureValue(void *where) override;
   size_t GetValueSize() const override;
};


/// Classes with dictionaries that can be inspected by TClass
template <typename T, typename=void>
class RTreeField : public RTreeFieldClass {
public:
   static std::string MyTypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RTreeField(std::string_view name) : RTreeFieldClass(name, MyTypeName()) {}
   ~RTreeField() = default;

   template <typename... ArgsT>
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where, ArgsT&&... args)
   {
      return ROOT::Experimental::RTreeValue<T>(this, static_cast<T*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where) final { return GenerateValue(where, T()); }
};

} // namespace Experimental
} // namespace ROOT


template <>
class ROOT::Experimental::RTreeField<float> : public ROOT::Experimental::Detail::RTreeFieldBase {
public:
   static std::string MyTypeName() { return "float"; }
   explicit RTreeField(std::string_view name) : Detail::RTreeFieldBase(name, MyTypeName(), true /* isSimple */) {}
   ~RTreeField() = default;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 1; }

   template <typename... ArgsT>
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where, ArgsT&&... args)
   {
      ROOT::Experimental::RTreeValue<float> v(this, static_cast<float*>(where), std::forward<ArgsT>(args)...);
      MakeSimpleValue(Detail::RColumnElement<float, EColumnType::kReal32>(static_cast<float*>(where)), &v);
      return v;
   }
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where) final { return GenerateValue(where, 0.0); }
   Detail::RTreeValueBase CaptureValue(void *where) final {
      ROOT::Experimental::RTreeValue<float> v(true, this, static_cast<float*>(where));
      MakeSimpleValue(Detail::RColumnElement<float, EColumnType::kReal32>(static_cast<float*>(where)), &v);
      return v;
   }
   size_t GetValueSize() const final { return sizeof(float); }
};

template <>
class ROOT::Experimental::RTreeField<std::string> : public ROOT::Experimental::Detail::RTreeFieldBase {
private:
   TreeIndex_t fIndex;
   Detail::RColumnElement<TreeIndex_t, EColumnType::kIndex> fElemIndex;

   void DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& value) final;
   void DoRead(ROOT::Experimental::TreeIndex_t index, ROOT::Experimental::Detail::RTreeValueBase* value) final;

public:
   static std::string MyTypeName() { return "std::string"; }
   explicit RTreeField(std::string_view name)
      : Detail::RTreeFieldBase(name, MyTypeName(), false /* isSimple */), fIndex(0), fElemIndex(&fIndex) {}
   ~RTreeField() = default;

   void DoGenerateColumns() final;
   unsigned int GetNColumns() const final { return 2; }

   template <typename... ArgsT>
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where, ArgsT&&... args)
   {
      return ROOT::Experimental::RTreeValue<std::string>(
         this, static_cast<std::string*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where) final { return GenerateValue(where, ""); }
   void DestroyValue(const Detail::RTreeValueBase& value, bool dtorOnly = false) {
      auto str = static_cast<std::string*>(value.GetRawPtr());
      str->~basic_string(); // TODO(jblomer) C++17 std::destroy_at
      if (!dtorOnly)
         free(str);
   }
   Detail::RTreeValueBase CaptureValue(void *where) {
      return ROOT::Experimental::RTreeValue<std::string>(true, this, static_cast<std::string*>(where));
   }
   size_t GetValueSize() const final { return sizeof(std::string); }
};


template <typename ItemT>
class ROOT::Experimental::RTreeField<std::vector<ItemT>> : public ROOT::Experimental::RTreeFieldVector {
   using ContainerT = typename std::vector<ItemT>;
public:
   static std::string MyTypeName() { return "std::vector<" + RTreeField<ItemT>::MyTypeName() + ">"; }
   explicit RTreeField(std::string_view name)
      : RTreeFieldVector(name, std::make_unique<RTreeField<ItemT>>(GetCollectionName(name.to_string())))
   {}
   ~RTreeField() = default;

   template <typename... ArgsT>
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where, ArgsT&&... args)
   {
      return ROOT::Experimental::RTreeValue<ContainerT>(
         this, static_cast<ContainerT*>(where), std::forward<ArgsT>(args)...);
   }
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue(void* where) final {
      return GenerateValue(where, ContainerT());
   }
   Detail::RTreeValueBase CaptureValue(void *where) final {
      return ROOT::Experimental::RTreeValue<ContainerT>(true, this, static_cast<ContainerT*>(where));
   }
   size_t GetValueSize() const final { return sizeof(ContainerT); }
};

#endif
