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
#include <ROOT/RForestUtil.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeValue.hxx>

#include <TError.h>

#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class RTreeFieldCollection;

namespace Detail {

class RTreeValueBase;
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
   friend class ROOT::Experimental::RTreeFieldCollection;

private:
   /// The field name is a unique within a tree and also the basis for the column name(s)
   std::string fName;
   /// The C++ type captured by this field
   std::string fType;
   /// A field on a trivial type that maps as-is to a single column
   bool fIsSimple;
   /// Collections have sub fields
   std::vector<RTreeFieldBase*> fSubFields;
   /// Sub fields point to their mother field
   RTreeFieldBase* fParent;


protected:
   /// All fields have a main column. For collection fields, the main column is the index field. Points into fColumns.
   RColumn* fPrincipalColumn;
   /// The columns are connected either to a sink or to a source (not to both); they are owned by the field.
   std::vector<std::unique_ptr<RColumn>> fColumns;

   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual void DoAppend(const RTreeValueBase& value);
   virtual void DoRead(TreeIndex_t index, const RTreeValueBase& value);
   virtual void DoReadV(TreeIndex_t index, TreeIndex_t count, void* dst);

public:
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
   virtual ~RTreeFieldBase();

   /// Registeres the backing columns with the physical storage
   virtual void GenerateColumns(Detail::RPageStorage *pageStorage) = 0;

   /// Generates a tree value of the field type.
   virtual RTreeValueBase GenerateValue() = 0;

   /// Write the given value to a tree. The value object has to be of the same type as the field.
   void Append(const RTreeValueBase &value) {
     if (!fIsSimple) {
        DoAppend(value);
        return;
     }
     fPrincipalColumn->Append(*(value.fPrincipalElement));
   }

   /// Populate a single value with data from the tree, which needs to be of the fitting type.
   /// Reading copies data into the memory wrapped by the tree value.
   void Read(TreeIndex_t index, RTreeValueBase &value) {
      if (!fIsSimple) {
         DoRead(index, value);
         return;
      }
      fPrincipalColumn->Read(index, value.fPrincipalElement);
   }

   /// Type unsafe bulk read interface; dst must point to a vector of objects of the field type.
   /// TODO(jblomer): can this be type safe?
   void ReadV(TreeIndex_t index, TreeIndex_t count, void *dst)
   {
      if (!fIsSimple) {
         DoReadV(index, count, dst);
         return;
      }
      fPrincipalColumn->ReadV(index, count, dst);
   }

   /// Only for simple types, let the pointer wrapped by the tree value simply point into the page buffer.
   /// The resulting tree value may only be used for as long as no request to another item of this field is made
   /// because another index might trigger a swap of the page buffer.
   /// The dst location must be an object of the field type.
   void Map(TreeIndex_t index, void** dst) {
      if (!fIsSimple) {
         // TODO(jblomer)
      }
      fPrincipalColumn->Map(index, dst);
   }

   /// The number of elements in the principal column. For top level fields, the number of entries.
   TreeIndex_t GetNItems();

   /// Ensure that all received items are written from page buffers to the storage.
   void Flush();

   std::string GetName() const { return fName; }
   std::string GetType() const { return fType; }
   const RTreeFieldBase* GetParent() const { return fParent; }

   RIterator begin();
   RIterator end();
};

} // namespace Detail

/// A Field covering a subtree containing a collection of sub fields (like std::vector<Jet>)
class RTreeFieldCollection : public Detail::RTreeFieldBase {
protected:
   void DoAppend(const Detail::RTreeValueBase& value) final;
   void DoRead(TreeIndex_t index, const Detail::RTreeValueBase& value) final;
   void DoReadV(TreeIndex_t index, TreeIndex_t count, void *dst) final;

public:
   RTreeFieldCollection(std::string_view name);
   ~RTreeFieldCollection();

   void GenerateColumns(Detail::RPageStorage* pageStorage) final;
   Detail::RTreeValueBase GenerateValue() final;
   void Attach(Detail::RTreeFieldBase* child);
};


/// Supported types are implemented as template specializations
template <typename T>
class RTreeField : public Detail::RTreeFieldBase {
};

} // namespace Experimental
} // namespace ROOT


template <>
class ROOT::Experimental::RTreeField<float> : public ROOT::Experimental::Detail::RTreeFieldBase {
public:
   explicit RTreeField(std::string_view name) : Detail::RTreeFieldBase(name, "float", true /* isSimple */) {}
   ~RTreeField() = default;

   void GenerateColumns(ROOT::Experimental::Detail::RPageStorage* pageStorage) final;
   ROOT::Experimental::Detail::RTreeValueBase GenerateValue() final;
};

#endif
