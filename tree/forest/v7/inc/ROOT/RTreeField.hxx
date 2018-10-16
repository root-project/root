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

#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {

class RTreeValueBase;
class RPageStorage;

// clang-format off
/**
\class ROOT::Experimental::RTreeFieldBase
\ingroup Forest
\brief An field translates read and write calls from/to underlying columns to/from tere values

A field is a serializable C++ type. The RTreeFieldBase and its type-safe descendants provide the object
to column mapper. They map C++ objects to primitive columns, where the mapping is trivial for simple types
such as 'double'. The field knows based on its type and the field name the type(s) and name(s) of the columns.
*/
// clang-format on
class RTreeFieldBase {
private:
   /// A field on a trivial type that maps as-is to a single column
   bool fIsSimple;
   /// All fields have a main column. For collection fields, the main column is the index field. Points into fColumns.
   RColumn *fPrincipalColumn;
   /// The columns are connected either to a sink or to a source (not to both); they are owned by the field.
   std::vector<RColumn> fColumns;

protected:
   /// Operations on values of complex types, e.g. ones that involve multiple columns or for which no direct
   /// column type exists.
   virtual void DoAppend(const RTreeValueBase& value) = 0;
   virtual void DoRead(TreeIndex_t index, const RTreeValueBase& value) = 0;
   virtual void DoReadV(TreeIndex_t index, TreeIndex_t count, void* dst) = 0;

public:
   /// The constructor creates the underlying column objects and connects them to either a sink or a source.
   RTreeFieldBase(std::string_view name);
   virtual ~RTreeFieldBase();

   /// Registeres the backing columns with the physical storage
   virtual void GenerateColumns(Detail::RPageStorage &storage) = 0;

   /// Generates a tree value of the field type.
   virtual std::unique_ptr<RTreeValueBase> GenerateValue() = 0;

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

   RPageSource* GetSource();
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

   void GenerateColumns(Detail::RPageStorage &storage) final;
   std::unique_ptr<Detail::RTreeValueBase> GenerateValue() final;
   void Attach(RTreeFieldBase* child);
};


/// Supported types are implemented as template specializations
template <typename T>
class RTreeField : public Detail::RTreeFieldBase {
};

/// TODO(jblomer): template specializations for simple types and TClass

} // namespace Experimental
} // namespace ROOT

#endif
