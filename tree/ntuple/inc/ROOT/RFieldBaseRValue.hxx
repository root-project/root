/// \file ROOT/RValue.hxx
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

#ifndef ROOT_RValue
#define ROOT_RValue

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx> // for RField<T>::TypeName()
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace ROOT {

/// Points to an object with RNTuple I/O support and keeps a pointer to the corresponding field.
/// Fields can create RValue objects through RFieldBase::CreateValue(), RFieldBase::BindValue(), or
/// RFieldBase::SplitValue().
class RFieldBase::RValue {
   friend class RFieldBase;

private:
   RFieldBase *fField = nullptr; ///< The field that created the RValue
   /// Set by Bind() or by RFieldBase::CreateValue(), RFieldBase::SplitValue() or RFieldBase::BindValue()
   std::shared_ptr<void> fObjPtr;
   RValue(RFieldBase *field, std::shared_ptr<void> objPtr) : fField(field), fObjPtr(objPtr) {}

   template <typename T>
   void EnsureMatchingType() const
   {
      if constexpr (!std::is_void_v<T>) {
         if (fField->GetTypeName() != ROOT::RField<T>::TypeName()) {
            throw RException(
               R__FAIL("type mismatch: " + fField->GetTypeName() + " vs. " + ROOT::RField<T>::TypeName()));
         }
      }
   }

public:
   RValue(const RValue &) = default;
   RValue &operator=(const RValue &) = default;
   RValue(RValue &&other) = default;
   RValue &operator=(RValue &&other) = default;
   ~RValue() = default;

   std::size_t Append() { return fField->Append(fObjPtr.get()); }
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
class RFieldBase::RBulkValues {
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

} // namespace ROOT

#endif
