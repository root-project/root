/// \file RNTupleDSFields.cxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
/// \date 2026-09-23

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFieldBase.hxx>

// clang-format off
/**
 * \class ROOT::Internal::RDF::RRDFCardinalityFieldBase
 * \ingroup dataframe
 * \brief Base for an artificial field that transforms an RNTuple column containing the offset of collections into collection sizes.
 *
 * It is used to provide the "number of" RDF columns for collections, e.g.  `R_rdf_sizeof_jets` for a collection named `jets`.
 */
// clang-format on
namespace ROOT::Internal::RDF {
class RRDFCardinalityFieldBase : public ROOT::RFieldBase {
protected:
   // We construct these fields and know that they match the page source
   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

   RRDFCardinalityFieldBase(std::string_view name, std::string_view type)
      : ROOT::RFieldBase(name, type, ROOT::ENTupleStructure::kPlain, false /* isSimple */)
   {
   }

   // Field is only used for reading
   void GenerateColumns() final { throw RException(R__FAIL("Cardinality fields must only be used for reading")); }
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final
   {
      GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
   }

public:
   RRDFCardinalityFieldBase(const RRDFCardinalityFieldBase &other) = delete;
   RRDFCardinalityFieldBase &operator=(const RRDFCardinalityFieldBase &other) = delete;
   RRDFCardinalityFieldBase(RRDFCardinalityFieldBase &&other) = default;
   RRDFCardinalityFieldBase &operator=(RRDFCardinalityFieldBase &&other) = default;
   ~RRDFCardinalityFieldBase() override = default;

   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                     {ENTupleColumnType::kIndex64},
                                                     {ENTupleColumnType::kSplitIndex32},
                                                     {ENTupleColumnType::kIndex32}},
                                                    {});
      return representations;
   }
};

// clang-format off
/**
 * \class ROOT::Internal::RDF::RRDFCardinalityField
 * \ingroup dataframe
 * \brief An artificial field that transforms an RNTuple column containing the offset of collections into collection sizes.
 *
 * It is used to provide the "number of" RDF columns for collections, e.g.  `R_rdf_sizeof_jets` for a collection named `jets`.
 *
 * This is similar to the RCardinalityField but it presents itself as an integer type.
 * The template argument T must be an integral type.
 */
// clang-format on
template <typename T>
class RRDFCardinalityField final : public RRDFCardinalityFieldBase {
   static_assert(std::is_integral_v<T>, "T must be an integral type");

   inline void CheckSize(ROOT::NTupleSize_t size) const
   {
      if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::uint64_t>)
         return;
      if (size > static_cast<ROOT::NTupleSize_t>(std::numeric_limits<T>::max())) {
         throw RException(R__FAIL(std::string("integer overflow in field ") + GetFieldName() +
                                  ". Please read the column with a larger-sized integral type."));
      }
   }

protected:
   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RRDFCardinalityField>(newName);
   }
   void ConstructValue(void *where) const final { *static_cast<T *>(where) = 0; }

public:
   RRDFCardinalityField(std::string_view name)
      : RRDFCardinalityFieldBase(name, ROOT::Internal::GetRenormalizedTypeName(typeid(T)))
   {
   }
   RRDFCardinalityField(const RRDFCardinalityField &other) = delete;
   RRDFCardinalityField &operator=(const RRDFCardinalityField &other) = delete;
   RRDFCardinalityField(RRDFCardinalityField &&other) = default;
   RRDFCardinalityField &operator=(RRDFCardinalityField &&other) = default;
   ~RRDFCardinalityField() override = default;

   std::size_t GetValueSize() const final { return sizeof(T); }
   std::size_t GetAlignment() const final { return alignof(T); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      CheckSize(size);
      *static_cast<T *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(localIndex, &collectionStart, &size);
      CheckSize(size);
      *static_cast<T *>(to) = size;
   }
};

// clang-format off
/**
 * \class ROOT::Internal::RDF::RArraySizeField
 * \ingroup dataframe
 * \brief An artificial field that provides the size of a fixed-size array
 *
 * This is the implementation of `R_rdf_sizeof_column` in case `column` contains
 * fixed-size arrays on disk.
 */
// clang-format on
class RArraySizeField final : public ROOT::RFieldBase {
private:
   std::size_t fArrayLength;

   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RArraySizeField>(newName, fArrayLength);
   }
   void GenerateColumns() final { throw RException(R__FAIL("RArraySizeField fields must only be used for reading")); }
   void GenerateColumns(const ROOT::RNTupleDescriptor &) final {}
   void ReadGlobalImpl(ROOT::NTupleSize_t /*globalIndex*/, void *to) final
   {
      *static_cast<std::size_t *>(to) = fArrayLength;
   }
   void ReadInClusterImpl(RNTupleLocalIndex /*localIndex*/, void *to) final
   {
      *static_cast<std::size_t *>(to) = fArrayLength;
   }

   // We construct these fields and know that they match the page source
   void ReconcileOnDiskField(const RNTupleDescriptor &) final {}

public:
   RArraySizeField(std::string_view name, std::size_t arrayLength)
      : ROOT::RFieldBase(name, ROOT::Internal::GetRenormalizedTypeName(typeid(std::size_t)),
                         ROOT::ENTupleStructure::kPlain, false /* isSimple */),
        fArrayLength(arrayLength)
   {
   }
   RArraySizeField(const RArraySizeField &other) = delete;
   RArraySizeField &operator=(const RArraySizeField &other) = delete;
   RArraySizeField(RArraySizeField &&other) = default;
   RArraySizeField &operator=(RArraySizeField &&other) = default;
   ~RArraySizeField() final = default;

   void ConstructValue(void *where) const final { *static_cast<std::size_t *>(where) = 0; }
   std::size_t GetValueSize() const final { return sizeof(std::size_t); }
   std::size_t GetAlignment() const final { return alignof(std::size_t); }
};
} // namespace ROOT::Internal::RDF
