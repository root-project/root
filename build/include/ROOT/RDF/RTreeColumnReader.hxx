// Author: Enrico Guiraud CERN 09/2020
// Author: Vincenzo Eduardo Padulano CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RTREECOLUMNREADER
#define ROOT_RDF_RTREECOLUMNREADER

#include "RColumnReaderBase.hxx"
#include <ROOT/RVec.hxx>
#include "ROOT/RDF/Utils.hxx"
#include <Rtypes.h> // Long64_t, R__CLING_PTRCHECK

#include <array>
#include <memory>
#include <string>
#include <cstddef>

class TTreeReader;

namespace ROOT {
namespace Internal {

class TTreeReaderOpaqueValue;
class TTreeReaderUntypedArray;
class TTreeReaderUntypedValue;

namespace RDF {

class R__CLING_PTRCHECK(off) RTreeOpaqueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<ROOT::Internal::TTreeReaderOpaqueValue> fTreeValue;

   void *GetImpl(Long64_t) override;

public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeOpaqueColumnReader(TTreeReader &r, std::string_view colName);

   // Rule of five

   RTreeOpaqueColumnReader(const RTreeOpaqueColumnReader &) = delete;
   RTreeOpaqueColumnReader &operator=(const RTreeOpaqueColumnReader &) = delete;
   RTreeOpaqueColumnReader(RTreeOpaqueColumnReader &&) = delete;
   RTreeOpaqueColumnReader &operator=(RTreeOpaqueColumnReader &&) = delete;
   ~RTreeOpaqueColumnReader() final; // Define destructor when data member type is fully defined
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderUntypedValue
class R__CLING_PTRCHECK(off) RTreeUntypedValueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedValue> fTreeValue;

   void *GetImpl(Long64_t) override;

public:
   RTreeUntypedValueColumnReader(TTreeReader &r, std::string_view colName, std::string_view typeName);

   // Rule of five

   RTreeUntypedValueColumnReader(const RTreeUntypedValueColumnReader &) = delete;
   RTreeUntypedValueColumnReader &operator=(const RTreeUntypedValueColumnReader &) = delete;
   RTreeUntypedValueColumnReader(RTreeUntypedValueColumnReader &&) = delete;
   RTreeUntypedValueColumnReader &operator=(RTreeUntypedValueColumnReader &&) = delete;
   ~RTreeUntypedValueColumnReader() final; // Define destructor when data member type is fully defined
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderUntypedValue
class R__CLING_PTRCHECK(off) RTreeUntypedArrayColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
public:
   enum class ECollectionType {
      kRVec,
      kStdArray,
      kRVecBool
   };

   RTreeUntypedArrayColumnReader(TTreeReader &r, std::string_view colName, std::string_view valueTypeName,
                                 ECollectionType collType);

   // Rule of five

   RTreeUntypedArrayColumnReader(const RTreeUntypedArrayColumnReader &) = delete;
   RTreeUntypedArrayColumnReader &operator=(const RTreeUntypedArrayColumnReader &) = delete;
   RTreeUntypedArrayColumnReader(RTreeUntypedArrayColumnReader &&) = delete;
   RTreeUntypedArrayColumnReader &operator=(RTreeUntypedArrayColumnReader &&) = delete;
   ~RTreeUntypedArrayColumnReader() final; // Define destructor when data member type is fully defined

private:
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedArray> fTreeArray;
   ECollectionType fCollectionType;

   using Byte_t = std::byte;
   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout.
   RVec<Byte_t> fRVec{};

   Long64_t fLastEntry = -1;

   /// The size of the collection value type.
   std::size_t fValueSize{};

   /// Whether we already printed a warning about performing a copy of the TTreeReaderArray contents
   bool fCopyWarningPrinted = false;

   void *GetImpl(Long64_t entry) override;
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
