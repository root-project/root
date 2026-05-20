// Author: Enrico Guiraud CERN 09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDEFINEREADER
#define ROOT_RDF_RDEFINEREADER

#include "RColumnReaderBase.hxx"
#include "RDefineBase.hxx"
#include "Utils.hxx"
#include <Rtypes.h>  // Long64_t, R__CLING_PTRCHECK

#include <limits>
#include <type_traits>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ROOT {
namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

/// Column reader for defined columns.
class R__CLING_PTRCHECK(off) RDefineReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   /// Non-owning reference to the node responsible for the defined column.
   RDFDetail::RDefineBase &fDefine;

   /// Non-owning ptr to the defined value.
   void *fValuePtr = nullptr;

   /// The slot this value belongs to.
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();

   void *GetImpl(Long64_t entry) final
   {
      fDefine.Update(fSlot, entry);
      return fValuePtr;
   }

public:
   RDefineReader(unsigned int slot, RDFDetail::RDefineBase &define)
      : fDefine(define), fValuePtr(define.GetValuePtr(slot)), fSlot(slot)
   {
   }
};

/// A helper type that keeps track of RDefine objects and their corresponding RDefineReaders.
class RDefinesWithReaders {

   // this is a shared_ptr only because we have to track its lifetime with a weak_ptr that we pass to jitted code
   // (see BookDefineJit). it is never null.
   std::shared_ptr<ROOT::Detail::RDF::RDefineBase> fDefine;
   // Column readers per variation (in the map) per slot (in the vector).
   std::vector<std::unordered_map<std::string_view, std::shared_ptr<RDefineReader>>> fReadersPerVariation;

   // Strings that were already used to represent column names in this RDataFrame instance.
   ROOT::Internal::RDF::RStringCache &fCachedColNames;

public:
   RDefinesWithReaders(std::shared_ptr<ROOT::Detail::RDF::RDefineBase> define, unsigned int nSlots,
                       ROOT::Internal::RDF::RStringCache &cachedColNames);
   ROOT::Detail::RDF::RDefineBase &GetDefine() const { return *fDefine; }
   ROOT::Internal::RDF::RDefineReader &GetReader(unsigned int slot, std::string_view variationName);
};

} // namespace RDF
} // namespace Internal
}

#endif
