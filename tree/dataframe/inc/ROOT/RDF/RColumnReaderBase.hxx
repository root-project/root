// Author: Enrico Guiraud CERN 09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_RDF_RCOLUMNREADERBASE
#define ROOT_INTERNAL_RDF_RCOLUMNREADERBASE

#include <ROOT/RDF/RMaskedEntryRange.hxx>

#include <Rtypes.h>

namespace ROOT {
namespace Detail {
namespace RDF {

namespace RDFInternal = ROOT::Internal::RDF;

/**
\class ROOT::Internal::RDF::RColumnReaderBase
\ingroup dataframe
\brief Pure virtual base class for all column reader types

This pure virtual class provides a common base class for the different column reader types, e.g. RTreeColumnReader and
RDSColumnReader.
**/
class R__CLING_PTRCHECK(off) RColumnReaderBase {
public:
   virtual ~RColumnReaderBase() = default;

   /// Load the column value for the given entry.
   /// \param entry The entry number to load.
   /// \param mask The entry mask. Values will be loaded only for entries for which the mask equals true.
   void Load(const RDFInternal::RMaskedEntryRange &mask) { this->LoadImpl(mask); }

   /// Return the column value for the given entry.
   /// \tparam T The column type
   /// \param idx The index of the value to load with respect to the beginning of the last entry mask passed to Load().
   template <typename T>
   T &Get(std::size_t idx)
   {
      return *static_cast<T *>(GetImpl(idx * sizeof(T)));
   }

private:
   /// Return the type-erased column value for the given entry.
   /// \param offset Offset, in bytes, of the address of the element to retrieve w.r.t. the first element in the bulk.
   virtual void *GetImpl(std::size_t offset) = 0;
   // TODO remove the default implementation when all readers will be required to do something non-trivial at load time
   virtual void LoadImpl(const Internal::RDF::RMaskedEntryRange &) {}
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif
