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

#include <Rtypes.h>

namespace ROOT {
namespace Detail {
namespace RDF {

/**
\class ROOT::Internal::RDF::RColumnReaderBase
\ingroup dataframe
\brief Pure virtual base class for all column reader types

This pure virtual class provides a common base class for the different column reader types, e.g. RTreeColumnReader and
RDSColumnReader.
**/
class R__CLING_PTRCHECK(off) RColumnReaderBase {
   Long64_t fLoadedEntry = -1;

public:
   virtual ~RColumnReaderBase() = default;

   /// Load the column value for the given entry.
   /// \param entry The entry number to load.
   /// \param mask The entry mask. Values will be loaded only for entries for which the mask equals true.
   void Load(Long64_t entry, bool mask)
   {
      // For now, as `mask` is just a single boolean, as an optimization we can return early here if `mask == false`.
      if (mask) {
         fLoadedEntry = entry;
         this->LoadImpl(entry, mask);
      }
   }

   /// Return the column value for the given entry.
   /// \tparam T The column type
   /// \param entry The entry number
   ///
   /// The caller is responsible for checking that the returned value actually
   /// exists.
   template <typename T>
   T *TryGet(std::size_t idx)
   {
      return static_cast<T *>(GetImpl(idx));
   }

private:
   virtual void *GetImpl(std::size_t idx) = 0;
   virtual void LoadImpl(Long64_t /*entry*/, bool /*mask*/) = 0;
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif
