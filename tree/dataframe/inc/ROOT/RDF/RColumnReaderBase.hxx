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
#include <ROOT/RDF/RMaskedEntryRange.hxx>

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

public:
   virtual ~RColumnReaderBase() = default;

   /// Load the column value for the given entry.
   /// \param entry The entry number to load.
   /// \param mask The entry mask. Values will be loaded only for entries for which the mask equals true.
   void Load(const ROOT::Internal::RDF::RMaskedEntryRange &mask) { LoadImpl(mask); }

   /// Return the column value for the given entry.
   /// \tparam T The column type
   /// \param entry The entry number
   ///
   /// The caller is responsible for checking that the returned value actually
   /// exists.
   template <typename T>
   T *TryGet(std::size_t entryInBulk)
   {
      return static_cast<T *>(GetImpl(entryInBulk));
   }

private:
   virtual void *GetImpl(std::size_t entryInBulk) = 0;
   virtual void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) = 0;
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif
