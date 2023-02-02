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
#include <Rtypes.h>  // Long64_t, R__CLING_PTRCHECK

#include <limits>
#include <type_traits>

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

}
}
}

#endif
