// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RACTIONBASE
#define ROOT_RACTIONBASE

#include "ROOT/RDF/RBookedCustomColumns.hxx"
#include "ROOT/RDF/Utils.hxx" // ColumnNames_t
#include "RtypesCore.h"

#include <memory>
#include <string>

namespace ROOT {

namespace Detail {
namespace RDF {
class RLoopManager;
class RCustomColumnBase;
}
}

namespace Internal {
namespace RDF {
namespace GraphDrawing {
class GraphNode;
}

using namespace ROOT::Detail::RDF;

// fwd decl for RActionBase
namespace GraphDrawing {
bool CheckIfDefaultOrDSColumn(const std::string &name,
                              const std::shared_ptr<ROOT::Detail::RDF::RCustomColumnBase> &column);
} // namespace GraphDrawing

class RActionBase {
protected:
   /// A raw pointer to the RLoopManager at the root of this functional graph.
   RLoopManager *fLoopManager;

   const unsigned int fNSlots; ///< Number of thread slots used by this node.
   bool fHasRun = false;
   const ColumnNames_t fColumnNames;

   RBookedCustomColumns fCustomColumns;

public:
   RActionBase(RLoopManager *implPtr, const unsigned int nSlots, const ColumnNames_t &colNames,
               const RBookedCustomColumns &customColumns);
   RActionBase(const RActionBase &) = delete;
   RActionBase &operator=(const RActionBase &) = delete;
   virtual ~RActionBase();

   virtual void Run(unsigned int slot, Long64_t entry) = 0;
   virtual void Initialize() = 0;
   virtual void InitSlot(TTreeReader *r, unsigned int slot) = 0;
   virtual void TriggerChildrenCount() = 0;
   virtual void ClearValueReaders(unsigned int slot) = 0;
   virtual void FinalizeSlot(unsigned int) = 0;
   virtual void Finalize() = 0;
   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   virtual void *PartialUpdate(unsigned int slot) = 0;
   virtual bool HasRun() const { return fHasRun; }

   virtual std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> GetGraph() = 0;
};

} // ns RDF
} // ns Internal
} // ns ROOT

#endif // ROOT_RACTIONBASE
