// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRANGEBASE
#define ROOT_RRANGEBASE

#include "ROOT/RDF/RNodeBase.hxx"
#include "RtypesCore.h"

#include <unordered_map>

namespace ROOT {
namespace Internal {
namespace RDF {
namespace GraphDrawing {
class GraphNode;
}
} // namespace RDF
} // namespace Internal

namespace Detail {
namespace RDF {

class RLoopManager;

class RRangeBase : public RNodeBase {
protected:
   unsigned int fStart;
   unsigned int fStop;
   unsigned int fStride;
   Long64_t fLastCheckedEntry{-1};
   bool fLastResult{true};
   ULong64_t fNProcessedEntries{0};
   bool fHasStopped{false};    ///< True if the end of the range has been reached
   const unsigned int fNSlots; ///< Number of thread slots used by this node, inherited from parent node.
   std::unordered_map<std::string, std::shared_ptr<RRangeBase>> fVariedRanges;

public:
   RRangeBase(RLoopManager *implPtr, unsigned int start, unsigned int stop, unsigned int stride,
              const unsigned int nSlots, const std::vector<std::string> &prevVariations);

   // Rule of five

   RRangeBase(const RRangeBase &) = delete;
   RRangeBase &operator=(const RRangeBase &) = delete;
   RRangeBase(RRangeBase &&) = delete;
   RRangeBase &operator=(RRangeBase &&) = delete;
   ~RRangeBase() override;

   void InitNode();
};

} // ns RDF
} // ns Detail
} // namespace ROOT

#endif // ROOT_RRANGEBASE
