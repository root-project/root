// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSLOTSTACK
#define ROOT_RSLOTSTACK

#include <ROOT/TSpinMutex.hxx>

#include <stack>

namespace ROOT {
namespace Internal {
namespace RDF {

/// This is an helper class to allow to pick a slot resorting to a map
/// indexed by thread ids.
/// WARNING: this class does not work as a regular stack. The size is
/// fixed at construction time and no blocking is foreseen.
class RSlotStack {
private:
   const unsigned int fSize;
   std::stack<unsigned int> fStack;
   ROOT::TSpinMutex fMutex;

public:
   RSlotStack() = delete;
   RSlotStack(unsigned int size);
   void ReturnSlot(unsigned int slotNumber);
   unsigned int GetSlot();
};
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
