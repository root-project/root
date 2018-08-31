// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSLOTSTACK
#define ROOT_RSLOTSTACK

#include "ROOT/TRWSpinLock.hxx"

#include <map>
#include <thread>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

/// This is an helper class to allow to pick a slot resorting to a map
/// indexed by thread ids.
/// WARNING: this class does not work as a regular stack. The size is
/// fixed at construction time and no blocking is foreseen.
class RSlotStack {
private:
   unsigned int &GetCount();
   unsigned int &GetIndex();
   unsigned int fCursor;
   std::vector<unsigned int> fBuf;
   ROOT::TRWSpinLock fRWLock;

public:
   RSlotStack() = delete;
   RSlotStack(unsigned int size);
   void ReturnSlot(unsigned int slotNumber);
   unsigned int GetSlot();
   std::map<std::thread::id, unsigned int> fCountMap;
   std::map<std::thread::id, unsigned int> fIndexMap;
};
} // ns RDF
} // ns Internal
} // ns ROOT

#endif
