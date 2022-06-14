// Author: Enrico Guiraud, 2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RNEWSAMPLENOTIFIER
#define ROOT_RDF_RNEWSAMPLENOTIFIER

#include <TNotifyLink.h>

#include <memory>

namespace ROOT {
namespace Internal {
namespace RDF {

struct RNewSampleFlag {
   bool fFlag = false;

public:
   void SetFlag() { fFlag = true; }
   void UnsetFlag() { fFlag = false; }
   bool CheckFlag() const { return fFlag; }
   bool Notify()
   {
      SetFlag();
      return true;
   }
};

class RNewSampleNotifier {
   // TNotifyLink and RNewSampleFlags per processing slot
   std::vector<std::unique_ptr<TNotifyLink<RNewSampleFlag>>> fNotifyLink;
   std::vector<RNewSampleFlag> fFlags;

public:
   RNewSampleNotifier(unsigned int nSlots) : fNotifyLink(nSlots), fFlags(nSlots) {}
   bool CheckFlag(unsigned int slot) const { return fFlags[slot].CheckFlag(); }
   void SetFlag(unsigned int slot) { fFlags[slot].SetFlag(); }
   void UnsetFlag(unsigned int slot) { fFlags[slot].UnsetFlag(); }
   TNotifyLink<RNewSampleFlag> &GetChainNotifyLink(unsigned int slot)
   {
      if (fNotifyLink[slot] == nullptr)
         fNotifyLink[slot] = std::make_unique<TNotifyLink<RNewSampleFlag>>(&fFlags[slot]);
      return *fNotifyLink[slot];
   }
};
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_RDATABLOCKNOTIFIER
