// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov 5/12/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TMacOSXSystem
#define ROOT_TMacOSXSystem

#include "TUnixSystem.h"

#include <memory>

////////////////////////////////////////////////////////////////////
//                                                                //
// Event loop with MacOS X and Cocoa is different                 //
// from TUnixSystem. The difference is mainly                     //
// in DispatchOneEvent, AddFileHandler and RemoveFileHandler.     //
//                                                                //
////////////////////////////////////////////////////////////////////

namespace ROOT {
namespace MacOSX {
namespace Details {

//'Private' pimpl class to hide Apple's specific things from CINT.
class MacOSXSystem;

}
}
}

class TMacOSXSystem : public TUnixSystem {
   friend class TGCocoa;
public:
   TMacOSXSystem();
   ~TMacOSXSystem();

   void DispatchOneEvent(Bool_t pendingOnly);
   bool CocoaInitialized() const;

private:
   void InitializeCocoa();

   bool ProcessPendingEvents();
   void WaitEvents(Long_t nextto);

   void AddFileHandler(TFileHandler *fh);
   TFileHandler *RemoveFileHandler(TFileHandler *fh);

   void ProcessApplicationDefinedEvent(void *event);

   std::unique_ptr<ROOT::MacOSX::Details::MacOSXSystem> fPimpl; //!
   bool fCocoaInitialized; //!
   bool fFirstDispatch; //!

   TMacOSXSystem(const TMacOSXSystem &rhs);
   TMacOSXSystem &operator = (const TMacOSXSystem &rhs);

   ClassDef(TMacOSXSystem, 0);//TSystem for Mac OSX.
};

#endif
