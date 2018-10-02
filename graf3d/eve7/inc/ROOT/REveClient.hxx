// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveClient
#define ROOT_REveClient

#include <memory>

namespace ROOT {
namespace Experimental {

class TWebWindow;
class REveScene;

class REveClient {

   friend class REveScene;

   unsigned fId{0};
   std::shared_ptr<TWebWindow> fWebWindow;

public:
   REveClient() = default;
   REveClient(unsigned int cId, std::shared_ptr<TWebWindow> &win) : fId(cId), fWebWindow(win) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
