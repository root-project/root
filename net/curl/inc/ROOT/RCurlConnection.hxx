// @(#)root/net:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCurlConnection
#define ROOT_RCurlConnection

#include <string>

namespace ROOT {
namespace Internal {

/// Encapsulates a curl easy handle and provides an interface to send HTTP HEAD and (multi-)range queries.
class RCurlConnection {
private:
   void *fHandle = nullptr;

public:
   explicit RCurlConnection(const std::string &url);
   ~RCurlConnection();
   RCurlConnection(const RCurlConnection &other) = delete;
   RCurlConnection(RCurlConnection &&other) = default;
   RCurlConnection &operator=(const RCurlConnection &other) = delete;
   RCurlConnection &operator=(RCurlConnection &&other) = default;
};

} // namespace Internal
} // namespace ROOT

#endif
