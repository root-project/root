// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpWSEngine
#define ROOT_THttpWSEngine

#include "Rtypes.h"

#include "THttpCallArg.h"

class THttpWSHandler;

class THttpWSEngine {

private:
   friend class THttpWSHandler;

   bool fMTSend{false};     ///<  true when multithreaded send operation is active

protected:
   THttpWSEngine() = default;

   /// Indicate if engine support send operation from different threads
   virtual Bool_t SupportMT() const { return kFALSE; }

public:
   virtual ~THttpWSEngine() {}

   virtual UInt_t GetId() const = 0;

   virtual void ClearHandle() = 0;

   virtual void Send(const void *buf, int len) = 0;

   virtual void SendHeader(const char *hdr, const void *buf, int len) = 0;

   virtual void SendCharStar(const char *str);

   virtual Bool_t PreviewData(std::shared_ptr<THttpCallArg> &arg);

   virtual void PostProcess(std::shared_ptr<THttpCallArg> &arg);
};

#endif
