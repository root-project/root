// $Id$
// Author: Sergey Linev   8/01/2018

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpLongPollEngine
#define ROOT_THttpLongPollEngine

#include "THttpWSEngine.h"

#include <string>
#include <queue>

class THttpServer;

class THttpLongPollEngine : public THttpWSEngine {
   friend class THttpServer;

protected:
   struct QueueItem {
      bool fBinary{false};     ///<! is binary data
      std::string fData;       ///<! text or binary data
      std::string fHdr;        ///<! optional header for raw data
      QueueItem(bool bin, std::string &&data, const std::string &hdr = "") : fBinary(bin), fData(data), fHdr(hdr) {}
   };

   bool fRaw{false};                    ///!< if true, only content can be used for data transfer
   std::shared_ptr<THttpCallArg> fPoll; ///!< hold polling request, which can be immediately used for the next sending
   std::queue<QueueItem> fQueue;        ///!< entries submitted to client
   static const std::string gLongPollNope;    ///!< default reply on the longpoll request

   std::string MakeBuffer(const void *buf, int len, const char *hdr = nullptr);

   virtual Bool_t CanSendDirectly() { return fPoll; }

public:
   THttpLongPollEngine(bool raw = false);

   virtual UInt_t GetId() const;

   virtual void ClearHandle();

   virtual void Send(const void *buf, int len);

   virtual void SendCharStar(const char *buf);

   virtual void SendHeader(const char *hdr, const void *buf, int len);

   virtual Bool_t PreviewData(std::shared_ptr<THttpCallArg> &arg);

   virtual void PostProcess(std::shared_ptr<THttpCallArg> &arg);
};

#endif
