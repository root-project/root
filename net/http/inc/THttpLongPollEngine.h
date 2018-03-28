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
#include <list>

class THttpServer;

class THttpLongPollEngine : public THttpWSEngine {
   friend class THttpServer;

protected:
   struct QueueItem {
      const void *fBuffer{nullptr}; ///<! raw memory with ownership
      int fLength{0};               ///<! size of raw memory
      std::string fMessage;         ///<! plain text message
      QueueItem(const std::string &msg) : fMessage(msg) {}
      QueueItem(const void *buf, int len, const std::string &msg = "") : fBuffer(buf), fLength(len), fMessage(msg) {}
      ~QueueItem();
   };

   bool fRaw{false};                 ///!< if true, only content can be used for data transfer
   THttpCallArg *fPoll{nullptr};     ///!< polling request, which can be used for the next sending
   std::list<QueueItem> fQueue;      ///!< entries submitted to client
   static const char *gLongPollNope; ///!< default reply on the longpoll request

   void *MakeBuffer(const void *buf, int &len, const char *hdr = nullptr);

public:
   THttpLongPollEngine(bool raw = false) : THttpWSEngine(), fRaw(raw) {}

   virtual UInt_t GetId() const;

   virtual void ClearHandle();

   virtual void Send(const void *buf, int len);

   virtual void SendCharStar(const char *buf);

   virtual void SendHeader(const char *hdr, const void *buf, int len);

   virtual Bool_t PreviewData(THttpCallArg &arg);

   virtual void PostProcess(THttpCallArg &arg);
};

#endif
