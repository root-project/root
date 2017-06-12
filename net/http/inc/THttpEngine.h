// $Id$
// Author: Sergey Linev   21/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpEngine
#define ROOT_THttpEngine

#include "TNamed.h"

class THttpServer;
class THttpCallArg;

class THttpEngine : public TNamed {
protected:
   friend class THttpServer;

   THttpServer *fServer; ///<! object server

   THttpEngine(const char *name, const char *title);

   void SetServer(THttpServer *serv) { fServer = serv; }

   /** Method regularly called in main ROOT context */
   virtual void Process() {}

public:
   virtual ~THttpEngine();

   /** Method to create all components of engine. Called once from by the server */
   virtual Bool_t Create(const char *) { return kFALSE; }

   /** Returns pointer to THttpServer associated with engine */
   THttpServer *GetServer() const { return fServer; }

   ClassDef(THttpEngine, 0) // abstract class which should provide http-based protocol for server
};

// ====================================================================

class THttpWSEngine : public TNamed {

protected:
   THttpWSEngine(const char *name, const char *title);

public:
   virtual ~THttpWSEngine();

   virtual UInt_t GetId() const = 0;

   virtual void ClearHandle() = 0;

   virtual void Send(const void *buf, int len) = 0;

   virtual void SendCharStar(const char *str);

   virtual Bool_t PreviewData(THttpCallArg *) { return kFALSE; }

   ClassDef(THttpWSEngine, 0) // abstract class for working with WebSockets-like protocol
};

// ====================================================================

class THttpWSHandler : public TNamed {

protected:
   THttpWSHandler(const char *name, const char *title);

public:
   virtual ~THttpWSHandler();

   virtual Bool_t ProcessWS(THttpCallArg *arg) = 0;

   ClassDef(THttpWSHandler, 0) // abstract class for handling websocket requests
};



#endif
