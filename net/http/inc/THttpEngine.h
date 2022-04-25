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

class THttpEngine : public TNamed {
protected:
   friend class THttpServer;

   THttpServer *fServer{nullptr}; ///<! object server

   THttpEngine(const char *name, const char *title);

   void SetServer(THttpServer *serv) { fServer = serv; }

   /** Method called when server want to be terminated */
   virtual void Terminate() {}

   /** Method regularly called in main ROOT context */
   virtual void Process() {}

public:
   /** Method to create all components of engine. Called once from by the server */
   virtual Bool_t Create(const char *) { return kFALSE; }

   /** Returns pointer to THttpServer associated with engine */
   THttpServer *GetServer() const { return fServer; }

   ClassDefOverride(THttpEngine, 0) // abstract class which should provide http-based protocol for server
};

#endif
