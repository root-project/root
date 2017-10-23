// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpWSHandler
#define ROOT_THttpWSHandler

#include "TNamed.h"

#include "TList.h"

class THttpCallArg;
class THttpWSEngine;
class THttpServer;

class THttpWSHandler : public TNamed {

friend class THttpServer;

private:

   THttpWSEngine *FindEngine(UInt_t id) const;

   Bool_t HandleWS(THttpCallArg *arg);

protected:

   TList    fEngines;         ///<!  list of of engines in use, cleaned automatically at the end

   THttpWSHandler(const char *name, const char *title);

public:
   virtual ~THttpWSHandler();

   /// Provides content of default web page for registered web-socket handler
   /// Can be content of HTML page or file name, where content should be taken
   /// For instance, file:/home/user/test.htm or file:$jsrootsys/files/canvas.htm
   /// If not specified, default index.htm page will be shown
   /// Used by the webcanvas
   virtual TString GetDefaultPageContent() { return ""; }

   /// Return kTRUE if websocket with given ID exists
   Bool_t HasWS(UInt_t wsid) const { return FindEngine(wsid) != 0; }

   void CloseWS(UInt_t wsid);

   void SendWS(UInt_t wsid, const void *buf, int len);

   void SendCharStarWS(UInt_t wsid, const char *str);

   virtual Bool_t ProcessWS(THttpCallArg *arg) = 0;

   ClassDef(THttpWSHandler, 0) // abstract class for handling websocket requests
};

#endif
