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

class THttpCallArg;

class THttpWSHandler : public TNamed {

protected:
   THttpWSHandler(const char *name, const char *title);

public:
   virtual ~THttpWSHandler();

   /// Provides content of default web page for registered web-socket handler
   /// Can be content of HTML page or file name, where content should be taken
   /// For instance, file:/home/user/test.htm or file:$jsrootsys/files/canvas.htm
   /// If not specified, default index.htm page will be shown
   /// Used by the webcanvas
   virtual TString GetDefaultPageContent() { return ""; }

   virtual Bool_t ProcessWS(THttpCallArg *arg) = 0;

   ClassDef(THttpWSHandler, 0) // abstract class for handling websocket requests
};

#endif
