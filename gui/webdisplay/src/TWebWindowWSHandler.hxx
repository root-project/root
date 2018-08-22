/// \file ROOT/TWebWindowWSHandler.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2018-08-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TWebWindowWSHandler
#define ROOT7_TWebWindowWSHandler

#include "THttpWSHandler.h"

#include <ROOT/TWebWindow.hxx>

namespace ROOT {
namespace Experimental {

/// just wrapper to deliver websockets call-backs to the TWebWindow class

class TWebWindowWSHandler : public THttpWSHandler {
public:
   TWebWindow &fWindow; ///<! window reference

   /// constructor
   TWebWindowWSHandler(TWebWindow &wind, const char *name)
      : THttpWSHandler(name, "TWebWindow websockets handler", kFALSE), fWindow(wind)
   {
   }

   virtual ~TWebWindowWSHandler()
   {
   }

   /// returns content of default web-page
   /// THttpWSHandler interface
   virtual TString GetDefaultPageContent() override { return IsDisabled() ? "" : fWindow.fDefaultPage.c_str(); }

   /// Process websocket request - called from THttpServer thread
   /// THttpWSHandler interface
   virtual Bool_t ProcessWS(THttpCallArg *arg) override { return arg && !IsDisabled() ? fWindow.ProcessWS(*arg) : kFALSE; }

   /// Allow processing of WS actions in arbitrary thread
   virtual Bool_t AllowMTProcess() const { return fWindow.fProcessMT; }

   /// Allows usage of multithreading in send operations
   virtual Bool_t AllowMTSend() const { return fWindow.fSendMT; }

   /// React on completion of multithreaded send operaiotn
   virtual void CompleteWSSend(UInt_t wsid) { if (!IsDisabled()) fWindow.CompleteWSSend(wsid); }
};

} // namespace Experimental
} // namespace ROOT

#endif
