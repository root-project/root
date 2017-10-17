/// \file ROOT/TWebDisplay.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TWebDisplay
#define ROOT7_TWebDisplay

#include <memory>
#include <list>

class THttpCallArg;
class THttpWSEngine;

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TWebDisplayManager
  Central handle to open web-based windows like Canvas or FitPanel.
  */

class TWebDisplayManager;
class TDisplayWSHandler;

class TWebDisplay {

friend class TWebDisplayManager;
friend class TDisplayWSHandler;
private:

   struct WebConn {
      THttpWSEngine *fHandle; ///<! websocket handle
      WebConn() : fHandle(0) {}
   };


   std::shared_ptr<TWebDisplayManager>  fMgr{};     ///<!  display manager
   unsigned                             fId{0};     ///<!  unique identifier
   TDisplayWSHandler               *fWSHandler{nullptr};  ///<!  specialize websocket handler for all incoming connections
   std::list<WebConn>                 fConn{};     ///<! list of all accepted connections

   void SetId(unsigned id) { fId = id; }

   bool ProcessWS(THttpCallArg *arg);

public:
   TWebDisplay() = default;

   ~TWebDisplay();

   unsigned GetId() const { return fId; }

};

} // namespace Experimental
} // namespace ROOT

#endif
