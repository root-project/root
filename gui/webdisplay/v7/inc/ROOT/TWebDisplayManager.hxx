/// \file ROOT/TWebDisplayManager.hxx
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

#ifndef ROOT7_TWebDisplayManager
#define ROOT7_TWebDisplayManager

#include <memory>

class THttpServer;

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TWebDisplayManager
  Central handle to open web-based windows like Canvas or FitPanel.
  */

class TWebDisplayManager {
private:
   THttpServer    *fServer{0};   ///<!  central communication with the all used displays

public:
   /// Create a temporary TCanvas
   TWebDisplayManager() = default;

   ~TWebDisplayManager() {}

   /// Returns central instance, which used by standard ROOT widgets like Canvas or FitPanel
   static const std::shared_ptr<TWebDisplayManager> &Get();

   /// Create custom instance with independent communication channel(s)
   // static std::shared_ptr<TWebDisplayManager> Create();

};

} // namespace Experimental
} // namespace ROOT

#endif
