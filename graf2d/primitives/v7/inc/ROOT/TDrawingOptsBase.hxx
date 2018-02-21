/// \file ROOT/TDrawingOptionsBase.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawingOptsBase
#define ROOT7_TDrawingOptsBase

#include <functional>
#include <string>

namespace ROOT {
namespace Experimental {
class TDrawingAttrBase;

class TDrawingOptsBase {
   /// Attribute style class of these options.
   std::string fStyleClass;

public:
   /// Initialize the options with a (possibly empty) style class.
   TDrawingOptsBase(const std::string &styleClass = {}): fStyleClass(styleClass) {}

   using VisitFunc_t = std::function<void(TDrawingAttrBase&)>;
   virtual ~TDrawingOptsBase();

   /// Get the attribute style class of these options.
   const std::string &GetStyleClass() const { return fStyleClass; }

   /// Invoke func with each attribute as argument.
   void VisitAttributes(const VisitFunc_t &func);

   /// Synchronize all shared attributes into their local copy.
   void Snapshot();
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TDrawingOptsBase
