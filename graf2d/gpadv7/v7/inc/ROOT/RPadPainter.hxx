/// \file ROOT/RPadPainter.hxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev
/// \date 2018-03-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPadPainter
#define ROOT7_RPadPainter

#include "ROOT/RDisplayItem.hxx"

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

class RPadDisplayItem;
class RPadBase;
class RPad;

namespace Internal {

/** \class ROOT::Experimental::Internal::RPadPainter
  Abstract interface for object painting on the pad/canvas.
  */

class RPadPainter {

   friend class ROOT::Experimental::RPad;

protected:

    std::unique_ptr<RPadDisplayItem> fPadDisplayItem; ///<! display items for all drawables in the pad
    std::string   fCurrentDrawableId; ///<! current drawable id

    void PaintDrawables(const RPadBase &pad);

public:

   /// Default constructor
   RPadPainter() = default;

   /// Default destructor.
   virtual ~RPadPainter();

   /// add display item to the canvas
   virtual void AddDisplayItem(std::unique_ptr<RDisplayItem> &&item);
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RPadPainter
