/// \file ROOT/TPadPainter.hxx
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

#ifndef ROOT7_TPadPainter
#define ROOT7_TPadPainter

#include "ROOT/TDisplayItem.hxx"

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

class TPadDisplayItem;
class TPadDrawable;
class TPadBase;

namespace Internal {

/** \class ROOT::Experimental::Internal::TPadPainter
  Abstract interface for object painting on the pad/canvas.
  */

class TPadPainter {

friend class ROOT::Experimental::TPadDrawable;

protected:

    std::unique_ptr<TPadDisplayItem> fPadDisplayItem; ///<! display items for all drawables in the pad
    std::string   fCurrentDrawableId; ///<! current drawable id

    void PaintDrawables(const TPadBase &pad);

public:

   /// Default constructor
   TPadPainter() = default;

   /// Default destructor.
   virtual ~TPadPainter();

   /// add display item to the canvas
   virtual void AddDisplayItem(std::unique_ptr<TDisplayItem> &&item);
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TPadPainter
