/// \file TDirectory.h
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice, it might do evil. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawable
#define ROOT7_TDrawable

namespace ROOT {
namespace Internal {

/** \class TDrawable
  Base class for drawable entities: objects that can be painted on a `TPad`.
  */

class TDrawable {
public:
  virtual ~TDrawable();

  /// Paint the object
  virtual void Paint() = 0;
};

} // namespace Internal
} // namespace ROOT

#endif
