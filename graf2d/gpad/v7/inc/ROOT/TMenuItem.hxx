/// \file ROOT/TMenuItem.h
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TMenuItem
#define ROOT7_TMenuItem

#include <string>

namespace ROOT {
namespace Experimental {
namespace Detail {

/** \class TMenuItem
  Class contains info for producing menu item on the JS side.
  */

class TMenuItem {
protected:

   enum ECheckedState {
      kNotDefined = -1,
      kOff = 0,
      kOn = 1
   };

   std::string   fName;          ///<  name of the menu item
   std::string   fTitle;         ///<  title of menu item
   ECheckedState fChecked = kNotDefined;  ///< -1 not exists, 0 - off, 1 - on
   std::string   fExec;          ///< execute when item is activated
public:

   TMenuItem() = default;
   TMenuItem(const std::string &name, const std::string &title) : fName(name), fTitle(title), fChecked(kNotDefined), fExec() {}
   TMenuItem(const TMenuItem &rhs) : fName(rhs.fName), fTitle(rhs.fTitle), fChecked(rhs.fChecked), fExec(rhs.fExec) {}
   virtual ~TMenuItem() {}

   void SetChecked(bool on = true) { fChecked = on ? kOff : kOn; }
   void SetExec(const std::string &exec) { fExec = exec; }

   std::string GetName() const { return fName; }
   std::string GetExec() const { return fExec; }
};


} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
