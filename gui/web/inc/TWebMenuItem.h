/// \file ROOT/TWebMenuItem.h
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-06-29

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TWebMenuItem
#define ROOT7_TWebMenuItem

#include <string>
#include <vector>

/** \class TMenuItem
  Class contains info for producing menu item on the JS side.
  */

class TWebMenuItem {
protected:
   std::string  fName;     //  name of the menu item
   std::string  fTitle;    //  title of menu item
   int          fChecked;  // -1 not exists, 0 - off, 1 - on
   std::string  fExec;     // execute when item is activated
public:

   TWebMenuItem() : fName(), fTitle(), fChecked(-1), fExec() {}
   TWebMenuItem(const std::string &name, const std::string &title) : fName(name), fTitle(title), fChecked(-1), fExec() {}
   TWebMenuItem(const TWebMenuItem &rhs) : fName(rhs.fName), fTitle(rhs.fTitle), fChecked(rhs.fChecked), fExec(rhs.fExec) {}
   virtual ~TWebMenuItem() {}

   void SetChecked(bool on = true) { fChecked = on ? 1 : 0; }
   void SetExec(const std::string &exec) { fExec = exec; }
};


#endif
