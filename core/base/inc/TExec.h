// @(#)root/base:$Id$
// Author: Rene Brun   29/12/99

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TExec
#define ROOT_TExec


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TExec                                                                //
//                                                                      //
// A TExec object can execute a CLING command.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TExec : public TNamed {

public:

   TExec();
   TExec(const char *name, const char *command);
   TExec(const TExec &text);
   virtual ~TExec();
   virtual void     Exec(const char *command = "");
   void             Paint(Option_t *option="") override;
   void             SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void     SetAction(const char *action) { SetTitle(action); }

   ClassDefOverride(TExec,1);  //To execute a CLING command
};

#endif
