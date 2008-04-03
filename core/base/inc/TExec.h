// @(#)root/base:$Id$
// Author: Rene Brun   29/12/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
// A TExec object can execute a CINT command.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif


class TExec : public TNamed {

public:

   TExec();
   TExec(const char *name, const char *command);
   TExec(const TExec &text);
   virtual ~TExec();
   virtual void     Exec(const char *command="");
   virtual void     Paint(Option_t *option="");
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetAction(const char *action) {SetTitle(action);}
   
   ClassDef(TExec,1);  //To execute a CINT command
};

#endif

