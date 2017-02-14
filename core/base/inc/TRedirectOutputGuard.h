// @(#)root/base:$Id$
// Author: G. Ganis   10/10/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRedirectOutputGuard
#define ROOT_TRedirectOutputGuard

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRedirectOutputGuard                                                 //
//                                                                      //
// This class provides output redirection to a file in a guaranteed     //
// exception safe way. Use like this:                                   //
// {                                                                    //
//    TRedirectOutputGuard guard(filelog, mode);                        //
//    ... // do something                                               //
// }                                                                    //
// when guard goes out of scope output is automatically redirected to   //
// the standard units in the TRedirectOutputGuard destructor.           //
// The exception mechanism takes care of calling the dtors              //
// of local objects so it is exception safe.                            //
// The 'mode' options follow the fopen write modes convention; default  //
// is "a".                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystem.h"

class TRedirectOutputGuard {

 public:
   TRedirectOutputGuard(const char *fout, const char *mode = "a")
                                   { gSystem->RedirectOutput(fout, mode); }
   virtual ~TRedirectOutputGuard() { gSystem->RedirectOutput(0); }

   ClassDef(TRedirectOutputGuard,0)  // Exception safe output redirection
};

#endif
