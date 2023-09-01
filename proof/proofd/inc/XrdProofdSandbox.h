// @(#)root/proofd:$Id$
// Author: G. Ganis  Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdSandbox
#define ROOT_XrdProofdSandbox

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdSandbox                                                     //
//                                                                      //
// Authors: G. Ganis, CERN, 2008                                        //
//                                                                      //
// Create and manage a Unix sandbox.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>

#include "XrdProofdAux.h"
#include "XrdOuc/XrdOucString.hh"

class XrdProofdSandbox {

private:

   bool                fChangeOwn;
   XrdOucString        fDir;
   XrdOucString        fErrMsg;
   bool                fValid;
   XrdProofUI          fUI;


   static int          fgMaxOldSessions;
   static XrdOucString fgWorkdir;
   static XrdProofUI   fgUI;

public:
   XrdProofdSandbox(XrdProofUI ui, bool full, bool changeown);

   const char *Dir() const { return fDir.c_str(); }

   int         GuessTag(XrdOucString &tag, int ridx = 1);

   const char *ErrMsg() const { return fErrMsg.c_str(); }
   bool        IsValid() const { return fValid; }

   // Active sessions file management
   int         AddSession(const char *tag);
   int         RemoveSession(const char *tag);

   //Parse the session dirs
   int         GetSessionDirs(int opt, std::list<XrdOucString *> *sdirs,
                              XrdOucString *tag = 0);
   int         TrimSessionDirs();

   // The manager uses these to set configurable members
   static void SetMaxOldSessions(int mxses) { fgMaxOldSessions = mxses; }
   static void SetWorkdir(const char *wdir) { fgWorkdir = wdir; }
};
#endif
