// @(#)root/rootd:$Name:  $:$Id: rootd.h,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Fons Rademakers   11/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rootd
#define ROOT_rootd


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rootd                                                                //
//                                                                      //
// This header file contains public definitions and declarations        //
// used by rootd.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Rootd error codes. In case of change update also strings in TNetFile.h.

enum ERootdErrors {
   kErrUndef,
   kErrNoFile,
   kErrBadFile,
   kErrFileExists,
   kErrNoAccess,
   kErrFileOpen,
   kErrFileWriteOpen,
   kErrFileReadOpen,
   kErrNoSpace,
   kErrBadOp,
   kErrBadMess,
   kErrFilePut,
   kErrFileGet,
   kErrNoUser,
   kErrNoAnon,
   kErrBadUser,
   kErrNoHome,
   kErrNoPasswd,
   kErrBadPasswd,
   kErrNoSRP,
   kErrFatal,
   kErrRestartSeek
};

#endif
