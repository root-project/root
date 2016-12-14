// @(#)root/auth:$Id$
// Author: Gerri Ganis  19/1/2004

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_DaemonUtils
#define ROOT_DaemonUtils


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DaemonUtils                                                          //
//                                                                      //
// This file defines wrappers to client utils calls used by server      //
// authentication daemons.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>

#ifndef ROOT_TSocket
#include "TSocket.h"
#endif
#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif
#ifndef ROOT_NetErrors
#include "NetErrors.h"
#endif
#ifndef ROOT_rpddefs
#include "rpddefs.h"
#endif

#include "rpdp.h"


extern Int_t SrvAuthImpl(TSocket *socket, const char *, const char *,
                         std::string &user, Int_t &meth,
                         Int_t &type, std::string &ctoken, TSeqCollection *);
extern Int_t SrvClupImpl(TSeqCollection *);

typedef void (*ErrorHandler_t)(int level, const char *msg, int size);


namespace ROOT {

// Error handlers prototypes ...
extern ErrorHandler_t gErrSys;
extern ErrorHandler_t gErrFatal;
extern ErrorHandler_t gErr;
void SrvSetSocket(TSocket *socket);

}

#endif
