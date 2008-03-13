// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XProofProtUtils
#define ROOT_XProofProtUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XProofProtUtils.h                                                    //
//                                                                      //
// Authors: G. Ganis, CERN 2005                                         //
//                                                                      //
// Utility functions prototypes for client-to-server                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace XPD {

   int clientMarshall(XPClientRequest* str);
   void clientUnmarshall(struct ServerResponseHeader* str);
   void ServerResponseHeader2NetFmt(struct ServerResponseHeader *srh);
   void ServerInitHandShake2HostFmt(struct ServerInitHandShake *srh);
   char *convertRequestIdToChar(kXR_int16 requestid);
   char *convertRespStatusToChar(kXR_int16 status);
   void smartPrintClientHeader(XPClientRequest* hdr);
   void smartPrintServerHeader(struct ServerResponseHeader* hdr);

} // namespace XPD

#endif
