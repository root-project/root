// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXProtocol
#define ROOT_TXProtocol

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXProtocol.h                                                         //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Utility functions prototypes for client-to-server                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef __XPROTOCOL_H
#include "XProtocol/XProtocol.hh"
#endif

namespace ROOT {

   void clientMarshall(ClientRequest* str);
   void clientUnmarshall(struct ServerResponseHeader* str);
   void ServerResponseHeader2NetFmt(struct ServerResponseHeader *srh);
   void ServerInitHandShake2HostFmt(struct ServerInitHandShake *srh);
   bool isRedir(struct ServerResponseHeader *ServerResponse);
   char *convertRequestIdToChar(kXR_int16 requestid);
   void PutFilehandleInRequest(ClientRequest* str, char *fHandle);
   char *convertRespStatusToChar(kXR_int16 status);
   void smartPrintClientHeader(ClientRequest* hdr);
   void smartPrintServerHeader(struct ServerResponseHeader* hdr);

} // namespace ROOT

#endif
