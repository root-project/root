// @(#)root/proofd:$Name:  $:$Id: XrdProofPhyConn.h,v 1.2 2006/03/01 15:46:33 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofPhyConn
#define ROOT_XrdProofPhyConn


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofPhyConn                                                      //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
//  XrdProofConn implementation using a simple phycical connection      //
//  (Unix or Tcp)                                                       //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_XrdProofConn
#include "XrdProofConn.h"
#endif

class XrdProofPhyConn  : public XrdProofConn {

friend class TXSocket;
friend class TXUnixSocket;

private:

   bool        fTcp;   // TRUE for TCP sockets

   int         Connect();
   bool        GetAccessToSrv();
   bool        Init(const char *url);
   void        SetAsync(XrdClientAbsUnsolMsgHandler *uh);

public:
   XrdProofPhyConn(const char *url, int psid = -1, char ver = -1,
                   XrdClientAbsUnsolMsgHandler *uh = 0, bool tcp = 0);
   virtual ~XrdProofPhyConn() { Close(); }

   void        Close(const char *opt = "");

   // Send, Recv interfaces
   int         ReadRaw(void *buf, int len);
   XrdClientMessage *ReadMsg();
   int         WriteRaw(const void *buf, int len);
};

#endif
