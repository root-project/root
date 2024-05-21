// @(#)root/proofd:$Id$
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

#include "XrdProofConn.h"

class XrdProofPhyConn  : public XrdProofConn {

friend class TXSocket;
friend class TXUnixSocket;

private:

   bool        fTcp;   // TRUE for TCP sockets

   void        Connect(int fd = -1);
   int         TryConnect(int fd = -1);
   bool        GetAccessToSrv(XrdClientPhyConnection * = 0);
   bool        Init(const char *url, int fd = -1);

public:
   XrdProofPhyConn(const char *url, int psid = -1, char ver = -1,
                   XrdClientAbsUnsolMsgHandler *uh = 0, bool tcp = 0, int fd = -1);
   virtual ~XrdProofPhyConn() { Close(); }

   void        Close(const char *opt = "");

   // Send, Recv interfaces
   int         ReadRaw(void *buf, int len, XrdClientPhyConnection * = 0);
   XrdClientMessage *ReadMsg();
   void        SetAsync(XrdClientAbsUnsolMsgHandler *uh, XrdProofConnSender_t = 0, void * = 0);
   int         WriteRaw(const void *buf, int len, XrdClientPhyConnection * = 0);
};

#endif
