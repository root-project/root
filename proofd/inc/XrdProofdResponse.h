// @(#)root/proofd:$Name:  $:$Id: XrdProofdResponse.h,v 1.2 2006/03/01 15:46:33 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdResponse
#define ROOT_XrdProofdResponse

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdResponse                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Utility class to handle replies to clients.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>

#include "XrdOuc/XrdOucPthread.hh"
#include "XProofProtocol.h"

class XrdLink;

class XrdProofdResponse
{
 public:
   XrdProofdResponse() { fLink = 0; *fTrsid = '\0';
                         fRespIO[0].iov_base = (caddr_t)&fResp;
                         fRespIO[0].iov_len  = sizeof(fResp); }
   XrdProofdResponse(XrdProofdResponse &rhs) { Set(rhs.fLink);
                                               Set(rhs.fResp.streamid); }
   virtual ~XrdProofdResponse() {}

   XrdProofdResponse &operator =(const XrdProofdResponse &rhs) {
                                               Set(rhs.fLink);
                                               Set((unsigned char *)rhs.fResp.streamid);
                                               return *this; }

   const  char          *ID() { return (const char *)fTrsid;}

   int                   Send(void);
   int                   Send(const char *msg);
   int                   Send(void *data, int dlen);
   int                   Send(struct iovec *, int iovcnt, int iolen=-1);
   int                   Send(XResponseType rcode);
   int                   Send(XResponseType rcode, void *data, int dlen);
   int                   Send(XErrorCode ecode, const char *msg);
   int                   Send(XPErrorCode ecode, const char *msg);
   int                   Send(XResponseType rcode, int info, char *data = 0);
   int                   Send(XResponseType rcode, XProofActionCode acode, int info);
   int                   Send(XResponseType rcode,
                              XProofActionCode acode, void *data, int dlen);
   int                   Send(XResponseType rcode, XProofActionCode acode,
                              kXR_int32 sid, void *data, int dlen);
   int                   Send(kXR_int32 int1, kXR_int32 int2, void *data = 0, int dlen = 0);
   int                   Send(kXR_int32 int1, void *data = 0, int dlen = 0);

   inline void           Set(XrdLink *lp) { fLink = lp;}
   void                  Set(unsigned char *stream);
   void                  Set(unsigned short streamid);

   void                  GetSID(unsigned short &sid);


   // To protect from concurrent use
   XrdOucRecMutex       fMutex;

 private:

   ServerResponseHeader fResp;
   XrdLink             *fLink;
   struct iovec         fRespIO[4];

   char                 fTrsid[8];  // sizeof() does not work here

   static const char   *fgTraceID;
};
#endif
