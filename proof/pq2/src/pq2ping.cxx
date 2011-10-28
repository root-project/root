// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// ************************************************************************* //
// *                                                                       * //
// *                           p q 2 p i n g                               * //
// *                                                                       * //
// * This file implements the daemon checking functions used by pq2main    * //
// *                                                                       * //
// ************************************************************************* //

#include <stdio.h>
#include <stdlib.h>

#include "Bytes.h"
#include "TSocket.h"
#include "TString.h"
#include "TSystem.h"
#include "TUrl.h"

#include "redirguard.h"
#include "pq2ping.h"

// Auxilliary structures for Xrootd/Xproofd pinging ...
// The client request
typedef struct {
   int first;
   int second;
   int third;
   int fourth;
   int fifth;
} clnt_HS_t;
// The body received after the first handshake's header
typedef struct {
   int msglen;
   int protover;
   int msgval;
} srv_HS_t;

// Global variables used by other PQ2 components
TUrl    gUrl;
Bool_t  gIsProof = kFALSE;

// Global variables defined by other PQ2 components
extern Int_t gverbose;

//_______________________________________________________________________________________
Int_t checkUrl(const char *url, const char *flog, bool def_proof)
{
   // Check if something is running at gUrl
   // Return
   //        0 if OK and data server
   //        1 if OK and PROOF server
   //       -1 if nothing valid is available

   gIsProof = kFALSE;
   gUrl.SetUrl(url);
   TString protocol(gUrl.GetProtocol());
   if (protocol == "root" || protocol == "xroot") {
      // Check Xrootd
      if (pingXrootdAt() != 0) {
         Printf("checkUrl: specified URL does not identifies a running (x)rootd server: %s", url);
         return -1;
      }
   } else if (protocol == "proof") {
      // Check PROOF
      if (pingXproofdAt() != 0) {
         Printf("checkUrl: specified URL does not identifies a running PROOF master: %s", url);
         return -1;
      }
      gIsProof = kTRUE;
      // Always force a new session (do not attach)
      gUrl.SetOptions("N");
   } else {
      Int_t rc = -1;
      if (def_proof) {
         // Check first PROOF
         {  redirguard rog(flog, "a", 0);
            if ((rc = pingXproofdAt()) == 0)
               gIsProof = kTRUE;
         }
         if (rc != 0) {
            // Check also a generic data server
            if (pingServerAt() != 0) {
               Printf("checkUrl: specified URL does not identifies a valid PROOF or data server: %s", url);
               return -1;
            }
         }
      } else {
         // Check first generic data server
         {  redirguard rog(flog, "a", 0);
            rc = pingServerAt();
         }
         if (rc != 0) {
            // Check also PROOF
            if (pingXproofdAt() != 0) {
               Printf("checkUrl: specified URL does not identifies a valid data or PROOF server: %s", url);
               return -1;
            }
            gIsProof = kTRUE;
         }
      }
   }
   if (gverbose > 0) Printf("checkUrl: %s service", (gIsProof ? "PROOF" : "Data"));

   // Done (if we are here the test was successful)
   return ((gIsProof) ? 1 : 0);
}

//_______________________________________________________________________________________
Int_t pingXrootdAt()
{
   // Check if a XrdXrootd service is running on 'port' at 'host'
   // Return
   //        0 if OK
   //       -1 if nothing is listening on the port (connection cannot be open)
   //        1 if something is listening but not XROOTD

   Int_t port = gUrl.GetPort();
   const char *host = gUrl.GetHost();

   // Open the connection
   TSocket s(host, port);
   if (!(s.IsValid())) {
      if (gDebug > 0)
         Printf("pingXrootdAt: could not open connection to %s:%d", host, port);
      return -1;
   }
   // Send the first bytes
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.fourth = host2net((int)4);
   initHS.fifth  = host2net((int)2012);
   int len = sizeof(initHS);
   s.SendRaw(&initHS, len);
   // Read first server response
   int type;
   len = sizeof(type);
   int readCount = s.RecvRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (gDebug > 0)
         Printf("pingXrootdAt: 1st: wrong number of bytes read: %d (expected: %d)",
                readCount, len);
      return 1;
   }
   // to host byte order
   type = net2host(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      readCount = s.RecvRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         if (gDebug > 0)
            Printf("pingXrootdAt: 2nd: wrong number of bytes read: %d (expected: %d)",
                   readCount, len);
         return 1;
      }

   } else if (type == 8) {
      // Standard proofd
      if (gDebug > 0)
         Printf("pingXrootdAt: server is ROOTD");
      return 1;
   } else {
      // We don't know the server type
      if (gDebug > 0)
         Printf("pingXrootdAt: unknown server type: %d", type);
      return 1;
   }
   // Done
   return 0;
}

//_______________________________________________________________________________________
Int_t pingXproofdAt()
{
   // Check if a XrdProofd service is running on 'port' at 'host'
   // Return
   //        0 if OK
   //       -1 if nothing is listening on the port (connection cannot be open)
   //        1 if something is listening but not XPROOFD

   Int_t port = gUrl.GetPort();
   const char *host = gUrl.GetHost();

   // Open the connection
   TSocket s(host, port);
   if (!(s.IsValid())) {
      if (gDebug > 0)
         Printf("pingXproofdAt: could not open connection to %s:%d", host, port);
      return -1;
   }
   // Send the first bytes
   int writeCount = -1;
   clnt_HS_t initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.third  = (int)host2net((int)1);
   int len = sizeof(initHS);
   writeCount = s.SendRaw(&initHS, len);
   if (writeCount != len) {
      if (gDebug > 0)
         Printf("pingXproofdAt: 1st: wrong number of bytes sent: %d (expected: %d)",
                writeCount, len);
      return 1;
   }
   // These 8 bytes are need by 'proofd' and discarded by XPD
   int dum[2];
   dum[0] = (int)host2net((int)4);
   dum[1] = (int)host2net((int)2012);
   writeCount = s.SendRaw(&dum[0], sizeof(dum));
   if (writeCount != sizeof(dum)) {
      if (gDebug > 0)
         Printf("pingXproofdAt: 2nd: wrong number of bytes sent: %d (expected: %d)",
                writeCount, (int) sizeof(dum));
      return 1;
   }
   // Read first server response
   int type;
   len = sizeof(type);
   int readCount = s.RecvRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (gDebug > 0)
         Printf("pingXproofdAt: 1st: wrong number of bytes read: %d (expected: %d)",
                readCount, len);
      return 1;
   }
   // to host byte order
   type = net2host(type);
   // Check if the server is the eXtended proofd
   if (type == 0) {
      srv_HS_t xbody;
      len = sizeof(xbody);
      readCount = s.RecvRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         if (gDebug > 0)
            Printf("pingXproofdAt: 2nd: wrong number of bytes read: %d (expected: %d)",
                   readCount, len);
         return 1;
      }
      xbody.protover = net2host(xbody.protover);
      xbody.msgval = net2host(xbody.msglen);
      xbody.msglen = net2host(xbody.msgval);

   } else if (type == 8) {
      // Standard proofd
      if (gDebug > 0)
         Printf("pingXproofdAt: server is PROOFD");
      return 1;
   } else {
      // We don't know the server type
      if (gDebug > 0)
         Printf("pingXproofdAt: unknown server type: %d", type);
      return 1;
   }
   // Done
   return 0;
}

//_______________________________________________________________________________________
Int_t pingServerAt()
{
   // Check if service is running at 'url'
   // Return
   //        0 if OK
   //       -1 if nothing is listening at the URL
   //        1 if not a directory

   Int_t rc = -1;
   FileStat_t st;
   if (gSystem->GetPathInfo(gUrl.GetUrl(), st) == 0) {
      rc = 1;
      if (R_ISDIR(st.fMode)) rc = 0;
   }

   // Done
   return rc;
}
