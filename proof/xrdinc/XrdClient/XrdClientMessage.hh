#ifndef XRC_MESSAGE_H
#define XRC_MESSAGE_H
/******************************************************************************/
/*                                                                            */
/*                 X r d C l i e n t M e s s a g e . h h                      */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/*  Alvise Dorigo, Fabrizio Furano                                            */
/*          INFN Padova, 2003                                                 */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A message coming from a physical connection. I.e. a server response  //
//  or some kind of error                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XProtocol/XProtocol.hh"
#include "XrdSys/XrdSysPthread.hh"

#ifndef WIN32
#include <netinet/in.h>
#endif

class XrdClientPhyConnection;

class XrdClientMessage {

private:
   bool           fAllocated;
   void           *fData;
   bool           fMarshalled;
   short          fStatusCode;
   XrdSysRecMutex fMultireadMutex;

public:

   static kXR_unt16       CharStreamid2Int(kXR_char *charstreamid);
   static void            Int2CharStreamid(kXR_char *charstreamid, short intstreamid);

   enum EXrdMSCStatus {             // Some status codes useful
      kXrdMSC_ok               = 0,
      kXrdMSC_readerr          = 1,
      kXrdMSC_writeerr         = 2,
      kXrdMSC_timeout          = 3
   };

   ServerResponseHeader fHdr;

   XrdClientMessage(ServerResponseHeader header);
   XrdClientMessage();

   ~XrdClientMessage();

   bool               CreateData();

   inline int         DataLen() { return fHdr.dlen; }

   void              *DonateData();
   inline void       *GetData() {return fData;}
   inline int         GetStatusCode() { return fStatusCode; }

   inline int         HeaderStatus() { return fHdr.status; }

   inline kXR_unt16   HeaderSID() { return CharStreamid2Int(fHdr.streamid); }

   bool               IsAttn() { return (HeaderStatus() == kXR_attn); }

   inline bool        IsError() { return (fStatusCode != kXrdMSC_ok); };

   inline bool        IsMarshalled() { return fMarshalled; }
   void               Marshall();
   inline bool        MatchStreamid(short sid) { return (HeaderSID() == sid);}
   int                ReadRaw(XrdClientPhyConnection *phy);
   inline void        SetHeaderStatus(kXR_unt16 sts) { fHdr.status = sts; }
   inline void        SetMarshalled(bool m) { fMarshalled = m; }
   inline void        SetStatusCode(kXR_unt16 status) { fStatusCode = status; }
   void               Unmarshall();

};
#endif
