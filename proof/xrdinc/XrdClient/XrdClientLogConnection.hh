#ifndef XRD_CLOGCONNECTION_H
#define XRD_CLOGCONNECTION_H
/******************************************************************************/
/*                                                                            */
/*           X r d C l i e n t L o g C o n n e c t i o n . h h                */
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
// Class implementing logical connections                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XProtocol/XPtypes.hh"
#include "XrdClient/XrdClientUnsolMsg.hh"

class XrdClientSid;
class XrdClientPhyConnection;

class XrdClientLogConnection: public XrdClientAbsUnsolMsgHandler, 
   public XrdClientUnsolMsgSender {
private:
   XrdClientPhyConnection            *fPhyConnection;

   // A logical connection has a private streamid
   kXR_unt16                         fStreamid;

   XrdClientSid                     *fSidManager;

public:
   XrdClientLogConnection(XrdClientSid *sidmgr);
   virtual ~XrdClientLogConnection();

   inline XrdClientPhyConnection     *GetPhyConnection() {
      return fPhyConnection;
   }

   UnsolRespProcResult               ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender,
							   XrdClientMessage *unsolmsg);

   int                               ReadRaw(void *buffer, int BufferLength);

   inline void                       SetPhyConnection(XrdClientPhyConnection *PhyConn) {
      fPhyConnection = PhyConn;
   }

    int                               WriteRaw(const void *buffer, int BufferLength, int substreamid);

   inline kXR_unt16                  Streamid() {
      return fStreamid;
   };
};
#endif
