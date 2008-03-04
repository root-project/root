//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientLogConnection                                               //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Class implementing logical connections                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef XRD_CLOGCONNECTION_H
#define XRD_CLOGCONNECTION_H


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
