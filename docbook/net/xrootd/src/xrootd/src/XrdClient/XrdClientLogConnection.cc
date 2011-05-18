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

const char *XrdClientLogConnectionCVSID = "$Id$";

#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientPhyConnection.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientSid.hh"



//_____________________________________________________________________________
XrdClientLogConnection::XrdClientLogConnection(XrdClientSid *sidmgr):
  fSidManager(sidmgr) {

   // Constructor

   fPhyConnection = 0;
   fStreamid = fSidManager ? fSidManager->GetNewSid() : 0;
}

//_____________________________________________________________________________
XrdClientLogConnection::~XrdClientLogConnection() {
   // Destructor

   // Decrement counter in the reference phy conn
   if (fPhyConnection)
      fPhyConnection->CountLogConn(-1);
   if (fSidManager) 
      fSidManager->ReleaseSidTree(fStreamid);
}

//_____________________________________________________________________________
int XrdClientLogConnection::WriteRaw(const void *buffer, int bufferlength,
				     int substreamid)
{
   // Send over the open physical connection 'bufferlength' bytes located
   // at buffer.
   // Return number of bytes sent.

   Info(XrdClientDebug::kDUMPDEBUG,
	"WriteRaw",
	"Writing " << bufferlength << " bytes to physical connection");
  
   return fPhyConnection->WriteRaw(buffer, bufferlength, substreamid);
}

//_____________________________________________________________________________
int XrdClientLogConnection::ReadRaw(void *buffer, int bufferlength)
{
   // Receive from the open physical connection 'bufferlength' bytes and 
   // save in buffer.
   // Return number of bytes received.

   Info(XrdClientDebug::kDUMPDEBUG,
	"ReadRaw",
	"Reading " << bufferlength << " bytes from physical connection");

   return fPhyConnection->ReadRaw(buffer, bufferlength);
  

}

//_____________________________________________________________________________
UnsolRespProcResult XrdClientLogConnection::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender,
								  XrdClientMessage *unsolmsg)
{
   // We are here if an unsolicited response comes from the connmgr
   // The response comes in the form of an XrdClientMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited responses
   // are asynchronous by nature.

   //Info(XrdClientDebug::kDUMPDEBUG, "LogConnection",
   //     "Processing unsolicited response (ID:"<<unsolmsg->HeaderSID()<<")");

   // Local processing ....

   // We propagate the event to the obj which registered itself here
   return SendUnsolicitedMsg(sender, unsolmsg);

}
