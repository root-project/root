//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientUnsolMsg                                                          //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Base classes for unsolicited msg senders/receivers                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef XRC_UNSOLMSG_H
#define XRC_UNSOLMSG_H

class XrdClientMessage;
class XrdClientUnsolMsgSender;

// The processing result for an unsolicited response
enum UnsolRespProcResult {
   kUNSOL_CONTINUE = 0, // Dispatching must continue to other interested handlers
   kUNSOL_KEEP,         // Dispatching ended, but stream still alive (must keep the SID)
   kUNSOL_DISPOSE       // Dispatching ended, stream no more to be used
};

// Handler

class XrdClientAbsUnsolMsgHandler {
 public:
  
   virtual ~XrdClientAbsUnsolMsgHandler() { }
   // To be called when an unsolicited response arrives from the lower layers
   virtual UnsolRespProcResult ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender, 
				      XrdClientMessage *unsolmsg) = 0;

};

// Sender

class XrdClientUnsolMsgSender {

 public:

   virtual ~XrdClientUnsolMsgSender() { }

   // The upper level handler for unsolicited responses
   XrdClientAbsUnsolMsgHandler *UnsolicitedMsgHandler;

   inline UnsolRespProcResult SendUnsolicitedMsg(XrdClientUnsolMsgSender *sender, XrdClientMessage *unsolmsg) {
      // We simply send the event
      if (UnsolicitedMsgHandler)
	 return (UnsolicitedMsgHandler->ProcessUnsolicitedMsg(sender, unsolmsg));

      return kUNSOL_CONTINUE;
   }

   inline XrdClientUnsolMsgSender() { UnsolicitedMsgHandler = 0; }
};

#endif
