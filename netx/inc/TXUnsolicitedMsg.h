// @(#)root/netx:$Name:  $:$Id: TXUnsolicitedMsg.h,v 1.2 2004/08/20 22:16:33 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXUnsolicitedMsg
#define ROOT_TXUnsolicitedMsg

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXUnsolicitedMsg                                                     //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Base classes to send unsolicited responses as event to other objs    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TXUnsolicitedMsgSender;
class TXMessage;

// Handler

class TXAbsUnsolicitedMsgHandler {
public:

  // To be called when an unsolicited response arrives from the lower layers
  virtual Bool_t ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                       TXMessage *unsolmsg) = 0;

};

// Sender

class TXUnsolicitedMsgSender {
public:
   // The upper level handler for unsolicited responses
  TXAbsUnsolicitedMsgHandler *UnsolicitedMsgHandler;

  TXUnsolicitedMsgSender();
  void SendUnsolicitedMsg(TXUnsolicitedMsgSender *sender, TXMessage *unsolmsg);
};

//________________________________________________________________________
inline void TXUnsolicitedMsgSender::SendUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                                       TXMessage *unsolmsg)
{
    // We simply send the event
    if (UnsolicitedMsgHandler)
      UnsolicitedMsgHandler->ProcessUnsolicitedMsg(sender, unsolmsg);
}

//________________________________________________________________________
inline TXUnsolicitedMsgSender::TXUnsolicitedMsgSender()
{
     UnsolicitedMsgHandler = 0;
}

#endif
