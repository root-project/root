// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

#include "TXMessage.h"

class TXUnsolicitedMsgSender;

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

  inline void SendUnsolicitedMsg(TXUnsolicitedMsgSender *sender, TXMessage *unsolmsg) {
    // We simply send the event
    if (UnsolicitedMsgHandler)
      UnsolicitedMsgHandler->ProcessUnsolicitedMsg(sender, unsolmsg);
  }

  inline TXUnsolicitedMsgSender() { UnsolicitedMsgHandler = 0; }
};

#endif
