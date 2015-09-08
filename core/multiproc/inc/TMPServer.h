/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPServer
#define ROOT_TMPServer

#include "TSysEvtHandler.h" //TFileHandler
#include "TSocket.h"
#include "TBufferFile.h"
#include "TClass.h"
#include "MPSendRecv.h"
#include <unistd.h> //pid_t
#include <string>
#include <memory> //unique_ptr

//////////////////////////////////////////////////////////////////////////
///
/// This class works in conjuction with TMPClient.
/// When TMPClient::Fork is called, a TMPServer instance is passed to it
/// which will take control of the ROOT session in the children processes.
///
/// After forking, everytime a message is sent or broadcast to the servers
/// TMPServer::Notify is called and the message is retrieved.
/// Messages exchanged between TMPClient and TMPServer should be sent with
/// the MPSend standalone function.
///
/// How to use:
/// Your worker class should inherit from TMPServer and implement a
/// HandleInput method that overrides TMPServer's.
/// When a message is received, if thee code is lower than 1000 the
/// overriding HandleInput method is called. Codes above 1000 are reserved
/// for special types of messages (see EMPCode) and are handled automatically.
///
//////////////////////////////////////////////////////////////////////////

class TMPServer : public TFileHandler {
   ClassDef(TMPServer, 0);
public:
   TMPServer();
   virtual ~TMPServer() {}; // TODO we might think of a way to get this dtor called at the end of execution, so we can move the Quit method here
   virtual void Init(int fd); 
   inline TSocket* GetSocket() { return fS.get(); }
   inline pid_t GetPid() { return fPid; }

private:
   virtual void HandleInput(MPCodeBufPair& msg);
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }

   std::shared_ptr<TSocket> fS; ///< this server's socket
   pid_t fPid; ///< the PID of the process in which this server is running
};

#endif
