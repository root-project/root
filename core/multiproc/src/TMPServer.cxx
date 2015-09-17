#include "TMPServer.h"
#include "EMPCode.h"
#include "MPSendRecv.h"
#include "TSystem.h"
#include <string>
#include <iostream>
#include <memory> //unique_ptr


//////////////////////////////////////////////////////////////////////////
/// Class constructor.
/// Note that this does not set variables like fPid or fS (server's socket).\n
/// These operations are handled by the Init method, which is called after
/// forking.\n
/// This separation is in place because the instantiation of a server
/// must be done once _before_ forking, while the initialization of the
/// members must be done _after_ forking by each of the children processes.
TMPServer::TMPServer() : TFileHandler(-1, kRead), fS(), fPid(0)
{
}


//////////////////////////////////////////////////////////////////////////
/// This method is called by children processes right after forking.
/// Initialization of server properties that must be delayed until after
/// forking must be done here.
/// For example, Init saves the pid into fPid, and TMPServer is added
/// to the main eventloop (as a TFileHandler).\n
/// This method should also be called by possible overriding implementations.
void TMPServer::Init(int fd)
{
   fS.reset(new TSocket(fd,"MPsock")); //TSocket's constructor with this signature seems much faster than TSocket(int fd)
   fPid = getpid();

   //TFileHandler's stuff
   //these operations _must_ be done in the overriding implementations too
   SetFd(fd);
   Add();
}


//////////////////////////////////////////////////////////////////////////
/// TMPServer's implementation of HandleInput.
/// This method is called upon receiving a message with a code >= 1000 (i.e.
/// EMPCode). It handles the most generic types of messages.\n
/// Classes inheriting from TMPServer should implement their own HandleInput
/// function, that should be able to handle codes specific to that application.\n
/// The appropriate version of the HandleInput method (TMPServer's or the 
/// overriding version) is automatically called depending on the message code.
void TMPServer::HandleInput(MPCodeBufPair& msg)
{
   unsigned code = msg.first;
   
   std::string reply = "S" + std::to_string(fPid);
   if (code == EMPCode::kMessage) {
      //general message, ignore it
      reply += ": ok";
      MPSend(fS.get(), EMPCode::kMessage, reply.data());
   } else if (code == EMPCode::kError) {
      //general error, ignore it
      reply += ": ko";
      MPSend(fS.get(), EMPCode::kMessage, reply.data());
   } else if (code == EMPCode::kShutdownOrder || code == EMPCode::kFatalError) {
      //client is asking the server to shutdown or client is dying
      MPSend(fS.get(), EMPCode::kShutdownNotice, reply.data());
      gSystem->Exit(0);
   } else {
      reply += ": unknown code received. code=" + std::to_string(code);
      MPSend(fS.get(), EMPCode::kError, reply.data());
   }
}

//////////////////////////////////////////////////////////////////////////
/// This method is called by TFileHandler when there's an event on the TSocket fS.
/// It checks what kind of message was received and calls the appropriate
/// handler function (TMPServer::HandleInput or overridden version).
Bool_t TMPServer::Notify()
{
   MPCodeBufPair msg = MPRecv(fS.get());
   if(msg.first == EMPCode::kRecvError) {
      std::cerr << "Lost connection to client\n";
      gSystem->Exit(0);
   }
   if(msg.first < 1000)
      HandleInput(msg); //call overridden method
   else
      TMPServer::HandleInput(msg); //call this class' method

   return kTRUE;
}
