// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXConnectionMgr                                                      //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// The Connection Manager handles socket communications for TXNetFile   //
// action: connect, disconnect, read, write. It is a static object of   //
// the TXNetFile class such that within a single application multiple   //
// TXNetFile objects share the same connection manager.                 //
// The connection manager maps multiple logical connections on a single //
// physical connection.                                                 //
// There is one and only one logical connection per client (XNTetFile   //
// object), and one and only one physical connection per server:port.   //
// Thus multiple TXNetFile objects withing a given application share    //
// the same physical TCP channel to communicate with the server.        //
// This reduces the time overhead for socket creation and reduces also  //
// the server load due to handling many sockets.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnv.h"
#include "TXConnectionMgr.h"
#include "TXDebug.h"
#include "TXMessage.h"
#include "TError.h"
#include "TXMutexLocker.h"

#ifdef AIX
#include <sys/sem.h>
#else
#include <semaphore.h>
#endif

extern TEnv *gEnv;

ClassImp(TXConnectionMgr);

TXConnectionMgr *TXConnectionMgr::fgInstance = 0;

//_____________________________________________________________________________
extern "C" void * GarbageCollectorThread(void * arg)
{
   // Function executed in the garbage collector thread

   int i;
   TXConnectionMgr *thisObj = (TXConnectionMgr *)arg;

   TThread::SetCancelDeferred();
   TThread::SetCancelOn();

   while (1) {
      TThread::CancelPoint();

      thisObj->GarbageCollect();

      for (i = 0; i < 10; i++) {
	 TThread::CancelPoint();

         gSystem->Sleep(200);
      }
   }

   TThread::Exit();
   return 0;
}

//_____________________________________________________________________________
TXConnectionMgr* TXConnectionMgr::Instance()
{
   // Create unique instance of the connection manager

   if(fgInstance == 0) {
      fgInstance = new TXConnectionMgr;
      if(!fgInstance) {
         gSystem->Error("Instance",
                        "Fatal ERROR *** Object creation with new failed !"
                        " Probable system resources exhausted.");
         gSystem->Abort();
      }
   }
   return fgInstance;
}

//_____________________________________________________________________________
void TXConnectionMgr::Reset()
{
   // Reset the connection manager

   delete(fgInstance);
   fgInstance = 0;
}

//____________________________________________________
TXConnectionMgr::TXConnectionMgr()
{
   // TXConnectionMgr constructor.
   // Creates a Connection Manager object.
   // Starts the garbage collector thread.

   // Initialization of lock mutex
   fMutex = new TMutex(kTRUE);

   if (!fMutex)
      Info("TXConnectionMgr", "Can't create mutex: out of system resources");

   fThreadHandler = 0;

   // Garbage collector thread creation void *(*start_routine, void*)
   if (gEnv->GetValue("XNet.StartGarbageCollectorThread",
                      DFLT_STARTGARBAGECOLLECTORTHREAD)) {

      // The type of the thread func makes it a detached thread
      fThreadHandler = new TThread((TThread::VoidFunc_t) GarbageCollectorThread,
					 this);


      if (!fThreadHandler)
         Info("TXConnectionMgr",
              "Can't create garbage collector thread: out of system resources");

      fThreadHandler->Run();


   }
   else
      if(DebugLevel() >= TXDebug::kHIDEBUG)
         Info("TXConnectionMgr",
              "Explicitly requested not to start the garbage collector"
              " thread. Are you sure?");
}

//_____________________________________________________________________________
TXConnectionMgr::~TXConnectionMgr()
{
   // Deletes mutex locks, stops garbage collector thread.

   UInt_t i=0;

   {
      TXMutexLocker mtx(fMutex);

      for (i = 0; i < fLogVec.size(); i++)
	 if (fLogVec[i]) Disconnect(i, kFALSE);

   }

   if (fThreadHandler) {
      fThreadHandler->Kill();
      //fThreadHandler->Join();
   }

   GarbageCollect();

   SafeDelete(fMutex);

   delete(fgInstance);
}

//_____________________________________________________________________________
void TXConnectionMgr::GarbageCollect()
{
   // Get rid of unused physical connections. 'Unused' means not used for a
   // TTL time from any logical one. The TTL depends on the kind of remote
   // server. For a load balancer the TTL is very high, while for a data server
   // is quite small.

   // Mutual exclusion on the vectors and other vars
   {
      TXMutexLocker mtx(fMutex);

      // We cycle all the physical connections
      for (unsigned short int i = 0; i < fPhyVec.size(); i++) { 
   
	 // If a single physical connection has no linked logical connections,
	 // then we kill it if its TTL has expired
	 if ( fPhyVec[i] && (GetPhyConnectionRefCount(fPhyVec[i]) <= 0) && 
	      fPhyVec[i]->ExpiredTTL() ) {
      
	    if (DebugLevel() >= TXDebug::kDUMPDEBUG)
	       Info("GarbageCollect", "Purging physical connection %d", i);

	    // Wait until the physical connection is unlocked (it may be in use by 
	    // slow processes)

	    fPhyVec[i]->Disconnect();
	    SafeDelete(fPhyVec[i]);
	    fPhyVec[i] = 0;
      
	    if (DebugLevel() >= TXDebug::kHIDEBUG)
	       Info("GarbageCollect", "Purged physical connection %d", i);

	 }
      }


   }

}

//_____________________________________________________________________________
short int TXConnectionMgr::Connect(TString RemoteAddress, 
                                   Int_t TcpPort, Int_t TcpWindowSize)
{
   // Connects to the remote server:
   //  - Looks for an existing physical connection already bound to 
   //    RemoteAddress:TcpPort;
   //  - If needed, creates a TCP channel to RemoteAddress:TcpPort
   //    (this is a physical connection);
   //  - Creates a logical connection and binds it to the previous 
   //    created physical connection;
   //  - Returns the logical connection ID. Every client will use this
   //    ID to deal with the server.

   TXLogConnection *logconn;
   TXPhyConnection *phyconn;
   short int  newid;
   Bool_t phyfound;

   // First we get a new logical connection object
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Connect", "Creating a logical connection...");

   logconn = new TXLogConnection();
   if (!logconn) {
      Error("Connect","Fatal ERROR *** Object creation with new failed !"
                      " Probable system resources exhausted.");
      gSystem->Abort();
   }
  
   if(DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("Connect", "Getting lock...");

   {
      TXMutexLocker mtx(fMutex);

      // If we already have a physical connection to that host:port, 
      // then we use that
      phyfound = kFALSE;
      if (DebugLevel() >= TXDebug::kHIDEBUG)
	 Info("Connect",
	      "Looking for an available physical connection for address [%s:%d]", 
	      RemoteAddress.Data(), TcpPort);

      for (unsigned short int i=0; i < fPhyVec.size(); i++) {
	 if (fPhyVec[i] && fPhyVec[i]->IsValid() &&
	     fPhyVec[i]->IsPort(TcpPort) && fPhyVec[i]->IsAddress(RemoteAddress)) {
	    // We link that physical connection to the new logical connection
	    fPhyVec[i]->Touch();
	    logconn->SetPhyConnection( fPhyVec[i] );
	    phyfound = kTRUE;
	    break;
	 }
      }

   }

   if (!phyfound) {

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("Connect",
              "Physical connection not found. Creating a new one...");

      // If not already present, then we must build a new physical connection, 
      // and try to connect it
      // While we are trying to connect, the mutex must be unlocked
      // Note that at this point logconn is a pure local instance, so it 
      // does not need to be protected by mutex
      phyconn = new TXPhyConnection(this);

      if (!phyconn) {
         Error("Connect","Fatal ERROR *** Object creation with new failed !"
                         " Probable system resources exhausted.");
         gSystem->Abort();
      }
      if (phyconn && phyconn->Connect(RemoteAddress, TcpPort, TcpWindowSize)) {

         logconn->SetPhyConnection(phyconn);

         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("Connect", "New physical connection to server [%s:%d]"
                            " succesfully created.",
                 RemoteAddress.Data(), TcpPort); 
      } else 
         return -1;
   }


   // Now, we are connected to the host desired.
   // The physical connection can be old or newly created
   {
      TXMutexLocker mtx(fMutex);

      // Then, if needed, we push the physical connection into its vector
      if (!phyfound)
	 fPhyVec.push_back(phyconn);

      // Then we push the logical connection into its vector
      fLogVec.push_back(logconn);
 
      // Its ID is its position inside the vector, we must return it later
      newid = fLogVec.size()-1;

      // Now some debug log
      if (DebugLevel() >= TXDebug::kHIDEBUG) {
	 Int_t logCnt = 0, phyCnt = 0;

	 for (unsigned short int i=0; i < fPhyVec.size(); i++)
	    if (fPhyVec[i])
	       phyCnt++;
	 for (unsigned short int i=0; i < fLogVec.size(); i++)
	    if (fLogVec[i])
	       logCnt++;

	 Info("Connect",
	      "LogConn: size:%d, count:%d - PhyConn: size:%d, count:%d",
	      fLogVec.size(), logCnt, phyCnt, fPhyVec.size());
      }

   }
  

   return newid;
}

//_____________________________________________________________________________
void TXConnectionMgr::Disconnect(short int LogConnectionID, 
                                 Bool_t ForcePhysicalDisc)
{
   // Deletes a logical connection.
   // Also deletes the related physical one if ForcePhysicalDisc=TRUE.

   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("Disconnect", "Getting lock...");

   {
      TXMutexLocker mtx(fMutex);

      if ((UInt_t(LogConnectionID) >= fLogVec.size()) || (!fLogVec[LogConnectionID])) {
	 Error("Disconnect", "Destroying nonexistent logconnid %d.", LogConnectionID);
	 return;
      }


      if (ForcePhysicalDisc) {
	 // We disconnect the phyconn
	 // But it will be removed by the garbagecollector as soon as possible
	 // Note that here we cannot destroy the phyconn, since there can be other 
	 // logconns pointing to it the phyconn will be removed when there are no 
	 // more logconns pointing to it
	 fLogVec[LogConnectionID]->GetPhyConnection()->SetTTL(0);
	 fLogVec[LogConnectionID]->GetPhyConnection()->Disconnect();
      }
    
      fLogVec[LogConnectionID]->GetPhyConnection()->Touch();
      SafeDelete(fLogVec[LogConnectionID]);
      fLogVec[LogConnectionID] = 0;

      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
	 Info("Disconnect", "Unlocking...");

   }

}

//_____________________________________________________________________________
Int_t TXConnectionMgr::ReadRaw(short int LogConnectionID, void *buffer, 
                               Int_t BufferLength, ESendRecvOptions opt)
{
   // Read BufferLength bytes from the logical connection LogConnectionID

   TXLogConnection *logconn;

   logconn = GetConnection(LogConnectionID);

   if (logconn) {
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
         Info("ReadRaw", "Reading from logical connection %d",
              LogConnectionID);

      return logconn->ReadRaw(buffer, BufferLength, opt);
   }
   else {
      Info("ReadRaw", "There's not a logical connection with id=%d",
           LogConnectionID);

      return(-1);
   }
}

//_____________________________________________________________________________
TXMessage *TXConnectionMgr::ReadMsg(short int LogConnectionID, ESendRecvOptions opt)
{
   TXLogConnection *logconn;
   TXMessage *mex;

   logconn = GetConnection(LogConnectionID);
   if (logconn) {
      //    if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      //      Info("ReadMsg", "Reading from logical connection %d",
      // 	   LogConnectionID);
   }

   // Parametric asynchronous stuff.
   // If we are going Sync, then we must build the message here,
   // otherwise the messages come directly from the queue
   if ( !gEnv->GetValue("XNet.GoAsynchronous", DFLT_GOASYNC) ) {

      // We get a new message directly from the socket.
      // The message gets inserted inside the phyconn queue
      // This line of code will be moved to a reader thread inside TXPhyConnection
      // Timeouts must not be ignored here, indeed they are an error
      // because we are waiting for a message that must come quickly
      mex = logconn->GetPhyConnection()->BuildXMessage(opt, kFALSE, kFALSE);

   }
   else {
      // Now we get the message from the queue, with the timeouts needed
      mex = logconn->GetPhyConnection()->ReadXMessage(LogConnectionID);
   }

   // Return the message unmarshalled to ClientServerCmd
   return mex;
}

//_____________________________________________________________________________
Int_t TXConnectionMgr::WriteRaw(short int LogConnectionID, const void *buffer, 
                                Int_t BufferLength, ESendRecvOptions opt)
{
   // Write BufferLength bytes into the logical connection LogConnectionID

   TXLogConnection *logconn;

   logconn = GetConnection(LogConnectionID);
   if (logconn) {
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
         Info("WriteRaw", "Writing %d bytes to logical connection %d.",
              BufferLength, LogConnectionID);

      return logconn->WriteRaw(buffer, BufferLength, opt);
   }
   else {
      Info("WriteRaw", "There's not a logical connection with id=%d",
           LogConnectionID);

      return(-1);
   }
}

//_____________________________________________________________________________
TXLogConnection *TXConnectionMgr::GetConnection(short int LogConnectionID)
{
   // Return a logical connection object that has LogConnectionID as its ID.

   TXLogConnection *res;

   {
      TXMutexLocker mtx(fMutex);
 
      res = fLogVec[LogConnectionID];
   }
  
   return res;
}

//_____________________________________________________________________________
short int TXConnectionMgr::GetPhyConnectionRefCount(TXPhyConnection *PhyConn)
{
   // Return the number of logical connections bound to the physical one 'PhyConn'
   int cnt = 0;

   {
      TXMutexLocker mtx(fMutex);

      for (unsigned short int i = 0; i < fLogVec.size(); i++)
	 if ( fLogVec[i] && (fLogVec[i]->GetPhyConnection() == PhyConn) ) cnt++;

   }
  
   return cnt;
}

//_____________________________________________________________________________
Bool_t TXConnectionMgr::ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                              TXMessage *unsolmsg)
{
   // We are here if an unsolicited response comes from a physical connection
   // The response comes in the form of an TXMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited responses
   // are asynchronous by nature.

   Info("Write", "Processing unsolicited response");

   // Local processing ....

   // Now we propagate the message to the interested objects.
   // In our architecture, the interested objects are the objects which
   // self-registered in the logical connections belonging to the Phyconn
   // which threw the event
   // So we throw the evt towards each logical connection
   {
      TXMutexLocker mtx(fMutex);

      for (unsigned short int i = 0; i < fLogVec.size(); i++)
	 if ( fLogVec[i] && (fLogVec[i]->GetPhyConnection() == sender) ) {
	    fLogVec[i]->ProcessUnsolicitedMsg(sender, unsolmsg);
	 }

   }


   return kTRUE;
}
