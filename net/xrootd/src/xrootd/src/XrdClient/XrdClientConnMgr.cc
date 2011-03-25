//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientConnMgr                                                     //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// The connection manager maps multiple logical connections on a single //
// physical connection.                                                 //
// There is one and only one logical connection per client              //
// and one and only one physical connection per server:port.            //
// Thus multiple objects withing a given application share              //
// the same physical TCP channel to communicate with a server.          //
// This reduces the time overhead for socket creation and reduces also  //
// the server load due to handling many sockets.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

const char *XrdClientConnMgrCVSID = "$Id$";

#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientThread.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientSid.hh"
#ifdef WIN32
#include "XrdSys/XrdWin32.hh"
#endif

#include <assert.h>

#ifdef AIX
#include <sys/sem.h>
#else
#include <semaphore.h>
#endif

// For user info
#ifndef WIN32
#include <sys/types.h>
#include <pwd.h>
#endif

// Max number allowed of logical connections ( 2**15 - 1, short int)
#define XRC_MAXVECTSIZE 32767

//_____________________________________________________________________________
void * GarbageCollectorThread(void *arg, XrdClientThread *thr)
{
   // Function executed in the garbage collector thread

   // Mask all allowed signals
   if (thr->MaskSignal(0) != 0)
      Error("GarbageCollectorThread", "Warning: problems masking signals");

   XrdClientConnectionMgr *thisObj = (XrdClientConnectionMgr *)arg;

   thr->SetCancelDeferred();
   thr->SetCancelOn();

   while (1) {
      thr->CancelPoint();

      thisObj->GarbageCollect();

      thr->CancelPoint();

      sleep(30);

   }

   return 0;
}

//_____________________________________________________________________________
int DisconnectElapsedPhyConn(const char *key,
			     XrdClientPhyConnection *p, void *voidcmgr)
{
   // Function applied to the hash table to disconnect the unused elapsed
   // physical connections

   XrdClientConnectionMgr *cmgr = (XrdClientConnectionMgr *)voidcmgr;
   assert(cmgr != 0);

   if (p) {
      if ((p->GetLogConnCnt() <= 0) && 
           p->ExpiredTTL() && p->IsValid()) {
         p->Touch();
         p->Disconnect();
      }
      
      if (!p->IsValid()) {

        // Make sure that we kill the socket in the case of conns killed by the server
        p->Touch();
        p->Disconnect();

	// And then add the instance to the trashed list
	cmgr->fPhyTrash.Push_back(p);

	// And remove the current from here
	return -1;
      }
   }
   
   // Process next
   return 0;
}



//_____________________________________________________________________________
int DumpPhyConn(const char *key,
		XrdClientPhyConnection *p, void *voidcmgr)
{

  if (!p) {
    Info(XrdClientDebug::kUSERDEBUG, "DumpPhyConn", "Phyconn entry, key=NULL");
    return 0;
  }
  
  Info(XrdClientDebug::kUSERDEBUG, "DumpPhyConn", "Phyconn entry, key='" <<
       (key ? key : "***def***") <<
       "', LogCnt=" << p->GetLogConnCnt() << (p->IsValid() ? " Valid" : " NotValid") )

  // Process next
  return 0;
}

//_____________________________________________________________________________
int DestroyPhyConn(const char *key,
			  XrdClientPhyConnection *p, void *voidcmgr)
{
  // Function applied to the hash table to destroy all the phyconns
  XrdClientConnectionMgr *cmgr = (XrdClientConnectionMgr *)voidcmgr;
  assert(cmgr != 0);

  if (p) {
    p->Touch();
    p->Disconnect();
    p->UnsolicitedMsgHandler = 0;
    delete(p);
  }

   // Process next, remove current item
   return -1;
}


//_____________________________________________________________________________
XrdClientConnectionMgr::XrdClientConnectionMgr() : fSidManager(0),
						   fGarbageColl(0)
{
   // XrdClientConnectionMgr constructor.
   // Creates a Connection Manager object.
   // Starts the garbage collector thread.
   BootUp();

}

//_____________________________________________________________________________
XrdClientConnectionMgr::~XrdClientConnectionMgr()
{
   // Deletes mutex locks, stops garbage collector thread.
  ShutDown();
}

//------------------------------------------------------------------------------
// Initializer
//------------------------------------------------------------------------------
bool XrdClientConnectionMgr::BootUp()
{
  fLastLogIdUsed = 0;

  fGarbageColl = new XrdClientThread(GarbageCollectorThread);

  if (!fGarbageColl)
     Error("ConnectionMgr",
           "Can't create garbage collector thread: out of system resources");

  fGarbageColl->Run(this);

  fSidManager = new XrdClientSid();
  if (!fSidManager) {
    Error("ConnectionMgr",
   "Can't create sid manager: out of system resources");
    abort();
  }

  return true;
}

bool XrdClientConnectionMgr::ShutDown()
{
  fPhyHash.Apply(DumpPhyConn, this);

  {
    XrdSysMutexHelper mtx(fMutex);

    for( int i = 0; i < fLogVec.GetSize(); i++)
      if (fLogVec[i]) Disconnect(i, TRUE);
  }

  if (fGarbageColl)
  {
    void *ret;
    fGarbageColl->Cancel();
    fGarbageColl->Join(&ret);
    delete fGarbageColl;
  }

  GarbageCollect();

  fPhyHash.Apply(DestroyPhyConn, this);

  for(int i = fPhyTrash.GetSize()-1; i >= 0; i--)
    DestroyPhyConn( "Trashed connection", fPhyTrash[i], this);

  fPhyTrash.Clear();
  fPhyHash.Purge();
  fLogVec.Clear();

  delete fSidManager;

  fSidManager  = 0;
  fGarbageColl = 0;

  return true;
}


//_____________________________________________________________________________
void XrdClientConnectionMgr::GarbageCollect()
{
   // Get rid of unused physical connections. 'Unused' means not used for a
   // TTL time from any logical one. The TTL depends on the kind of remote
   // server. For a load balancer the TTL is very high, while for a data server
   // is quite small.

   // Mutual exclusion on the vectors and other vars
   XrdSysMutexHelper mtx(fMutex);

   if (fPhyHash.Num() > 0) {

     if(DebugLevel() >= XrdClientDebug::kUSERDEBUG)
       fPhyHash.Apply(DumpPhyConn, this);

      // Cycle all the physical connections to disconnect the elapsed ones
      fPhyHash.Apply(DisconnectElapsedPhyConn, this);

   }

   // Cycle all the trashed physical connections to destroy the elapsed once more
   // after a disconnection. Doing this way, a phyconn in async mode has
   // all the time it needs to terminate its reader thread pool
   for (int i = fPhyTrash.GetSize()-1; i >= 0; i--) {

     DumpPhyConn("Trashed connection",
		 fPhyTrash[i], this);

     if ( !fPhyTrash[i] || 
	  ((fPhyTrash[i]->GetLogConnCnt() <= 0) && (fPhyTrash[i]->ExpiredTTL())) ) {

        if (fPhyTrash[i] && (fPhyTrash[i]->GetReaderThreadsCnt() <= 0)) {
           delete fPhyTrash[i];
           

        }


       fPhyTrash.Erase(i);
     }
   }

}

//_____________________________________________________________________________
int XrdClientConnectionMgr::Connect(XrdClientUrlInfo RemoteServ)
{
   // Connects to the remote server:
   //  - Looks first for an existing physical connection already bound to
   //    User@RemoteAddress:TcpPort;
   //  - If needed, creates a TCP channel to User@RemoteAddress:TcpPort
   //   (this is a physical connection);
   //  - Creates a logical connection and binds it to the previous 
   //    created physical connection;
   //  - Returns the logical connection ID. Every client will use this
   //    ID to deal with the server.

   XrdClientLogConnection *logconn = 0;
   XrdClientPhyConnection *phyconn = 0;
   CndVarInfo *cnd = 0;

   int newid = -1;
   bool phyfound = FALSE;

   // First we get a new logical connection object
   Info(XrdClientDebug::kHIDEBUG,
	"Connect", "Creating a logical connection...");

   logconn = new XrdClientLogConnection(fSidManager);
   if (!logconn) {
      Error("Connect", "Object creation failed. Aborting.");
      abort();
   }

   // If empty, fill the user name with the default to avoid fake mismatches
   if (RemoteServ.User.length() <= 0) {
#ifndef WIN32
      struct passwd *pw = getpwuid(getuid());
      RemoteServ.User = (pw) ? pw->pw_name : "";
#else
      char  name[256];
      DWORD length = sizeof (name);
      ::GetUserName(name, &length);
      RemoteServ.User = name;
#endif
   }

   // Keys
   XrdOucString key;
   XrdOucString key1(RemoteServ.User.c_str(), 256); key1 += '@';
   key1 += RemoteServ.Host; key1 += ':'; key1 += RemoteServ.Port;
   XrdOucString key2(RemoteServ.User.c_str(), 256); key2 += '@';
   key2 += RemoteServ.HostAddr; key2 += ':'; key2 += RemoteServ.Port;

   do {

     
       fMutex.Lock();
       cnd = 0;

       cnd = fConnectingCondVars.Find(key1.c_str());
       if (!cnd) cnd = fConnectingCondVars.Find(key2.c_str());

       // If there are no connection attempts in progress...
       if (!cnd) {

	 // If we already have a physical connection to that host:port, 
	 // then we use that
	 if (fPhyHash.Num() > 0) {
	   XrdClientPhyConnection *p = 0;

           // We must avoid the unfortunate pick of a disconnected phyconn
           GarbageCollect();

	   if (((p = fPhyHash.Find(key1.c_str())) ||
		(p = fPhyHash.Find(key2.c_str()))) && p->IsValid()) {
	     // We link that physical connection to the new logical connection
	     phyconn = p;
	     phyconn->CountLogConn();
	     phyconn->Touch();
	     logconn->SetPhyConnection(phyconn);

	     phyfound = TRUE;
	   }
	   else {
	     // no connection attempts in progress and no already established connections
	     // Mark this as an ongoing attempt
	     // Now we have a pending conn attempt
	     fConnectingCondVars.Rep(key1.c_str(), new CndVarInfo(), 0, Hash_keepdata);
	   }
	 }

	 fMutex.UnLock();
       }
       else {
	 // this is an attempt which must wait for the result of a previous one
	 // In any case after the wait we loop and recheck
	 cnd->cv.Lock();
	 cnd->cnt++;
	 fMutex.UnLock();
	 cnd->cv.Wait();	 
	 cnd->cnt--;
	 cnd->cv.UnLock();
       }

   } while (cnd); // here cnd means "if there is a condvar to wait on"

   // We are here if cnd == 0
  
   if (!phyfound) {
      
      Info(XrdClientDebug::kHIDEBUG,
	   "Connect",
	   "Physical connection not found. Creating a new one...");

      if (DebugLevel() >= XrdClientDebug::kHIDEBUG)
	fPhyHash.Apply(DumpPhyConn, this);


      // If not already present, then we must build a new physical connection, 
      // and try to connect it
      // While we are trying to connect, the mutex must be unlocked
      // Note that at this point logconn is a pure local instance, so it 
      // does not need to be protected by mutex
      if (!(phyconn = new XrdClientPhyConnection(this, fSidManager))) {
         Error("Connect", "Object creation failed. Aborting.");
         abort();
      }
      if ( phyconn && phyconn->Connect(RemoteServ) ) {

         phyconn->CountLogConn();
         logconn->SetPhyConnection(phyconn);

         if (DebugLevel() >= XrdClientDebug::kHIDEBUG)
            Info(XrdClientDebug::kHIDEBUG,
		 "Connect",
		 "New physical connection to server " <<
		 RemoteServ.Host << ":" << RemoteServ.Port <<
		 " succesfully created.");

      } else {

	// The connection attempt failed, so we signal all the threads waiting for a result
	{
	  XrdSysMutexHelper mtx(fMutex);
	  int cnt;

          key = key1;
          cnd = fConnectingCondVars.Find(key.c_str());
          if (!cnd) { key = key2; cnd = fConnectingCondVars.Find(key.c_str()); }
	  if (cnd) {
	    cnd->cv.Lock();
	    cnd->cv.Broadcast();
	    fConnectingCondVars.Del(key.c_str());
	    cnt = cnd->cnt;
	    cnd->cv.UnLock();

	    if (!cnt) {
	      Info(XrdClientDebug::kHIDEBUG, "Connect",
		   "Destroying connection condvar for " << key );
	      delete cnd;
	    }
	  }
	}
	 delete logconn;
	// And then add the instance to the trashed list
	 phyconn->Touch();
	 fPhyTrash.Push_back(phyconn);
	 //delete phyconn;
	 
         return -1;
      }

   }


   // Now, we are connected to the host desired.
   // The physical connection can be old or newly created
   {
      XrdSysMutexHelper mtx(fMutex);
      phyconn->WipeStreamid(logconn->Streamid());

      // Then, if needed, we push the physical connection into its vector
      if (!phyfound) {
         if (!phyconn)
	    Error("Connect"," problems connecting to " << key1);
  	 fPhyHash.Rep(key1.c_str(), phyconn, 0, Hash_keepdata);
      }

      if (fLogVec.GetSize() < XRC_MAXVECTSIZE) {
            // Then we push the logical connection into its vector, up to a max size
            fLogVec.Push_back(logconn);
            // and the new position is the ID
            newid = fLogVec.GetSize()-1;
      }
      else {
         // The array is too big, get a free slot, if any
         newid = -1;
         for (int i = 0; i < fLogVec.GetSize(); i++) {
            int idx = (fLastLogIdUsed + i) % fLogVec.GetSize();
            if (!fLogVec[idx]) {
               fLogVec[idx] = logconn;
               newid = idx;
               fLastLogIdUsed = idx;
               break;
            }
         }
         if (newid == -1) {
            delete logconn;
            Error("Connect", "Critical error - Out of allocated resources:"
                  " max number allowed of logical connections reached ("<<XRC_MAXVECTSIZE<<")");
            return -1;
         }
      }

      // Now some debug log
      if (DebugLevel() >= XrdClientDebug::kHIDEBUG) {

         int logCnt = 0;
         for (int i=0; i < fLogVec.GetSize(); i++)
            if (fLogVec[i])
               logCnt++;

         Info(XrdClientDebug::kHIDEBUG, "Connect",
              "LogConn: size:" << fLogVec.GetSize() << " count: " << logCnt <<
              "PhyConn: size:" << fPhyHash.Num());
      }
   


      // The connection attempt went ok, so we signal all the threads waiting for a result, if we can still find the corresponding condvar
      int cnt;
      //      if (!phyfound) {
        key = key1;
        cnd = fConnectingCondVars.Find(key.c_str());
        if (!cnd) { key = key2; cnd = fConnectingCondVars.Find(key.c_str()); }
	if (cnd) {
	  cnd->cv.Lock();
	  cnd->cv.Broadcast();
	  fConnectingCondVars.Del(key.c_str());
	  cnt = cnd->cnt;
	  cnd->cv.UnLock();

	  if (!cnt) {
	    Info(XrdClientDebug::kHIDEBUG, "Connect",
		 "Destroying connection condvar for " << key );
	    delete cnd;
	  }
	}
	//      }

   } // mutex

   return newid;
}

//_____________________________________________________________________________
void XrdClientConnectionMgr::Disconnect(int LogConnectionID, 
                                 bool ForcePhysicalDisc)
{
   // Deletes a logical connection.
   // Also deletes the related physical one if ForcePhysicalDisc=TRUE.
   if (LogConnectionID < 0) return;

   {
      XrdSysMutexHelper mtx(fMutex);

      if ((LogConnectionID < 0) ||
	  (LogConnectionID >= fLogVec.GetSize()) || (!fLogVec[LogConnectionID])) {
	 Error("Disconnect", "Destroying nonexistent logconn " << LogConnectionID);
	 return;
      }

      fLogVec[LogConnectionID]->GetPhyConnection()->WipeStreamid(fLogVec[LogConnectionID]->Streamid());
      if (ForcePhysicalDisc) {
	 // We disconnect the phyconn
	 // But it will be removed by the garbagecollector as soon as possible
	 // Note that here we cannot destroy the phyconn, since there can be other 
	 // logconns pointing to it the phyconn will be removed when there are no 
	 // more logconns pointing to it
 	 fLogVec[LogConnectionID]->GetPhyConnection()->UnsolicitedMsgHandler = 0;
	 fLogVec[LogConnectionID]->GetPhyConnection()->Disconnect();
	 GarbageCollect();
      }

    
      fLogVec[LogConnectionID]->GetPhyConnection()->Touch();
      delete fLogVec[LogConnectionID];
      fLogVec[LogConnectionID] = 0;

      Info(XrdClientDebug::kHIDEBUG, "Disconnect",
           " LogConnID: " << LogConnectionID <<" destroyed");
   }

}

//_____________________________________________________________________________
int XrdClientConnectionMgr::ReadRaw(int LogConnectionID, void *buffer, 
                               int BufferLength)
{
   // Read BufferLength bytes from the logical connection LogConnectionID

   XrdClientLogConnection *logconn;

   logconn = GetConnection(LogConnectionID);

   if (logconn) {
      return logconn->ReadRaw(buffer, BufferLength);
   }
   else {
      Error("ReadRaw", "There's not a logical connection with id " <<
           LogConnectionID);

      return(TXSOCK_ERR);
   }
}

//_____________________________________________________________________________
XrdClientMessage *XrdClientConnectionMgr::ReadMsg(int LogConnectionID)
{
   XrdClientLogConnection *logconn;
   XrdClientMessage *mex;

   logconn = GetConnection(LogConnectionID);

   // Now we get the message from the queue, with the timeouts needed
   // Note that the physical connection know about streamids, NOT logconnids !!
   mex = logconn->GetPhyConnection()->ReadMessage(logconn->Streamid());

   // Return the message unmarshalled to ClientServerCmd
   return mex;
}

//_____________________________________________________________________________
int XrdClientConnectionMgr::WriteRaw(int LogConnectionID, const void *buffer, 
				     int BufferLength, int substreamid) {
   // Write BufferLength bytes into the logical connection LogConnectionID

   XrdClientLogConnection *logconn;

   logconn = GetConnection(LogConnectionID);

   if (logconn) {
       return logconn->WriteRaw(buffer, BufferLength, substreamid);
   }
   else {
      Error("WriteRaw", "There's not a logical connection with id " <<
	    LogConnectionID);

      return(TXSOCK_ERR);
   }
}

//_____________________________________________________________________________
XrdClientLogConnection *XrdClientConnectionMgr::GetConnection( int LogConnectionID)
{
   // Return a logical connection object that has LogConnectionID as its ID.
   XrdSysMutexHelper mtx(fMutex);
 
   return (LogConnectionID > -1) ? fLogVec[LogConnectionID] : (XrdClientLogConnection *)0;
}

//____________________________________________________________________________
XrdClientPhyConnection *XrdClientConnectionMgr::GetPhyConnection(XrdClientUrlInfo server)
{
  // Gets pointer to the physical connection to 'server', if any.
  // Return 0 if none is found.

  XrdClientPhyConnection *p = 0;

  // If empty, fill the user name with the default to avoid fake mismatches
  if (server.User.length() <= 0) {
#ifndef WIN32
    struct passwd *pw = getpwuid(getuid());
    server.User = (pw) ? pw->pw_name : "";
#else
    char  name[256];
    DWORD length = sizeof (name);
    ::GetUserName(name, &length);
    server.User = name;
#endif
  }

  // Keys
  XrdOucString key;
  XrdOucString key1(server.User.c_str(), 256); key1 += '@';
  key1 += server.Host; key1 += ':'; key1 += server.Port;
  XrdOucString key2(server.User.c_str(), 256); key2 += '@';
  key2 += server.HostAddr; key2 += ':'; key2 += server.Port;

  if (fPhyHash.Num() > 0) {
    if (((p = fPhyHash.Find(key1.c_str())) ||
	 (p = fPhyHash.Find(key2.c_str()))) && !(p->IsValid())) {
      // We found an invalid connection: do not use it
      p = 0;
    }
  }

  // Done
  return p;
}

//_____________________________________________________________________________
UnsolRespProcResult XrdClientConnectionMgr::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender,
                                             XrdClientMessage *unsolmsg)
{
   // We are here if an unsolicited response comes from a physical connection
   // The response comes in the form of an TXMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited responses
   // are asynchronous by nature.

   //Info(XrdClientDebug::kDUMPDEBUG, "ConnectionMgr",
   //    "Processing unsolicited response (ID:"<<unsolmsg->HeaderSID()<<")");
   UnsolRespProcResult res = kUNSOL_CONTINUE;
   // Local processing ....

   // Now we propagate the message to the interested objects.
   // In our architecture, the interested objects are the objects which
   // self-registered in the logical connections belonging to the Phyconn
   // which threw the event
   // So we throw the evt towards each logical connection
   {
      // Find an interested logid
      XrdSysMutexHelper mtx(fMutex);
      
      for (int i = 0; i < fLogVec.GetSize(); i++) {
      
	 if ( fLogVec[i] && (fLogVec[i]->GetPhyConnection() == sender) ) {
	    fMutex.UnLock();
	    res = fLogVec[i]->ProcessUnsolicitedMsg(sender, unsolmsg);
            fMutex.Lock();

	    if (res != kUNSOL_CONTINUE) break;
	 }
      }
   }
   return res;
}
