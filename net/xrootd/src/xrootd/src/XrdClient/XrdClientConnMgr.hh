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

#ifndef XRC_CONNMGR_H
#define XRC_CONNMGR_H


#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClientUnsolMsg.hh"
#include "XrdClient/XrdClientPhyConnection.hh"
#include "XrdClient/XrdClientVector.hh"

class XrdClientSid;
class XrdClientLogConnection;
class XrdClientMessage;
class XrdClientThread;

// Ugly prototype to avoid warnings under solaris
//void * GarbageCollectorThread(void * arg, XrdClientThread *thr);

class XrdClientConnectionMgr: public XrdClientAbsUnsolMsgHandler, 
                       XrdClientUnsolMsgSender {

private:
   XrdClientSid *fSidManager;

   XrdClientVector<XrdClientLogConnection*> fLogVec;
   XrdOucHash<XrdClientPhyConnection> fPhyHash;

   // To try not to reuse too much the same array ids
   int fLastLogIdUsed;
   // Phyconns are inserted here when they have to be destroyed later
   // All the phyconns here are disconnected.
   XrdClientVector<XrdClientPhyConnection *> fPhyTrash;

   // To arbitrate between multiple threads trying to connect to the same server.
   // The first has to connect, all the others have to wait for the completion
   // The meaning of this is: if there is a condvar associated to the hostname key,
   //  then wait for it to be signalled before deciding what to do
  class CndVarInfo {
  public:
    XrdSysCondVar cv;
    int cnt;
    CndVarInfo(): cv(0), cnt(0) {};
  };

   XrdOucHash<CndVarInfo> fConnectingCondVars;

   XrdSysRecMutex                fMutex; // mutex used to protect local variables
                                      // of this and TXLogConnection, TXPhyConnection
                                      // classes; not used to protect i/o streams

   XrdClientThread            *fGarbageColl;

   friend void * GarbageCollectorThread(void *, XrdClientThread *thr);
   UnsolRespProcResult
                 ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *sender,
                                       XrdClientMessage *unsolmsg);
public:
   XrdClientConnectionMgr();

   virtual ~XrdClientConnectionMgr();

  bool BootUp();
  bool ShutDown();


   int           Connect(XrdClientUrlInfo RemoteAddress);
   void          Disconnect(int LogConnectionID, bool ForcePhysicalDisc);

   void          GarbageCollect();

   XrdClientLogConnection 
                 *GetConnection(int LogConnectionID);
   XrdClientPhyConnection *GetPhyConnection(XrdClientUrlInfo server);

   XrdClientMessage*   
                 ReadMsg(int LogConnectionID);

   int           ReadRaw(int LogConnectionID, void *buffer, int BufferLength);
   int           WriteRaw(int LogConnectionID, const void *buffer, 
                          int BufferLength, int substreamid);

  XrdClientSid *SidManager() { return fSidManager; }

  friend int DisconnectElapsedPhyConn(const char *,
				      XrdClientPhyConnection *, void *);
  friend int DestroyPhyConn(const char *,
			    XrdClientPhyConnection *, void *);
};


#endif
