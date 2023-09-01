#ifndef _XrdClientPhyConnection
#define _XrdClientPhyConnection
/******************************************************************************/
/*                                                                            */
/*           X r d C l i e n t P h y C o n n e c t i o n . h h                */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/*  Alvise Dorigo, Fabrizio Furano                                            */
/*          INFN Padova, 2003                                                 */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Class handling physical connections to xrootd servers                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "XrdClient/XrdClientSock.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUnsolMsg.hh"
#include "XrdClient/XrdClientInputBuffer.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysSemWait.hh"

#include <time.h> // for time_t data type

enum ELoginState {
    kNo      = 0,
    kYes     = 1, 
    kPending = 2
};

enum ERemoteServerType {
    kSTError      = -1,  // Some error occurred: server type undetermined 
    kSTNone       = 0,   // Remote server type un-recognized
    kSTRootd      = 1,   // Remote server type: old rootd server
    kSTBaseXrootd = 2,   // Remote server type: xrootd dynamic load balancer
    kSTDataXrootd = 3,   // Remote server type: xrootd data server
    kSTMetaXrootd = 4    // Remote server type: xrootd meta manager
}; 

class XrdClientSid;
class XrdClientThread;
class XrdSecProtocol;

class XrdClientPhyConnection: public XrdClientUnsolMsgSender {

private:
    time_t              fLastUseTimestamp;
    enum ELoginState    fLogged;       // only 1 login/auth is needed for physical  
    XrdSecProtocol     *fSecProtocol;  // authentication protocol

    XrdClientInputBuffer
    fMsgQ;         // The queue used to hold incoming messages

    int                 fRequestTimeout;
    bool                fMStreamsGoing;
    XrdSysRecMutex         fRwMutex;     // Lock before using the physical channel 
    // (for reading and/or writing)

    XrdSysRecMutex         fMutex;
    XrdSysRecMutex         fMultireadMutex; // Used to arbitrate between multiple
                                            // threads reading msgs from the same conn

    XrdClientThread     *fReaderthreadhandler[64]; // The thread which is going to pump
    // out the data from the socket

    int                  fReaderthreadrunning;

    XrdClientUrlInfo          fServer;

    XrdClientSock       *fSocket;

    UnsolRespProcResult HandleUnsolicited(XrdClientMessage *m);

    XrdSysSemWait       fReaderCV;

    short               fLogConnCnt; // Number of logical connections using this phyconn

    XrdClientSid       *fSidManager;

public:
    long                fServerProto;        // The server protocol
    ERemoteServerType   fServerType;
    long                fTTLsec;

    XrdClientPhyConnection(XrdClientAbsUnsolMsgHandler *h, XrdClientSid *sid);
    ~XrdClientPhyConnection();

    XrdClientMessage     *BuildMessage(bool IgnoreTimeouts, bool Enqueue);
    bool                  CheckAutoTerm();

    bool           Connect(XrdClientUrlInfo RemoteHost, bool isUnix = 0);

    //--------------------------------------------------------------------------
    //! Connect to a remote location
    //!
    //! @param RemoteHost address descriptor
    //! @param isUnix     true if the address points to a Unix socket
    //! @param fd         a descriptor pointing to a connected socket
    //!                   if the subroutine is supposed to reuse an existing
    //!                   connection, -1 otherwise
    //--------------------------------------------------------------------------
    bool Connect( XrdClientUrlInfo RemoteHost, bool isUnix , int fd );

    void           CountLogConn(int d = 1);
    void           Disconnect();

    ERemoteServerType
    DoHandShake(ServerInitHandShake &xbody,
		int substreamid = 0);

    bool           ExpiredTTL();
    short          GetLogConnCnt() const { return fLogConnCnt; }
    int            GetReaderThreadsCnt() { XrdSysMutexHelper l(fMutex); return fReaderthreadrunning; }

    long           GetTTL() { return fTTLsec; }

    XrdSecProtocol *GetSecProtocol() const { return fSecProtocol; }
    int            GetSocket() { return fSocket ? fSocket->fSocket : -1; }

    // Tells to the sock to rebuild the list of interesting selectors
    void           ReinitFDTable() { if (fSocket) fSocket->ReinitFDTable(); }

    int            SaveSocket() { fTTLsec = 0; return fSocket ? (fSocket->SaveSocket()) : -1; }
    void           SetInterrupt() { if (fSocket) fSocket->SetInterrupt(); }
    void           SetSecProtocol(XrdSecProtocol *sp) { fSecProtocol = sp; }

    void           StartedReader();

    bool           IsAddress(const XrdOucString &addr) {
	return ( (fServer.Host == addr) ||
		 (fServer.HostAddr == addr) );
    }

    ELoginState    IsLogged();

    bool           IsPort(int port) { return (fServer.Port == port); };
    bool           IsUser(const XrdOucString &usr) { return (fServer.User == usr); };
    bool           IsValid();
    

    void           LockChannel();

    // see XrdClientSock for the meaning of the parameters
    int            ReadRaw(void *buffer, int BufferLength, int substreamid = -1,
			   int *usedsubstreamid = 0);

    XrdClientMessage     *ReadMessage(int streamid);
    bool           ReConnect(XrdClientUrlInfo RemoteHost);
    void           SetLogged(ELoginState status) { fLogged = status; }
    inline void    SetTTL(long ttl) { fTTLsec = ttl; }
    void           StartReader();
    void           Touch();
    void           UnlockChannel();
    int            WriteRaw(const void *buffer, int BufferLength, int substreamid = 0);

    int TryConnectParallelStream(int port, int windowsz, int sockid) { return ( fSocket ? fSocket->TryConnectParallelSock(port, windowsz, sockid) : -1); }
    int EstablishPendingParallelStream(int tmpid, int newid) { return ( fSocket ? fSocket->EstablishParallelSock(tmpid, newid) : -1); }
    void RemoveParallelStream(int substreamid) { if (fSocket) fSocket->RemoveParallelSock(substreamid); }
    // Tells if the attempt to establish the parallel streams is ongoing or was done
    //  and mark it as ongoing or done
    bool TestAndSetMStreamsGoing();

    int GetSockIdHint(int reqsperstream) { return ( fSocket ? fSocket->GetSockIdHint(reqsperstream) : 0); }
    int GetSockIdCount() {return ( fSocket ? fSocket->GetSockIdCount() : 0); }
    void PauseSelectOnSubstream(int substreamid) { if (fSocket) fSocket->PauseSelectOnSubstream(substreamid); }
    void RestartSelectOnSubstream(int substreamid) { if (fSocket) fSocket->RestartSelectOnSubstream(substreamid); }

    // To prohibit/re-enable a socket descriptor from being looked at by the reader threads
    virtual void BanSockDescr(int sockdescr, int sockid) { if (fSocket) fSocket->BanSockDescr(sockdescr, sockid); }
    virtual void UnBanSockDescr(int sockdescr) { if (fSocket) fSocket->UnBanSockDescr(sockdescr); }

    void ReadLock() { fMultireadMutex.Lock(); }
    void ReadUnLock() { fMultireadMutex.UnLock(); }

   int WipeStreamid(int streamid) { return fMsgQ.WipeStreamid(streamid); }
};




//
// Class implementing a trick to automatically unlock an XrdClientPhyConnection
//
class XrdClientPhyConnLocker {
private:
    XrdClientPhyConnection *phyconn;

public:
    XrdClientPhyConnLocker(XrdClientPhyConnection *phyc) {
	// Constructor
	phyconn = phyc;
	phyconn->LockChannel();
    }

    ~XrdClientPhyConnLocker(){
	// Destructor. 
	phyconn->UnlockChannel();
    }

};
#endif
