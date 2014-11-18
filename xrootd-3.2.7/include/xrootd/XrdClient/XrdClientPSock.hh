//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientPSock                                                       //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Client Socket with multiple streams and timeout features             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//           $Id$

#ifndef XRC_PSOCK_H
#define XRC_PSOCK_H

#include "XrdClient/XrdClientSock.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdOuc/XrdOucRash.hh"
#include "XrdSys/XrdSysPthread.hh"



struct fdinfo {
  fd_set fdset;
  int maxfd;
};



class XrdClientPSock: public XrdClientSock {

friend class XrdClientPhyConnection;

private:


    XrdSysRecMutex fMutex;

    // The set of interesting sock descriptors
    fdinfo globalfdinfo;

    Sockid lastsidhint;

    // To have a pool of the ids in use,
    // e.g. to select a random stream from the set of possible streams
    XrdClientVector<Sockid> fSocketIdRepo;
 
    // To translate from socket id to socket descriptor
    XrdOucRash<Sockid, Sockdescr> fSocketPool;

    // To keep track of the sockets which have to be 
    // temporarily ignored in the global fd
    // because they have not yet been handshaked
    XrdOucRash<Sockdescr, Sockid> fSocketNYHandshakedIdPool;

    Sockdescr GetSock(Sockid id) {
        XrdSysMutexHelper mtx(fMutex);

	Sockdescr *fd = fSocketPool.Find(id);
	if (fd) return *fd;
	else return -1;
    }

    Sockdescr GetMainSock() {
	return GetSock(0);
    }

    // To translate from socket descriptor to socket id
    XrdOucRash<Sockdescr, Sockid> fSocketIdPool;

    // From a socket descriptor, we get its id
    Sockid GetSockId(Sockdescr sock) {
        XrdSysMutexHelper mtx(fMutex);

	Sockid *id = fSocketIdPool.Find(sock);
	if (id) return *id;
	else return -1;
    }

protected:

    virtual int    SaveSocket() {
        XrdSysMutexHelper mtx(fMutex);

	// this overwrites the main stream
	Sockdescr *fd = fSocketPool.Find(0);

	fSocketIdPool.Del(*fd);
	fSocketPool.Del(0);

	fConnected = 0;
	fRDInterrupt = 0;
	fWRInterrupt = 0;

	if (fd) return *fd;
	else return 0;
    }

public:
    XrdClientPSock(XrdClientUrlInfo host, int windowsize = 0);
    virtual ~XrdClientPSock();
   
    void BanSockDescr(Sockdescr s, Sockid newid) { XrdSysMutexHelper mtx(fMutex); fSocketNYHandshakedIdPool.Rep(s, newid); }
    void UnBanSockDescr(Sockdescr s) { XrdSysMutexHelper mtx(fMutex); fSocketNYHandshakedIdPool.Del(s); }

    // Gets length bytes from the parsockid socket
    // If substreamid = -1 then
    //  gets length bytes from any par socket, and returns the usedsubstreamid
    //   where it got the bytes from
    virtual int    RecvRaw(void* buffer, int length, Sockid substreamid = -1,
			   Sockid *usedsubstreamid = 0);

    // Send the buffer to the specified substream
    // if substreamid == 0 then use the main socket
    virtual int    SendRaw(const void* buffer, int length, Sockid substreamid = 0);

    virtual void   TryConnect(bool isUnix = 0);

    virtual Sockdescr TryConnectParallelSock(int port, int windowsz, Sockid &tmpid);

    virtual int EstablishParallelSock(Sockid tmpsockid, Sockid newsockid);

    virtual void   Disconnect();

    virtual int RemoveParallelSock(Sockid sockid);

    // Suggests a sockid to be used for a req
    virtual Sockid GetSockIdHint(int reqsperstream);

    // And this is the total stream count
    virtual int GetSockIdCount() { 
        XrdSysMutexHelper mtx(fMutex);

        return fSocketPool.Num();
    }

    virtual void PauseSelectOnSubstream(Sockid substreamid);
    virtual void RestartSelectOnSubstream(Sockid substreamid);

};

#endif
