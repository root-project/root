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

#define XRDCLI_PSOCKTEMP -2

struct fdinfo {
  fd_set fdset;
  int maxfd;
};

class XrdClientPSock: public XrdClientSock {

friend class XrdClientPhyConnection;

private:
    typedef int       Sockid;
    typedef int       Sockdescr;

    XrdSysRecMutex fMutex;

    // The set of interesting sock descriptors
    fdinfo globalfdinfo;

    int lastsidhint;

    // To have a pool of the ids in use,
    // e.g. to select a random stream from the set of possible streams
    XrdClientVector<Sockid> fSocketIdRepo;
 
    // To translate from socket id to socket descriptor
    XrdOucRash<Sockid, Sockdescr> fSocketPool;

    Sockdescr GetSock(Sockid id) {
        XrdSysMutexHelper mtx(fMutex);

	Sockdescr *fd = fSocketPool.Find(id);
	if (fd) return *fd;
	else return -1;
    }
    int GetMainSock() {
	return GetSock(0);
    }

    // To translate from socket descriptor to socket id
    XrdOucRash<Sockdescr, Sockid> fSocketIdPool;


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
	int *fd = fSocketPool.Find(0);

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

    // Gets length bytes from the parsockid socket
    // If substreamid = -1 then
    //  gets length bytes from any par socket, and returns the usedsubstreamid
    //   where it got the bytes from
    virtual int    RecvRaw(void* buffer, int length, int substreamid = -1,
			   int *usedsubstreamid = 0);


    // Send the buffer to the specified substream
    // if substreamid == 0 then use the main socket
    virtual int    SendRaw(const void* buffer, int length, int substreamid = 0);

    virtual void   TryConnect(bool isUnix = 0);

    virtual int TryConnectParallelSock(int port = 0, int windowsz = 0);

    virtual int EstablishParallelSock(int sockid);

    virtual void   Disconnect();

    virtual int RemoveParallelSock(int sockid);

    // Suggests a sockid to be used for a req
    virtual int GetSockIdHint(int reqsperstream);

    // And this is the total stream count
    virtual int GetSockIdCount() { 
        XrdSysMutexHelper mtx(fMutex);

        return fSocketPool.Num();
    }

    virtual void PauseSelectOnSubstream(int substreamid);
    virtual void RestartSelectOnSubstream(int substreamid);

};

#endif
