//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientSock                                                        //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Client Socket with timeout features                                  //
//                                                                      //
// June 06 - Fabrizio Furano                                            //
// The function prototypes allow specializations for multistream xfer   //
//  purposes. In this class only monostream xfers are allowed.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//           $Id$

#ifndef XRC_SOCK_H
#define XRC_SOCK_H

#include <XrdClient/XrdClientUrlInfo.hh>

struct XrdClientSockConnectParms {
    XrdClientUrlInfo TcpHost;
    int TcpWindowSize;
};

class XrdClientSock {

    friend class XrdClientPhyConnection;

private:

    int fSocket;

protected:
    int                       fRequestTimeout;
    XrdClientSockConnectParms fHost;

    bool                      fConnected;
    bool                      fRDInterrupt;
    bool                      fWRInterrupt;

    // Tells if we have to reinit the table of the fd selectors
    // after adding or removing one of them
    bool                      fReinit_fd;

    virtual int    SaveSocket() { int fd = fSocket; fSocket = -1;
	fConnected = 0; fRDInterrupt = 0; fWRInterrupt = 0; return fd; }

    void   SetInterrupt(int which = 0) { if (which == 0 || which == 1) fRDInterrupt = 1;
                                         if (which == 0 || which == 2) fWRInterrupt = 1; }

    // returns the socket descriptor or -1
    int   TryConnect_low(bool isUnix = 0, int altport = 0, int windowsz = 0);

    // Send the buffer to the specified socket
    virtual int    SendRaw_sock(const void* buffer, int length, int sock);
public:
    XrdClientSock(XrdClientUrlInfo host, int windowsize = 0);
    virtual ~XrdClientSock();

    // Makes a pending recvraw to rebuild the list of interesting selectors
    void           ReinitFDTable() { fReinit_fd = true; }

    // Gets length bytes from the specified substreamid
    // If substreamid = 0 then use the main stream
    // If substreamid = -1 then
    //  use any stream which has something pending
    //  and return its id in usedsubstreamid
    // Note that in this base class only the multistream intf is provided
    //  but the implementation is monostream
    virtual int    RecvRaw(void* buffer, int length, int substreamid = -1,
			   int *usedsubstreamid = 0);

    // Send the buffer to the specified substream
    // if substreamid <= 0 then use the main one
    virtual int    SendRaw(const void* buffer, int length, int substreamid = 0);

    void   SetRequestTimeout(int timeout = -1);

    // Performs a SOCKS4 handshake in a given stream
    // Returns the handshake result
    // If successful, we are connected through a socks4 proxy
    virtual int Socks4Handshake(int sockid);

    virtual void   TryConnect(bool isUnix = 0);

    // Returns a temporary socket id or -1
    // The temporary sock id XRDCLI_PSOCKTEMP is to be used to send the kxr_bind_request
    // If all this goes ok, then the caller must call EstablishParallelSock, otherwise the
    //  creation of parallel streams should be aborted (but the already created streams are OK)
    virtual int TryConnectParallelSock(int /*port*/ = 0, int /*windowsz*/ = 0) { return -1; }

    // Attach the pending (and hidden) sock associated to the substreamid XRDCLI_PSOCKTEMP
    //  to the given substreamid
    // the given substreamid could be an integer suggested by the server
    virtual int EstablishParallelSock(int /*sockid*/) { return -1; }

    virtual int RemoveParallelSock(int /* sockid */) { return -1; };

    // Suggests a sockid to be used for a req
    virtual int GetSockIdHint(int /* reqsperstream */ ) { return 0; }

    virtual void   Disconnect();

    bool   IsConnected() {return fConnected;}
    virtual int GetSockIdCount() { return 1; }
    virtual void PauseSelectOnSubstream(int /* substreamid */) {  }
    virtual void RestartSelectOnSubstream(int /*substreamid */) {  }
};

#endif
