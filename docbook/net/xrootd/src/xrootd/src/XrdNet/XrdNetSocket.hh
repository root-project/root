#ifndef __NETSOCKET__
#define __NETSOCKET__
/******************************************************************************/
/*                                                                            */
/*                       X r d N e t S o c k e t . h h                        */
/*                                                                            */
/* (C) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/
 
//         $Id$
  
#ifndef WIN32
#include <sys/socket.h>
#else
#include <Winsock2.h>
#endif

/******************************************************************************/
/*                      C l a s s   D e f i n i t i o n                       */
/******************************************************************************/
  
class XrdSysError;

class XrdNetSocket
{
public:

// When creating a socket object, you may pass an optional error routing object.
// If you do so, error messages will be writen via the error object. Otherwise,
// errors will be returned quietly. Addionally, you can attach a file descriptor
// to the socket object. This is useful when creating an object for accepted
// connections, e.g., ClientSock = new XrdNetSocket("", ServSock.Accept()).
//
            XrdNetSocket(XrdSysError *erobj=0, int SockFileDesc=-1);

           ~XrdNetSocket() {Close();}

// Create a named socket. Returns a NetSocket object that can be used for the
// given path. A udp or tcp socket can be created on the path with the given
// file name. The access permission mode must also be supplied. Upon failure,
// a null pointer is returned.
//
static XrdNetSocket *Create(XrdSysError *Say, const char *path,
                            const char *fn, mode_t mode, int isudp=0);

// Open a socket. Returns socket number upon success otherwise a -1. Use
// LastError() to find out the reason for failure. Only one socket at a time
// may be created. Use Close() to close the socket of Detach() to remove
// the socket association before creating a new one.

//         |<-------- C l i e n t -------->|  |<-------- S e r v e r -------->|
//         Unix Socket       Internet Socket  Unix Socket       Internet Socket
// path  = Filname           hostname.        filename          0 or ""
// port  = -1                port number      -1                port number
// flags = ~XRDNET_SERVER    ~XRDNET_SERVER   XRDNET_SERVER     XRDNET_SERVER

// If the client path does not start with a slash and the port number is -1
// then hostname must be of the form hostname:port. Open() will always set
// the REUSEADDR option when binding to a port number.
//
       int  Open(const char *path, int port=-1, int flags=0, int sockbuffsz=0);

// Issue accept on the created socket. Upon success return socket FD, upon
// failure return -1. Use LastError() to obtain reason for failure. Note that
// Accept() is valid only for Server Sockets. An optional millisecond
// timeout may be specified. If no new connection is attempted within the
// millisecond time limit, a return is made with -1 and an error code of 0.
// Accept() always sets the "close on exec" flag for the new fd.
//
       int  Accept(int ms=-1);

// Close a socket.
//
       void Close();

// Detach the socket filedescriptor without closing it. Useful when you
// will be attaching the descriptor to a stream. Returns the descriptor so
// you can do something like Stream.Attach(Socket.Detach()).
//
       int  Detach();

// Return last errno.
//
inline int  LastError() {return ErrCode;}

// Obtain the name of the host on the other side of a socket. Upon success,
// a pointer to the hostname is returned. Otherwise null is returned. An
// optional address for holding the vided to obtain the hostname for it.
// The string is strdup'd and is deleted when the socket object is deleted.
//
const char *Peername(struct sockaddr **InetAddr=0);

// Set socket options (see definitions in XrdNetOpts.hh). The defaults
// defaults are such that each option must be set to override the default
// behaviour. The method is static so it can be used in any context. 
// An optional error routing object may be specified if error messages are 
// wanted. Only when all option settings succeed is 0 is returned.
//
static int setOpts(int fd, int options, XrdSysError *eDest=0);

// Set socket recv/send buffer sizes. The method is static so it can be used in 
// any context. An optional error routing object may be specified if error 
// messages are wanted. Only when all option settings succeed is 0 is returned.
//
static int setWindow(int fd, int  Windowsz, XrdSysError *eDest=0);

static int getWindow(int fd, int &Windowsz, XrdSysError *eDest=0);

// Return socket file descriptor number (useful when attaching to a stream).
//
inline int  SockNum() {return SockFD;}

// Create an appropriate sockaddr structure for the supplied path which is
// either a hostname:port or a unix path. If successful, 0 is returned
// otherwise a const error message is returned. The address of the sockaddr
// is returned in sockAP and it's size is returned in sockAL upon success.
//
static const char *socketAddr(XrdSysError *Say, const char *dest,
                              struct sockaddr **sockAP, int &sockAL);

// Create a path to a named socket returning the actual name of the socket.
// This method does not actually create the socket, only the path to the
// socket. If the full path exists then it must be a named socket. Upon
// success, it returns a pointer to the buffer holding the name (supplied by
// the caller). Otherwise, it returns a null pointer.
//
static char *socketPath(XrdSysError *Say, char *inbuff,
                        const char *path, const char *fn, 
                        mode_t mode);

/******************************************************************************/
  
private:
int             SockFD;
int             ErrCode;
struct sockaddr PeerAddr;
char           *PeerName;
XrdSysError    *eroute;
};
#endif
