#ifndef __NET_LINK_H__
#define __NET_LINK_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d N e t L i n k . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#ifndef WIN32
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <fcntl.h>
#else
#include <Winsock2.h>
#endif

#include "XrdNet/XrdNetBuffer.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucChain.hh"
#include "XrdSys/XrdSysPthread.hh"

// Options for SetOpts and Alloc()
//
#define XRDNETLINK_NOBLOCK  0x0001
#define XRDNETLINK_NOCLOSE  0x0002
#define XRDNETLINK_NOSTREAM 0x0004

// The XrdNetLink class defines the i/o operations on a network link.
//
class XrdNet;
class XrdNetPeer;
class XrdSysError;
class XrdOucStream;
class XrdOucTokenizer;

class XrdNetLink
{
public:

XrdOucQSItem<XrdNetLink> LinkLink;

static XrdNetLink *Alloc(XrdSysError   *erp, XrdNet *Net, XrdNetPeer &Peer,
                         XrdNetBufferQ *bq,  int opts=0);

// Closes() closes the link. Specify defer=1 to postpone deallocating
//          attached objects until the this object is destroyed. You should
//          use defered close for cross-thread unsynchronized closes.
//
int           Close(int defer=0);

// FDnum() returns the associated file descriptor
//
int           FDnum() {return FD;}

// The following implement text-oriented read routines. The correspond to
// those implemented by the XrdOucStream object which is the one used.
//
char         *GetLine();

char         *GetToken(char **rest);
char         *GetToken(void);
void          RetToken(void);

// isConnected() returns true if this object is connected to an XrdOucStream
//               object and false otherwise.
//
int           isConnected(void) {return (Stream != 0) && (FD >= 0);}

// LastError() returns the error number associated with the last error
//
int           LastError();

// Addr() returns the IPV4 address of the endpoint
//
unsigned int Addr() {return XrdNetDNS::IPAddr(&InetAddr);}

// Moniker() returns the short form of the host name at the endpoint
//
const char  *Moniker() {return Sname;}

// Name() returns the full host name of the endpoint
//
const char  *Name() {return Lname;}

// Moniker() returns the short form of the host name at the endpoint
//
const char  *Nick() {return Sname;}

// OK2Recv() returns true if data can be received within tmo, false otherwise.
//
int           OK2Recv(int mills);

// Recycle() makes this object available for reuse.
//
void          Recycle();

// Send() set of methods that accept a char buffer are text oriented send
//        routines. They all add a new-line (\n) character to end the buffer
//        if it does not exist already.
//
int           Send(const char *buff,     // -> Data to send
                         int   blen=0,   // Length. If 0, it's compued via strlen()
                         int   tmo=-1);  // Millisecond timeout (default is none)

int           Send(const char *dest,     // -> Hostname to send UDP datagram
                   const char *buff,     // Remaining parms as above
                         int   blen=0,
                          int   tmo=-1);

int           Send(const struct iovec iov[], // writev() style plist
                   int iovcnt,               // Number of elements om iov[]
                   int tmo = -1);            // Optional timeout

int           Send(const char   *dest,       // Hostname to send UDP datagram
                   const struct iovec iov[], // Remaining parms as above
                   int   iovcnt,
                   int   tmo=-1);

// Send() set of methods that accept a void buffer are byte oriented send
//        routines. These do not inspect the data at all.
//
int           Send(const void *buff,     // -> Data to send
                         int   blen=0,   // Length. If 0, it's compued via strlen()
                         int   tmo=-1);  // Millisecond timeout (default is none)

// Recv() receives up to blen bytes. It may receive less than that if
//        additional bytes are not immediately available to receive.
//        it returns the number of bytes read or -1 if an error occurs.
//
int           Recv(char *buff, int blen);

// Set() sets the maximum number of XrdNetLink objects that may be kept
//       for future re-use.
//
void          Set(int maxl);

// SetOpts() is used to set socket options, as defined above.
//
void          SetOpts(int opts);

// Instantiate this object with the pointer to an error object for message
// routing. Additionally, a pointer to a UDP buffer may be provided. This
// buffer must contain a text datagram suitable for tokenization.
//
              XrdNetLink(XrdSysError *erp, XrdNetBufferQ *bq) : LinkLink(this)
                          {FD = -1; Lname = Sname = 0; recvbuff = sendbuff = 0;
                           BuffQ = bq; Stream = 0; Bucket = 0; eDest = erp;
                          }
             ~XrdNetLink() {Close();}

private:

int OK2Send(int timeout=0, const char *dest=0);
int retErr(int ecode, const char *dest=0);

XrdSysMutex         rdMutex;
XrdSysMutex         wrMutex;
XrdNetBufferQ      *BuffQ;
int                 FD;
int                 noclose;
int                 isReset;
struct sockaddr     InetAddr;
char               *Lname;     // Long  hostname
char               *Sname;     // Short hostname (may be the same as Lname)
XrdNetBuffer       *recvbuff;  // udp receive buffer
XrdNetBuffer       *sendbuff;  // udp send    buffer
XrdOucStream       *Stream;    // tcp tokenizer
XrdOucTokenizer    *Bucket;    // udp tokenizer
XrdSysError        *eDest;

static XrdSysMutex             LinkList;
static XrdOucStack<XrdNetLink> LinkStack;
static int                     size;
static int                     maxlink;
static int                     numlink;
static int                     devNull;
};
#endif
