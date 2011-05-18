/******************************************************************************/
/*                                                                            */
/*                       X r d N e t S o c k e t . c c                        */
/*                                                                            */
/* (C) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                          All Rights Reserved                               */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Deprtment of Energy               */
/******************************************************************************/

//         $Id$

const char *XrdNetSocketCVSID = "$Id$";

#ifndef WIN32
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <Winsock2.h>
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdNet/XrdNetConnect.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                         l o c a l   d e f i n e s                          */
/******************************************************************************/
  
#define XRDNETSOCKET_MAXBKLG 255
#define XRDNETSOCKET_LINGER    3

#define Err(p,a,b,c) (ErrCode = (eroute ? eroute->Emsg(#p, a, b, c) : ErrCode),-1)

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdNetSocket::XrdNetSocket(XrdSysError *erobj, int SockFileDesc)
{
   ErrCode  = 0;
   PeerName = 0;
   SockFD   = SockFileDesc;
   eroute   = erobj;
}

/******************************************************************************/
/*                                A c c e p t                                 */
/******************************************************************************/
  
int XrdNetSocket::Accept(int timeout)
{
   int retc, ClientSock;

   ErrCode = 0;

   // Check if a timeout was requested
   //
   if (timeout >= 0)
      {struct pollfd sfd = {SockFD,
                            POLLIN|POLLRDNORM|POLLRDBAND|POLLPRI|POLLHUP, 0};
       do {retc = poll(&sfd, 1, timeout);}
                  while(retc < 0 && (errno == EAGAIN || errno == EINTR));
       if (!sfd.revents) return -1;
      }

   do {ClientSock = accept(SockFD, (struct sockaddr *)0, 0);}
      while(ClientSock < 0 && errno == EINTR);

   if (ClientSock < 0 && eroute) eroute->Emsg("Accept",errno,"accept connection");

   // Return the socket number.
   //
   return ClientSock;
}

/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/

void XrdNetSocket::Close()
{
     // Close any open file descriptor.
     //
     if (SockFD >= 0) {close(SockFD); SockFD=-1;} 

     // Free any previous peername pointer
     //
     if (PeerName) {free(PeerName); PeerName = 0;}

     // Reset values and return.
     //
     ErrCode=0;
}

/******************************************************************************/
/*                                C r e a t e                                 */
/******************************************************************************/

XrdNetSocket *XrdNetSocket::Create(XrdSysError *Say, const char *path, 
                                   const char *fn, mode_t mode, int opts)
{
   XrdNetSocket *ASock;
   int pflags    = (opts & XRDNET_FIFO ? S_IFIFO : S_IFSOCK);
   int sflags    = (opts & XRDNET_UDPSOCKET) | XRDNET_SERVER;
   int rc = 0;
   mode_t myMode = (mode & (S_IRWXU | S_IRWXG));
   const char *eMsg = 0;
   char fnbuff[1024] = {0};

// Setup the path
//
   if (!socketPath(Say, fnbuff, path, fn, mode|pflags))
      return (XrdNetSocket *)0;

// Connect to the path
//
   ASock = new XrdNetSocket(Say);
#ifndef WIN32
   if (opts & XRDNET_FIFO)
      {if ((ASock->SockFD = mkfifo(fnbuff, mode)) < 0 && errno != EEXIST)
         {eMsg = "create fifo"; rc = errno;}
         else if ((ASock->SockFD = open(fnbuff, O_RDWR, myMode)) < 0)
                 {eMsg = "open fifo"; rc = ASock->LastError();}
      } else if (ASock->Open(fnbuff, -1, sflags) < 0) 
                {eMsg = "create socket"; rc = ASock->LastError();}
#else
   if (ASock->Open(fnbuff, -1, sflags) < 0)
      {eMsg = "create socket"; rc = ASock->LastError();}
#endif

// Return the result
//
   if (eMsg) {Say->Emsg("Create", rc, eMsg, fnbuff);
              delete ASock; ASock = 0;
             }
   return ASock;
}

/******************************************************************************/
/*                                D e t a c h                                 */
/******************************************************************************/
  
int XrdNetSocket::Detach()
{  int oldFD = SockFD;
   SockFD = -1;
   if (PeerName) {free(PeerName); PeerName = 0;}
   return oldFD;
}

/******************************************************************************/
/*                             g e t W i n d o w                              */
/******************************************************************************/
  
int XrdNetSocket::getWindow(int fd, int &Windowsz, XrdSysError *eDest)
{
   socklen_t szb = (socklen_t)sizeof(Windowsz);

   if (getsockopt(fd, SOL_SOCKET, SO_RCVBUF, (Sokdata_t)&Windowsz, &szb))
      {if (eDest) eDest->Emsg("setWindow", errno, "set socket RCVBUF");
       return -1;
      }
    return 0;
}

/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/
  
int XrdNetSocket::Open(const char *inpath, int port, int flags, int windowsz)
{
   struct sockaddr_in InetAddr;
   struct sockaddr_un UnixAddr;
   struct sockaddr *SockAddr;
   char *errtxt = 0, pbuff[128];
   const char *action = "configure socket";
   const char *path = (inpath ? inpath : "");
   const char *epath= (inpath ? inpath : pbuff);
   int myEC, SockSize, backlog;
   int SockType = (flags & XRDNET_UDPSOCKET ? SOCK_DGRAM : SOCK_STREAM);
   const int one = 1;
   const SOCKLEN_t szone = (SOCKLEN_t)sizeof(one);

// Supply actual port number in error messages
//
   if (!inpath) sprintf(pbuff, "port %d", port);

// Make sure this object is available for a new socket
//
   if (SockFD >= 0) return Err(Open, EBUSY, "create socket for", epath);

// Save the request flags, sometimes we need to check them from the local copy
//
   myEC = ErrCode = 0;

// Allocate a socket descriptor and bind connection address, if requested.
// In most OS's we must unlink the corresponding Unix path name or bind will
// fail. In some OS's, this creates a problem (e.g., Solaris) since the
// file inode is used to identify the socket and will likely change. This
// means that connects occuring before the bind will hang up to 3 minutes.
// So, the client better be prepared to timeout connects and try again.
//
   if (port < 0 && *path == '/')
      {if (strlen(path) >= sizeof(UnixAddr.sun_path))
          return Err(Open, ENAMETOOLONG, "create unix socket ", epath);
       if ((SockFD = socket(PF_UNIX, SockType, 0)) < 0)
          return Err(Open, errno, "create unix socket ", epath);
       UnixAddr.sun_family = AF_UNIX;
       strcpy(UnixAddr.sun_path, path);
       SockAddr = (struct sockaddr *)&UnixAddr;
       SockSize = sizeof(UnixAddr);
       if (flags & XRDNET_SERVER) unlink((const char *)path);
      } else {
       if ((SockFD = socket(PF_INET, SockType, 0)) < 0)
          return Err(Open, errno, "create inet socket to", epath);
       if (port < 0 && *path)
                     XrdNetDNS::Host2Dest(inpath,(sockaddr &)InetAddr,&errtxt);
             else   {XrdNetDNS::getHostAddr(path,(sockaddr &)InetAddr,&errtxt);
                     XrdNetDNS::setPort((sockaddr &)InetAddr, port);
                    }
          if (errtxt)
             {if(eroute) eroute->Emsg("Open", "Unable to obtain address for",
                                     epath, errtxt);
              Close();
              ErrCode = EHOSTUNREACH;
              return -1;
             }
       SockAddr = (struct sockaddr *)&InetAddr;
       SockSize = sizeof(InetAddr);
      }

// Set the options and window size, as needed (ignore errors)
//
   setOpts(SockFD, (flags | (*path == '/' ? XRDNET_UDPSOCKET : 0)), eroute);
   if (windowsz) setWindow(SockFD, windowsz, eroute);

// Make sure the local endpoint can be reused
//
   if ((*path != '/') && setsockopt(SockFD,SOL_SOCKET,SO_REUSEADDR,
                                   (Sokdata_t)&one, szone) && eroute)
      eroute->Emsg("open", errno, "set socket REUSEADDR");

// Either do a connect or a bind.
//
   if (flags & XRDNET_SERVER)
      {action = "bind socket to";
       if ( bind(SockFD, SockAddr, SockSize) ) myEC = errno;
          else if (SockType == SOCK_STREAM)
                  {action = "listen on stream";
                   if (!(backlog = flags & XRDNET_BKLG))
                      backlog = XRDNETSOCKET_MAXBKLG;
                   if (listen(SockFD, backlog)) myEC = errno;
                  }
       if (*path == '/') chmod(path, S_IRWXU);
      } else {
       if (SockType == SOCK_STREAM)
          {int tmo = flags & XRDNET_TOUT;
           action = "connect socket to";
           if (tmo) myEC = XrdNetConnect::Connect(SockFD,SockAddr,SockSize,tmo);
              else if (connect(SockFD, SockAddr, SockSize)) myEC = errno;
          }
       if (!myEC) {PeerName = strdup((path ? path : "?"));
                   if (*path == '/') XrdNetDNS::getHostAddr(0, PeerAddr);
                      else memcpy((void *)&PeerAddr,SockAddr,sizeof(PeerAddr));
                  }
      }

// Check for any errors and return.
//
   if (myEC)
      {Close(); 
       ErrCode = myEC;
       if (!(flags & XRDNET_NOEMSG) && eroute)
          eroute->Emsg("Open", ErrCode, action, epath);
       return -1;
      }
   return SockFD;
}

/******************************************************************************/
/*                              P e e r n a m e                               */
/******************************************************************************/
  
const char *XrdNetSocket::Peername(struct sockaddr **InetAddr)
{
   char *errtxt;

// Make sure  we have something to look at
//
   if (SockFD < 0) 
      {if (eroute) eroute->Emsg("Peername", 
                                "Unable to obtain peer name; socket not open");
       return (char *)0;
      }

// Get the host name on the other side of this socket
//
   if (!PeerName 
   &&  !(PeerName = XrdNetDNS::Peername(SockFD, &PeerAddr, &errtxt)))
      {if (eroute) 
          eroute->Emsg("Peername", "Unable to obtain peer name;",errtxt);
       ErrCode = ESRCH;
      }

// Return possible address and the name
//
     if (InetAddr) *InetAddr = &PeerAddr;
     return PeerName;
}

/******************************************************************************/
/*                               s e t O p t s                                */
/******************************************************************************/
  
int XrdNetSocket::setOpts(int xfd, int opts, XrdSysError *eDest)
{
   int rc = 0;
   const int one = 1;
   const SOCKLEN_t szone = (SOCKLEN_t)sizeof(one);
   static int tcpprotid = XrdNetDNS::getProtoID("tcp");
   static struct linger liopts = {1, XRDNETSOCKET_LINGER};
   const SOCKLEN_t szlio = (SOCKLEN_t)sizeof(liopts);

   if (!(opts & XRDNET_NOCLOSEX) && fcntl(xfd, F_SETFD, FD_CLOEXEC))
      {rc = 1;
       if (eDest) eDest->Emsg("setOpts", errno, "set fd close on exec");
      }

   if (opts & XRDNET_UDPSOCKET) return rc;

   if (!(opts & XRDNET_NOLINGER)
   &&  setsockopt(xfd,SOL_SOCKET,SO_LINGER,(Sokdata_t)&liopts,szlio))
      {rc = 1;
       if (eDest) eDest->Emsg("setOpts", errno, "set socket LINGER");
      }

   if ((opts & XRDNET_KEEPALIVE)
   &&  setsockopt(xfd,SOL_SOCKET,SO_KEEPALIVE,(Sokdata_t)&one,szone))
      {rc = 1;
       if (eDest) eDest->Emsg("setOpts", errno, "set socket KEEPALIVE");
      }

   if (!(opts & XRDNET_DELAY)
   &&  setsockopt(xfd, tcpprotid, TCP_NODELAY, (Sokdata_t)&one,szone))
      {rc = 1;
       if (eDest) eDest->Emsg("setOpts", errno, "set socket NODELAY");
      }

   return rc;
}

/******************************************************************************/
/*                             s e t W i n d o w                              */
/******************************************************************************/
  
int XrdNetSocket::setWindow(int xfd, int Windowsz, XrdSysError *eDest)
{
   int rc = 0;
   const SOCKLEN_t szwb  = (SOCKLEN_t)sizeof(Windowsz);

   if (setsockopt(xfd, SOL_SOCKET, SO_SNDBUF,
                       (Sokdata_t)&Windowsz, szwb))
      {rc = 1;
       if (eDest) eDest->Emsg("setWindow", errno, "set socket SNDBUF");
      }

   if (setsockopt(xfd, SOL_SOCKET, SO_RCVBUF,
                       (Sokdata_t)&Windowsz, szwb))
      {rc = 1;
       if (eDest) eDest->Emsg("setWindow", errno, "set socket RCVBUF");
      }
   return rc;
}

/******************************************************************************/
/*                            s o c k e t A d d r                             */
/******************************************************************************/
  
const char *XrdNetSocket::socketAddr(XrdSysError *Say, const char *dest,
                                     struct sockaddr **sockAP, int &sockAL)
{
   struct sockaddr_in InetAddr;
   struct sockaddr_un UnixAddr;
   struct sockaddr *SockAddr;
   char *errtxt = 0;
   int port = 0, SockSize;

   if (*dest == '/')
      {if (strlen(dest) >= sizeof(UnixAddr.sun_path))
          {if (Say) Say->Emsg("Net", ENAMETOOLONG, "generate addr from", dest);
           return "socket address path too long";
          }
       UnixAddr.sun_family = AF_UNIX;
       strcpy(UnixAddr.sun_path, dest);
       SockAddr = (struct sockaddr *)&UnixAddr;
       SockSize = sizeof(UnixAddr);
      } else {
       if (*dest && *dest != ':')
                     XrdNetDNS::Host2Dest(dest,(sockaddr &)InetAddr,&errtxt);
          else      {XrdNetDNS::getHostAddr(dest,(sockaddr &)InetAddr,&errtxt);
                     if (*dest && *dest == ':') port = atoi(dest+1);
                     XrdNetDNS::setPort((sockaddr &)InetAddr, port);
                    }
       if (errtxt)
          {if (Say) Say->Emsg("Net","Unable to obtain address for",dest,errtxt);
           return errtxt;
          }
       SockAddr = (struct sockaddr *)&InetAddr;
       SockSize = sizeof(InetAddr);
      }

    sockAL = SockSize;
   *sockAP = (struct sockaddr *)malloc(SockSize);
   memcpy(*sockAP, SockAddr, SockSize);
   return 0;
}

/******************************************************************************/
/*                            s o c k e t P a t h                             */
/******************************************************************************/

char *XrdNetSocket::socketPath(XrdSysError *Say, char *fnbuff,
                               const char *path, const char *fn, mode_t mode)
{
   const int srchOK = S_IXUSR | S_IXGRP;
   const int sfMask = (S_IFIFO | S_IFSOCK);
   int rc, i, fnlen = strlen(fnbuff);
   mode_t myMode = (mode & (S_IRWXU | S_IRWXG)) | srchOK;
   struct stat buf;
   char *sp = 0;

// Copy the const char path because makePath modifies it
//
   i = strlen(path);
   if (strlcpy(fnbuff, path, 1024) >= 1024 || (i + fnlen + 1) >= 1024)
      {Say->Emsg("createPath", "Socket path", path, "too long");
       return 0;
      }

// Check if we should separate the filename from the path
//
   if (!fn)
      {if (fnbuff[i-1] == '/') fnbuff[i-1] = '\0';
       if ((sp = rindex(fnbuff, '/'))) *sp = '\0';
      }
   
// Create the directory if it is not already there
//
   if ((rc = XrdOucUtils::makePath(fnbuff, myMode)))
      {Say->Emsg("createPath", errno, "create path", path);
       return 0;
      }

// Construct full filename
//
   if (sp) *sp = '/';
      else {if (path[i-1] != '/') fnbuff[i++] = '/';
            if (fn) strcpy(fnbuff+i, fn);
           }

// Check is we have already created it and whether we can access
//
   if (!stat(fnbuff,&buf))
      {if ((buf.st_mode & S_IFMT) != (mode & sfMask))
          {Say->Emsg("createPath","Path",fnbuff,
                     (mode & S_IFSOCK) ? "exists but is not a socket"
                                       : "exists but is not a pipe");
           return 0;
          }
       if (access(fnbuff, W_OK))
          {Say->Emsg("cratePath", errno, "access path", fnbuff);
           return 0;
          }
      } else chmod(fnbuff, mode); // This may fail on some platforms

// All set now
//
   return fnbuff;
}
