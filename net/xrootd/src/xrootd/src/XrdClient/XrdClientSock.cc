//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientSock                                                        //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Client Socket with timeout features using XrdNet                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientSockCVSID = "$Id$";

#include <memory>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "XrdClient/XrdClientSock.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"

#ifndef WIN32
#include <netinet/in.h>
#include <unistd.h>
#include <sys/poll.h>
#else
#include "XrdSys/XrdWin32.hh"
#endif

#ifdef __FreeBSD__
#include <sys/types.h>
#include <pwd.h>
#endif

//_____________________________________________________________________________
XrdClientSock::XrdClientSock(XrdClientUrlInfo Host, int windowsize)
{
    // Constructor

    fHost.TcpHost = Host;
    fHost.TcpWindowSize = windowsize;
    fConnected = FALSE;
    fRDInterrupt = 0;
    fWRInterrupt = 0;
    fSocket = -1;
    fRequestTimeout = EnvGetLong(NAME_REQUESTTIMEOUT);
}

//_____________________________________________________________________________
XrdClientSock::~XrdClientSock()
{
    // Destructor
    Disconnect();
}

//_____________________________________________________________________________
void XrdClientSock::SetRequestTimeout(int timeout)
{
   // Set request timeout. If timeout is non-positive reset the request
   // timeout to the default value

   fRequestTimeout = (timeout > 0) ? timeout : EnvGetLong(NAME_REQUESTTIMEOUT);
}

//_____________________________________________________________________________
void XrdClientSock::Disconnect()
{
    // Close the connection
//     if (fConnected && fSocket >= 0) {
// 	::close(fSocket);
// 	fConnected = FALSE;
// 	fSocket = -1;
//     }
    if (fSocket >= 0) {
	::close(fSocket);
    }

    fConnected = false;
    fSocket = -1;

}

//_____________________________________________________________________________
int XrdClientSock::RecvRaw(void* buffer, int length, int substreamid,
			   int *usedsubstreamid)
{
    // Read bytes following carefully the timeout rules
    struct pollfd fds_r;
    int bytesread = 0;
    int pollRet;

    // We cycle reading data.
    // An exit occurs if:
    // We have all the data we are waiting for
    // Or a timeout occurs
    // The connection is closed by the other peer

    // Init of the pollfd struct
    if (fSocket < 0) {
       Error("XrdClientSock::RecvRaw", "socket fd is " << fSocket);
       return TXSOCK_ERR;
    }

    fds_r.fd     = fSocket;
    //   fds_r.events = POLLIN | POLLPRI | POLLERR | POLLHUP | POLLNVAL;
    fds_r.events = POLLIN;

    while (bytesread < length) {

        // We cycle on the poll, ignoring the possible interruptions
        // We are waiting for something to come from the socket

      // We cycle on the poll, ignoring the possible interruptions
      // We are waiting for something to come from the socket,
      // but we will not wait forever
      int timeleft = fRequestTimeout;
      do {
         // Wait for some event from the socket
         pollRet = poll(&fds_r,
                        1,
                        1000 // 1 second as a step
                        );

         if ((pollRet < 0) && (errno != EINTR) && (errno != EAGAIN) )
            return TXSOCK_ERR;

      } while (--timeleft && pollRet <= 0 && !fRDInterrupt);


      // If we are here, pollRet is > 0 why?
      //  Because the timeout and the poll error are handled inside the previous loop

      if (fSocket < 0) {
         if (fConnected) {
            Error("XrdClientSock::RecvRaw", "since we entered RecvRaw, socket "
                  "file descriptor has changed to " << fSocket);
            fConnected = FALSE;
         }
         return TXSOCK_ERR;
      }

      // If we have been timed-out, return a specific error code
      if (timeleft <= 0) {
         if ((DebugLevel() >= XrdClientDebug::kDUMPDEBUG))
            Info(XrdClientDebug::kNODEBUG,
                 "ClientSock::RecvRaw",
                 "Request timed out "<< fRequestTimeout << 
                 "seconds reading " << length << " bytes" <<
                 " from server " << fHost.TcpHost.Host <<
                 ":" << fHost.TcpHost.Port);
         return TXSOCK_ERR_TIMEOUT;
      }

      // If we have been interrupt, reset the inetrrupt and exit
      if (fRDInterrupt) {
         fRDInterrupt = 0;
         Error("XrdClientSock::RecvRaw", "got interrupt");
         return TXSOCK_ERR_INTERRUPT;
      }


      // First of all, we check if there is something to read
      if (fds_r.revents & (POLLIN | POLLPRI)) {
         int n = 0;

         do {
         n = ::recv(fSocket, static_cast<char *>(buffer) + bytesread,
                        length - bytesread, 0);
         } while(n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR));

         // If we read nothing, the connection has been closed by the other side
         if (n <= 0) {
	    if (errno > 0) {
	       Error("XrdClientSock::RecvRaw", "Error reading from socket: " <<
                                               ::strerror(errno));
	    }
            return TXSOCK_ERR;
         }
         bytesread += n;
      }

      // Then we check if poll reports a complaint from the socket like disconnections
      if (fds_r.revents & (POLLERR | POLLHUP | POLLNVAL)) {
         Error("ClientSock::RecvRaw",
               "Disconnection detected reading " << length <<
               " bytes from socket " << fds_r.fd <<
               " (server[" << fHost.TcpHost.Host <<
               ":" << fHost.TcpHost.Port <<
               "]). Revents=" << fds_r.revents );
               return TXSOCK_ERR;
      }

    } // while

    // Return number of bytes received
    return bytesread;
}

//_____________________________________________________________________________
int XrdClientSock::SendRaw_sock(const void* buffer, int length, int sock)
{
    // Write bytes following carefully the timeout rules
    // (writes will not hang)

    struct pollfd fds_w;
    int byteswritten = 0;
    int pollRet;

    // Init of the pollfd structs. If sock is not valid... we can do this anyway
    fds_w.fd     = sock;

    fds_w.events = POLLOUT | POLLERR | POLLHUP | POLLNVAL;

    // We cycle until we write all we have to write
    // Or until a timeout occurs

    while (byteswritten < length) {

      // We will not wait forever
      int timeleft = fRequestTimeout;
      do {
         // Wait for some event from the socket
         pollRet = poll(&fds_w,
                        1,
                        1000 // 1 second as a step
                        );
         if (((pollRet < 0) && (errno != EINTR)) || !fConnected)
            return TXSOCK_ERR;

      } while (--timeleft && pollRet <= 0 && !fWRInterrupt);

      // If we have been timed-out, return a specific error code
      if (timeleft <= 0) { //gEnv
         Error("ClientSock::SendRaw",
               "Request timed out "<< fRequestTimeout << //gEnv
               "seconds writing " << length << " bytes" <<
               " to server " << fHost.TcpHost.Host <<
               ":" << fHost.TcpHost.Port);

         return TXSOCK_ERR_TIMEOUT;
      }

      // If we have been interrupt, reset the interrupt and exit
      if (fWRInterrupt) {
         fWRInterrupt = 0;
         Error("XrdClientSock::SendRaw", "got interrupt");
         return TXSOCK_ERR_INTERRUPT;
      }

      // First of all, we check if we are allowed to write
      if (fds_w.revents & POLLOUT) {

         // We will be retrying on errors like EAGAIN or EWOULDBLOCK,
         // but not forever
         timeleft = fRequestTimeout;
         int n = -1;
         while (n <= 0 && timeleft--) {
            if ((n = send(sock, static_cast<const char *>(buffer) + byteswritten,
                          length - byteswritten, 0)) <= 0) {
               if (timeleft <= 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
                  // Real error: nothing more to do!
                  // If we wrote nothing, the connection has been closed by the other
                  Error("ClientSock::SendRaw", "Error writing to a socket: " <<
                  ::strerror(errno));
                  return (TXSOCK_ERR);
               } else {
                  // Sleep one second
                  sleep(1);
               }
            }
         }
         byteswritten += n;
      }

      // Then we check if poll reports a complaint from the socket like disconnections
      if (fds_w.revents & (POLLERR | POLLHUP | POLLNVAL)) {

         Error("ClientSock::SendRaw",
               "Disconnection detected writing " << length <<
               " bytes to socket " << fds_w.fd <<
               " (server[" << fHost.TcpHost.Host <<
               ":" << fHost.TcpHost.Port <<
               "]). Revents=" << fds_w.revents );
         return TXSOCK_ERR;
      }

    } // while

    // Return number of bytes sent
    return byteswritten;
}

//_____________________________________________________________________________
int XrdClientSock::SendRaw(const void* buffer, int length, int substreamid)
{
    // Note: here substreamid is used as "alternative socket" instead of fSocket

    if (substreamid > 0) 
       return SendRaw_sock(buffer, length, substreamid);
    else
       return SendRaw_sock(buffer, length, fSocket);
}

//_____________________________________________________________________________
void XrdClientSock::TryConnect(bool isUnix)
{
    // Already connected - we are done.
    //
    if (fConnected) {
       assert(fSocket >= 0);
       return;
    }

    
    fSocket = TryConnect_low(isUnix);

    if (fSocket >= 0) {
	
	// If we are using a SOCKS4 host...
	if ( EnvGetString(NAME_SOCKS4HOST) ) {

	    Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect", "Handshaking with SOCKS4 host");

	    switch (Socks4Handshake(fSocket)) {

	    case 90:
		// Everything OK!
		Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect", "SOCKS4 connection OK");
		break;
	    case 91:
	    case 92:
	    case 93:
		// Failed
		Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect",
		     "SOCKS host refused the connection.");
		Disconnect();
		break;
	    }



	}

    }
}

//_____________________________________________________________________________
int XrdClientSock::TryConnect_low(bool isUnix, int altport, int windowsz)
{
    int sock = -1;
    XrdOucString host;
    int port;
    if (!windowsz) windowsz = EnvGetLong(NAME_DFLTTCPWINDOWSIZE);


    host = EnvGetString(NAME_SOCKS4HOST);
    port = EnvGetLong(NAME_SOCKS4PORT);

    if (host.length() == 0) {
	host = fHost.TcpHost.HostAddr;
	port = fHost.TcpHost.Port;

	if (altport) port = altport;
    }
    else
	Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect_low", "Trying SOCKS4 host " <<
	     host << ":" << port);

    std::auto_ptr<XrdNetSocket> s(new XrdNetSocket());

    // Log the attempt
    //
    if (!isUnix) {
       Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect_low",
	    "Trying to connect to " <<
	    fHost.TcpHost.Host << "(" << host << "):" <<
	    port << " Windowsize=" << windowsz << " Timeout=" << EnvGetLong(NAME_CONNECTTIMEOUT));
    
       // Connect to a remote host
       //
       if (port)
          sock = s->Open(host.c_str(),
                         port, EnvGetLong(NAME_CONNECTTIMEOUT),
                         windowsz );
    } else {
	Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect_low",
	     "Trying to UNIX connect to" << fHost.TcpHost.File <<
	     "; timeout=" << EnvGetLong(NAME_CONNECTTIMEOUT));

	// Connect to a remote host
	//
	sock = s->Open(fHost.TcpHost.File.c_str(), -1, EnvGetLong(NAME_CONNECTTIMEOUT));
    }

    // Check if we really got a connection and the remote host is available
    //
    if (sock < 0)  {
	if (isUnix) {
	    Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect_low", "Connection to" <<
		 fHost.TcpHost.File << " failed. (" << sock << ")");
	} else {
	    Info(XrdClientDebug::kHIDEBUG, "ClientSock::TryConnect_low", "Connection to" <<
		 fHost.TcpHost.Host << ":" << fHost.TcpHost.Port << " failed. (" << sock << ")");
	}
      
    } else {
	fConnected = TRUE;
	int detachedFD = s->Detach();
	if (sock != detachedFD) {
	    Error("ClientSock::TryConnect_low",
		  "Socket detach returned " << detachedFD << " but expected " << sock);
	}
    }

    return sock;
}

int XrdClientSock::Socks4Handshake(int sockid) {

    char buf[4096], userid[4096];
#ifdef __FreeBSD__
    struct passwd *pwd;
#endif
    uint16_t port;
    char a, b, c, d;

    // Issue a Connect req
    buf[0] = 4; // Socks version
    buf[1] = 1; // Connect request

    port = htons(fHost.TcpHost.Port);
    memcpy(buf+2, &port, sizeof(port)); // The port

    sscanf(fHost.TcpHost.HostAddr.c_str(), "%hhd.%hhd.%hhd.%hhd", &a, &b, &c, &d ); 
    buf[4] = a;
    buf[5] = b;
    buf[6] = c;
    buf[7] = d;

#ifdef __FreeBSD__
    if ((pwd = getpwuid(geteuid())) == NULL)
        *userid = '\0';
    else
        (void)strncpy(userid, pwd->pw_name, L_cuserid);
#else
    cuserid(userid);
#endif
    strcpy(buf+8, userid);

    // send the buffer to the server!
    SendRaw(buf, 8+strlen(userid)+1, sockid);

    // Now wait the answer on the same sock
    RecvRaw(buf, 8, sockid);

    return buf[1];


}
