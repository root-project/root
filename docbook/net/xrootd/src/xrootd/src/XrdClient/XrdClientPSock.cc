//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientPSock                                                       //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Client Socket with parallel streams and timeout features using XrdNet//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientPSockCVSID = "$Id$";

#include <memory>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "XrdClient/XrdClientPSock.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"

#ifdef __solaris__
#include <sunmath.h>
#endif

#ifndef WIN32
#include <unistd.h>
#include <sys/poll.h>
#else
#include "XrdSys/XrdWin32.hh"
#endif

//_____________________________________________________________________________
XrdClientPSock::XrdClientPSock(XrdClientUrlInfo Host, int windowsize):
    XrdClientSock(Host, windowsize) {

    lastsidhint = 0;
    fReinit_fd = true;
    
}

//_____________________________________________________________________________
XrdClientPSock::~XrdClientPSock()
{
   // Destructor
   Disconnect();
}

//_____________________________________________________________________________
int CloseSockFunc(int K, int V, void *arg) {
    ::close(V);
    
    // And also we delete this item by returning < 0
    return -1;
}
//_____________________________________________________________________________
void XrdClientPSock::Disconnect()
{
  // Close the connection
  XrdSysMutexHelper mtx(fMutex);

  fConnected = FALSE;
    
  // Make the SocketPool invoke the closing of all sockets
  fSocketPool.Apply( CloseSockFunc, 0 );
    
  fSocketIdPool.Purge();
  fSocketIdRepo.Clear();

}



//_____________________________________________________________________________

struct FdSetSockFuncPars {
   struct fdinfo *fdnfo;
   XrdOucRash<XrdClientSock::Sockdescr, XrdClientSock::Sockid> *banned;
};

int FdSetSockFunc(int sockid, int sockdescr, void *arg) {
   struct FdSetSockFuncPars *pars = (struct FdSetSockFuncPars *)arg;
   struct fdinfo *fds = pars->fdnfo;
   

   // There could some sockets in the "banned" state
   // I.e. still in the process of being handshaked, but present in the tables
   // Those sockets must not be taken into acct by the global selecting mechanism
   if ( (sockdescr >= 0) && (!pars->banned->Find(sockdescr)) ) {
      FD_SET(sockdescr, &fds->fdset);
      fds->maxfd = xrdmax(fds->maxfd, sockdescr);
   }


   // And we continue
   return 0;
}

//_____________________________________________________________________________
int XrdClientPSock::RecvRaw(void* buffer, int length, int substreamid,
			   int *usedsubstreamid)
{
  // Read bytes following carefully the timeout rules
  time_t starttime;
  int bytesread = 0;
  int selRet;
  // The local set of interesting sock descriptors
  struct fdinfo locfdinfo;

   // We cycle reading data.
   // An exit occurs if:
   // We have all the data we are waiting for
   // Or a timeout occurs
   // The connection is closed by the other peer

   if (!fConnected) {
       Error("XrdClientPSock::RecvRaw", "Not connected.");
       return TXSOCK_ERR;
   }
   if (GetMainSock() < 0) {
       Error("XrdClientPSock::RecvRaw", "cannot find main socket.");
       return TXSOCK_ERR;
   }


   starttime = time(0);

   while (bytesread < length) {

     // We cycle on the select, ignoring the possible interruptions
     // We are waiting for something to come from the socket(s)
     do {
        
       if (fReinit_fd) {
         // we want to reconstruct the global fd_set
         Info(XrdClientDebug::kDUMPDEBUG, "XrdClientPSock::RecvRaw", "Reconstructing global fd table.");

	 XrdSysMutexHelper mtx(fMutex);

         FD_ZERO(&globalfdinfo.fdset);
	 globalfdinfo.maxfd = 0;

         // We are interested in any sock, except for the banned ones
         struct FdSetSockFuncPars fdpars;
         fdpars.fdnfo = &globalfdinfo;
         fdpars.banned = &fSocketNYHandshakedIdPool;

         fSocketPool.Apply( FdSetSockFunc, (void *)&fdpars );
         fReinit_fd = false;
       }

       // If we already read something, then we are stuck to a single socket
       // waiting for the completion of its read
       // This is reflected in the local fdset hence we don't have to touch it
       //       if ((!bytesread) || (substreamid == -1)) {

       if (substreamid == -1) {
	   // We are interested in any sock and we are not stuck
	   // to any in particular so we take the global fdset
	   locfdinfo = globalfdinfo;
	   
	 } else {
	   // we are using a single specified sock
	   XrdSysMutexHelper mtx(fMutex);

	   FD_ZERO(&locfdinfo.fdset);
	   locfdinfo.maxfd = 0;
	   
	   int sock = GetSock(substreamid);
	   if (sock >= 0) {
	     FD_SET(sock, &locfdinfo.fdset);
	     locfdinfo.maxfd = sock;
	     
	   }
	   else {
	     Error("XrdClientPSock::RecvRaw", "since we entered RecvRaw, the substreamid " <<
		   substreamid << " has been removed.");
	   
	     // A dropped parallel stream is not considered
	     // as an error
	     if (substreamid == 0)
	       return TXSOCK_ERR;
	     else {
	       XrdSysMutexHelper mtx(fMutex);
               if (sock >= 0)
                  FD_CLR(sock, &globalfdinfo.fdset);

	       RemoveParallelSock(substreamid);
	       //ReinitFDTable();
	       return TXSOCK_ERR_TIMEOUT;
	     }
	   }
	 }

	 //       }

       
         // If too much time has elapsed, then we return an error
         if ((time(0) - starttime) > EnvGetLong(NAME_REQUESTTIMEOUT)) {

	    return TXSOCK_ERR_TIMEOUT;
         }

	 struct timeval tv = { 0, 100000 }; // .1 second as timeout step

	 // Wait for some events from the socket pool
	 errno = 0;
         selRet = select(locfdinfo.maxfd+1, &locfdinfo.fdset, NULL, NULL, &tv);

	 if ( (selRet < 0) && (errno != EINTR) && (errno != EAGAIN) ) {
	     Error("XrdClientPSock::RecvRaw", "Error in select() : " <<
		   ::strerror(errno));

	     ReinitFDTable();
	     return TXSOCK_ERR;
	 }

      } while (selRet <= 0 && !fRDInterrupt);

      // If we are here, selRet is > 0 why?
      //  Because the timeout and the select error are handled inside the previous loop
      // But we could have been requested to interrupt

      if (GetMainSock() < 0) {
         Error("XrdClientPSock::RecvRaw", "since we entered RecvRaw, the main socket "
	       "file descriptor has been removed.");
         return TXSOCK_ERR;
      }

      // If we have been interrupt, reset the interrupt and exit
      if (fRDInterrupt) {
         fRDInterrupt = 0;
         Error("XrdClientPSock::RecvRaw", "got interrupt");
         return TXSOCK_ERR_INTERRUPT;
      }

      // First of all, we check if there is something to read from any sock.
      // the first we find is ok for now
      for (int ii = 0; ii <= locfdinfo.maxfd; ii++) {

	  if (FD_ISSET(ii, &locfdinfo.fdset)) {
	      int n = 0;

	      do {
                 errno = 0;
                 n = ::recv(ii, static_cast<char *>(buffer) + bytesread,
                            length - bytesread, 0);
              } while (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR));

	      // If we read nothing, the connection has been closed by the other side
	      if ((n <= 0)  && (errno != EINTR)) {
		Error("XrdClientPSock::RecvRaw", "Error reading from socket " << ii << ". n=" << n <<
		      " Error:'" <<
		      ::strerror(errno) << "'");

		  // A dropped parallel stream is not considered
		  // as an error
		  if (( GetSockId(ii) == 0 ) || ( GetSockId(ii) == -1 ))
		      return TXSOCK_ERR;
		  else {
		    XrdSysMutexHelper mtx(fMutex);
		    FD_CLR(ii, &globalfdinfo.fdset);
		    RemoveParallelSock(GetSockId(ii));
		    //ReinitFDTable();
		    return TXSOCK_ERR_TIMEOUT;
		  }

	      }
	      
	      if (n > 0) bytesread += n;
	      
	      // If we need to loop more than once to get the whole amount
	      // of requested bytes, then we have to select only on this fd which
	      // started providing a chunk of data
	      FD_ZERO(&locfdinfo.fdset);
	      FD_SET(ii, &locfdinfo.fdset);
	      locfdinfo.maxfd = ii;
	      substreamid = GetSockId(ii);

	      if (usedsubstreamid) *usedsubstreamid = GetSockId(ii);

	      // We got some data, hence we stop scanning the fd list,
	      // but we remain stuck to the socket which started providing data
	      break;
	  }
      }

   } // while

   // Return number of bytes received
   // And also usedparsockid has been initialized with the sockid we got something from

   return bytesread;
}

int XrdClientPSock::SendRaw(const void* buffer, int length, int substreamid) {

    int sfd = GetSock(substreamid);

    Info(XrdClientDebug::kDUMPDEBUG,
	 "SendRaw",
	 "Writing to substreamid " <<
	 substreamid << " mapped to socket fd " << sfd);

    return XrdClientSock::SendRaw(buffer, length, sfd);

}

//_____________________________________________________________________________
void XrdClientPSock::TryConnect(bool isUnix) {
    // Already connected - we are done.
    //
    if (fConnected) {
   	assert(GetMainSock() >= 0);
	return;
    }

    int s = TryConnect_low(isUnix);

    if (s >= 0) {
        XrdSysMutexHelper mtx(fMutex);

	int z = 0;
	fSocketPool.Rep(0, s);
	fSocketIdPool.Rep(s, z);
	//	fSocketIdRepo.Push_back(z);
    }

}

XrdClientSock::Sockdescr XrdClientPSock::TryConnectParallelSock(int port, int windowsz, Sockid &newid) {

    int s = TryConnect_low(false, port, windowsz);

    if (s >= 0) {

        XrdSysMutexHelper mtx(fMutex);

        // Now we have a good connection, valid from the TCP point of view

        // But we prevent the socket from appearing in the global fd table for now
        BanSockDescr(s, newid);

        // We put the descriptor and the id in the tables
	fSocketPool.Rep(newid, s);
	fSocketIdPool.Rep(s, newid);

    }

    return s;
}

int XrdClientPSock::RemoveParallelSock(int sockid) {

    XrdSysMutexHelper mtx(fMutex);

    int s = GetSock(sockid);

    if (s >= 0) ::close(s);

    fSocketIdPool.Del(s);
    fSocketPool.Del(sockid);

    for (int i = 0; i < fSocketIdRepo.GetSize(); i++)
	if (fSocketIdRepo[i] == sockid) {
	    fSocketIdRepo.Erase(i);
	    break;
	}

    return 0;
}

int XrdClientPSock::EstablishParallelSock(Sockid tmpsockid, Sockid newsockid) {
    XrdSysMutexHelper mtx(fMutex);

    Sockdescr s = GetSock(tmpsockid);
    if (s >= 0) {
  
        fSocketPool.Del(tmpsockid);
        fSocketIdPool.Del(s);

        fSocketPool.Rep(newsockid, s);
        fSocketIdPool.Rep(s, newsockid);
        fSocketIdRepo.Push_back(newsockid);

        Info(XrdClientDebug::kUSERDEBUG,
             "XrdClientSock::EstablishParallelSock", "Sockid " << newsockid << " established.");

        return 0;
    }

    return -1;

}

int XrdClientPSock::GetSockIdHint(int reqsperstream) {

  XrdSysMutexHelper mtx(fMutex);

  // A round robin through the secondary streams. We avoid
  // requesting data through the main one because it can become a bottleneck
  if (fSocketIdRepo.GetSize() > 0) {
     int tmp = lastsidhint+1;
     lastsidhint = ( ( tmp % (fSocketIdRepo.GetSize()*reqsperstream) )  );
  }
  else lastsidhint = 0;

  return fSocketIdRepo[lastsidhint / reqsperstream];
  //return (random() % (fSocketIdRepo.GetSize()+1));

}



void XrdClientPSock::PauseSelectOnSubstream(int substreamid) {
  XrdSysMutexHelper mtx(fMutex);

   int sock = GetSock(substreamid);

   if (sock >= 0)
      FD_CLR(sock, &globalfdinfo.fdset);

}


void XrdClientPSock::RestartSelectOnSubstream(int substreamid) {
  XrdSysMutexHelper mtx(fMutex);

   int sock = GetSock(substreamid);

   if (sock >= 0)
      FD_SET(sock, &globalfdinfo.fdset);

}
