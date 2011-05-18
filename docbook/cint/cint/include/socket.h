/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************
* socket.h 
***********************************************************/
#ifndef G__SOCKET_H
#define G__SOCKET_H


#pragma include_noerr "cintsock.dll"

#ifndef G__CINTSOCK_H /* G__CINTSOCK_H is defined in socket.dll */
#error cintsock.dll is not ready. Run setup script in $CINTSYSDIR/lib/socket directory
#endif


#define AF_INET 2
#define AF_UNIX 1
#define AF_ISO 7
#define AF_NS 6
#define AF_IMPLINK 3
#define INADDR_ANY 0
#define SOCK_STREAM 1
#define SOCK_DGRAM 2
#define SOCK_RAW 3
#define SOCK_SEQPACKET 5
#define SOCK_RDM 4
#define MSG_OOB 1
#define MSG_DONTROUTE 4
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1


#endif
