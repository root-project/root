#ifndef __XRDNETOPTS_H__
#define __XRDNETOPTS_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d N e t O p t s . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

/******************************************************************************/
/*                    X r d N e t W o r k   O p t i o n s                     */
/******************************************************************************/

// Force a new file descriptor when setting up UDP communications
//
#define XRDNET_NEWFD     0x00000100

// This side of the socket will never receive any data
//
#define XRDNET_SENDONLY  0x00000200

// Multiple threads may attempts a read (very unusual)
//
#define XRDNET_MULTREAD 0x000000400

// Do not trim off fomain in the host name.
//
#define XRDNET_NODNTRIM 0x000000800

/******************************************************************************/
/*     X r d N e t W o r k   &   X r d N e t S o c k e t   O p t i o n s      */
/******************************************************************************/

// Turn off TCP_NODELAY
//
#define XRDNET_DELAY     0x00010000
  
// Enable SO_KEEPALIVE
//
#define XRDNET_KEEPALIVE 0x00020000

// Do not close the socket in child processes hwne they exec
//
#define XRDNET_NOCLOSEX  0x00040000

// Do not print common error messages (spotty right now)
//
#define XRDNET_NOEMSG    0x00080000

// Do not linger on a close
//
#define XRDNET_NOLINGER  0x00100000

// Define a UDP socket
//
#define XRDNET_UDPSOCKET 0x00200000

// Define a FIFO (currently only for NetSocket)
//
#define XRDNET_FIFO      0x00400000

// Avoid DNS reverse lookups
//
#define XRDNET_NORLKUP   0x00800000

/******************************************************************************/
/*                  X r d N e t S o c k e t   O p t i o n s                   */
/******************************************************************************/
  
// This socket will be used for server activities (only for XrdNetS
//
#define XRDNET_SERVER    0x10000000

// Maximum backlog for incomming connections. The backlog value goes in low
// order byte and is used only when XRDNET_SERVER is specified.
//
#define XRDNET_BKLG      0x000000FF

// Maximum wait time for outgoing connect. The timeout value goes in low
// order byte and is used only when XRDNET_SERVER is *NOT* specified.
// The value is in seconds (maximum timeout is 255 seconds).
//
#define XRDNET_TOUT      0x000000FF
#endif
