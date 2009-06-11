#ifndef __SecsssRR__
#define __SecsssRR__
/******************************************************************************/
/*                                                                            */
/*                        X r d S e c s s s R R . h h                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

#include <string.h>
#include <time.h>

// The following is the packet header and is always unencrypted.
//
struct XrdSecsssRR_Hdr
{
char      ProtID[4];                 // Protocol ID ("sss")
char      Pad[3];                    // Padding bytes
char      EncType;                   // Encryption type as one of:
static const char etBFish32 = '0';   // Blowfish

long long KeyID;                     // Key ID for encryption
};

// The data portion of the packet is encrypted with the private shared key
// It immediately follows the header and has a maximum size (defined here).
//
struct XrdSecsssRR_Data
{
char      Rand[32];                  // 256-bit random string (avoid text attacks)
int       GenTime;                   // Time data generated   (time(0) - BaseTime)
char      Pad[3];                    // Reserved
char      Options;                   // One of the following:
static const char UseData= 0x00;     // Use the ID data  as authenticated name
static const char SndLID = 0x01;     // Server to send login ID

static const int  DataSz = 4040;
char      Data[DataSz];              // Optional data, as follows:

//           (<Flag><packed null terminated string>)+
//
static const char theName = 0x01;
static const char theVorg = 0x02;
static const char theRole = 0x03;
static const char theGrps = 0x04;
static const char theEndo = 0x05;
//                theCert = 0x06; // Reserved for future use
static const char theRand = 0x07; // Random string (ignored)
static const char theLgid = 0x10; // from server only
static const char theHost = 0x20; // from client only (required)
};

static const int  XrdSecsssRR_Data_HdrLen = sizeof(XrdSecsssRR_Data) -
                                            XrdSecsssRR_Data::DataSz;
#endif
