// $Id$
#ifndef  __SUT_PFENTRY_H
#define  __SUT_PFENTRY_H

/******************************************************************************/
/*                                                                            */
/*                      X r d S u t P F E n t r y . h h                       */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <XProtocol/XProtocol.hh>

/******************************************************************************/
/*                                                                            */
/*  Class defining the basic entry into a PFile                               */
/*                                                                            */
/******************************************************************************/

enum kPFEntryStatus {
   kPFE_inactive = -2,     // -2  inactive: eliminated at next trim
   kPFE_disabled,          // -1  disabled, cannot be enabled
   kPFE_allowed,           //  0  empty creds, can be enabled 
   kPFE_ok,                //  1  enabled and OK
   kPFE_onetime,           //  2  enabled, can be used only once
   kPFE_expired,           //  3  enabled, creds to be changed at next used
   kPFE_special,           //  4  special (non-creds) entry
   kPFE_anonymous,         //  5  enabled, OK, no creds, counter
   kPFE_crypt              //  6  enabled, OK, crypt-like credentials
};

//
// Buffer used internally by XrdSutPFEntry
//
class XrdSutPFBuf {
public:
   char      *buf;
   kXR_int32  len;   
   XrdSutPFBuf(char *b = 0, kXR_int32 l = 0);
   XrdSutPFBuf(const XrdSutPFBuf &b);

   virtual ~XrdSutPFBuf() { if (len > 0 && buf) delete[] buf; }

   void SetBuf(const char *b = 0, kXR_int32 l = 0);
};

//
// Generic File entry: it stores a 
//
//        name
//        status                     2 bytes
//        cnt                        2 bytes
//        mtime                      4 bytes
//        buf1, buf2, buf3, buf4
//
// The buffers are generic buffers to store bufferized info
//
class XrdSutPFEntry {
public:
   char        *name;
   short        status;
   short        cnt;            // counter
   kXR_int32    mtime;          // time of last modification / creation
   XrdSutPFBuf  buf1;
   XrdSutPFBuf  buf2;
   XrdSutPFBuf  buf3;
   XrdSutPFBuf  buf4;
   XrdSutPFEntry(const char *n = 0, short st = 0, short cn = 0,
                 kXR_int32 mt = 0);
   XrdSutPFEntry(const XrdSutPFEntry &e);
   virtual ~XrdSutPFEntry() { if (name) delete[] name; } 
   kXR_int32 Length() const { return (buf1.len + buf2.len + 2*sizeof(short) +
                                      buf3.len + buf4.len + 5*sizeof(kXR_int32)); }
   void Reset();
   void SetName(const char *n = 0);
   char *AsString() const;

   XrdSutPFEntry &operator=(const XrdSutPFEntry &pfe);
};

#endif
