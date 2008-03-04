// $Id$
#ifndef __SUT_RNDM_H__
#define __SUT_RNDM_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d S u t R n d m . h h                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#ifndef __SUT_AUX_H__
#include <XrdSut/XrdSutAux.hh>
#endif

/******************************************************************************/
/*                                                                            */
/*  Provider of random bunches of bits                                        */
/*                                                                            */
/******************************************************************************/

class XrdOucString;

class XrdSutRndm {

public:
   static bool   fgInit;

   XrdSutRndm() { if (!fgInit) fgInit = XrdSutRndm::Init(); }
   virtual ~XrdSutRndm() { }

   // Initializer
   static bool   Init(bool force = 0);

   // Buffer provider
   static char  *GetBuffer(int len, int opt = -1);
   // String provider
   static int    GetString(int opt, int len, XrdOucString &s);
   static int    GetString(const char *copt, int len, XrdOucString &s);
   // Integer providers
   static unsigned int GetUInt();
   // Random Tag
   static int    GetRndmTag(XrdOucString &rtag);
}
;

#endif

