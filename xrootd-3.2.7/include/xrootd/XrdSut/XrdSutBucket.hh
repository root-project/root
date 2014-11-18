// $Id$
#ifndef __SUT_BUCKET_H__
#define __SUT_BUCKET_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d S u t B u c k e t . h h                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#ifndef __SUT_STRING_H__
#include <XrdSut/XrdSutAux.hh>
#endif

class XrdOucString;

/******************************************************************************/
/*                                                                            */
/*  Unit for information exchange                                             */
/*                                                                            */
/******************************************************************************/

class XrdSutBucket
{
public:
   kXR_int32   type;
   kXR_int32   size;
   char       *buffer;

   XrdSutBucket(char *bp=0, int sz=0, int ty=0);
   XrdSutBucket(XrdOucString &s, int ty=0);
   XrdSutBucket(XrdSutBucket &b);
   virtual ~XrdSutBucket() {if (membuf) delete[] membuf;}

   void Update(char *nb = 0, int ns = 0, int ty = 0); // Uses 'nb'
   int Update(XrdOucString &s, int ty = 0);
   int SetBuf(const char *nb = 0, int ns = 0);         // Duplicates 'nb'

   void Dump(int opt = 1);
   void ToString(XrdOucString &s);

   // Equality operator
   int operator==(const XrdSutBucket &b);

   // Inequality operator
   int operator!=(const XrdSutBucket &b) { return !(*this == b); }

private:
   char *membuf;
};

#endif

