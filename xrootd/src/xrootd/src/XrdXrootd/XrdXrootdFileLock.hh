#ifndef _XROOTD_FILELOCK_H_
#define _XROOTD_FILELOCK_H_
/******************************************************************************/
/*                                                                            */
/*                  X r d X r o o t d F i l e L o c k . h h                   */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//      $Id$
 
#include "XrdXrootd/XrdXrootdFile.hh"
  
class XrdXrootdFileLock
{
public:

virtual int   Lock(XrdXrootdFile *fp, int force=0) = 0;

virtual void  numLocks(XrdXrootdFile *fp, int &rcnt, int &wcnt) = 0;

virtual int Unlock(XrdXrootdFile *fp) = 0;

            XrdXrootdFileLock() {}
virtual    ~XrdXrootdFileLock() {}
};
#endif
