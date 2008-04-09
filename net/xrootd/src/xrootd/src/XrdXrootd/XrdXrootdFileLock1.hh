#ifndef _XROOTD_FILELOCK1_H_
#define _XROOTD_FILELOCK1_H_
/******************************************************************************/
/*                                                                            */
/*                 X r d X r o o t d F i l e L o c k 1 . h h                  */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//      $Id$
 
#include "XrdSys/XrdSysPthread.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdFileLock.hh"

// This class implements a single server per host lock manager by simply using
// an in-memory hash table to keep track of file locks.
//
class XrdXrootdFileLock1 : XrdXrootdFileLock
{
public:

        int   Lock(XrdXrootdFile *fp, int force=0);

        void  numLocks(XrdXrootdFile *fp, int &rcnt, int &wcnt);

        int Unlock(XrdXrootdFile *fp);

            XrdXrootdFileLock1() {}
           ~XrdXrootdFileLock1() {} // This object is never destroyed!
private:
static const char *TraceID;
static XrdSysMutex  LTMutex;
};
#endif
