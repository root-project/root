// $Id$
#ifndef __SUT_CACHE_H__
#define __SUT_CACHE_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d S u t C a c h e . h h                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <XProtocol/XPtypes.hh>
#include <XrdSut/XrdSutPFEntry.hh>
#include <XrdOuc/XrdOucHash.hh>
#include <XrdOuc/XrdOucString.hh>
#include <XrdSys/XrdSysPthread.hh>

/******************************************************************************/
/*                                                                            */
/*  For caching temporary information during the authentication handshake     */
/*                                                                            */
/******************************************************************************/

class XrdSutCache
{
private:
   XrdSysRWLock    rwlock;  // Access synchronizator
   int             cachesz; // Number of entries allocated
   int             cachemx; // Largest Index of allocated entries
   XrdSutPFEntry **cachent; // Pointers to filled entries
   kXR_int32       utime;   // time at which was last updated
   int             lifetime; // lifetime (in secs) of the cache info 
   XrdOucHash<kXR_int32> hashtable; // Reflects the file index structure
   kXR_int32       htmtime;   // time at which hash table was last rebuild
   XrdOucString    pfile;   // file name (if loaded from file)

public:
   XrdSutCache() { cachemx = -1; cachesz = 0; cachent = 0; lifetime = 300;
                   utime = -1; htmtime = -1; pfile = "";}
   virtual ~XrdSutCache();

   // Status
   int            Entries() const { return (cachemx+1); }
   bool           Empty() const { return (cachemx == -1); }

   // Initialization methods
   int            Init(int capacity = 100);
   int            Reset(int newsz = -1);
   int            Load(const char *pfname);  // build cache of a pwd file
   int            Flush(const char *pfname = 0);   // flush content to pwd file
   int            Refresh();    // refresh content from source file
   int            Rehash(bool force = 0, bool lock = 1);  // (re)build hash table
   void           SetLifetime(int lifet = 300) { lifetime = lifet; }

   // Cache management
   XrdSutPFEntry *Get(int i) const { return (i<=cachemx) ? cachent[i] :
                                                          (XrdSutPFEntry *)0; }
   XrdSutPFEntry *Get(const char *ID, bool *wild = 0);
   XrdSutPFEntry *Add(const char *ID, bool force = 0);
   bool           Remove(const char *ID, int opt = 1);
   int            Trim(int lifet = 0);

   // For debug purposes
   void           Dump(const char *msg= 0);
};

#endif

