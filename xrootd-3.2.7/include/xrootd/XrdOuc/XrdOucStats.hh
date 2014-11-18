#ifndef _XRDOUCSTATS_HH_
#define _XRDOUCSTATS_HH_
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c S t a t s . h h                         */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdSys/XrdSysAtomics.hh"

#ifdef HAVE_ATOMICS
#define _statsINC(x) AtomicInc(x)
#else
#define _statsINC(x) statsMutex.Lock(); x++; statsMutex.UnLock()
#endif
  
class XrdOucStats
{
public:

inline void Bump(int &val)       {_statsINC(val);}

inline void Bump(long long &val) {_statsINC(val);}

XrdSysMutex statsMutex;   // Mutex to serialize updates

            XrdOucStats() {}
           ~XrdOucStats() {}
};
#endif
