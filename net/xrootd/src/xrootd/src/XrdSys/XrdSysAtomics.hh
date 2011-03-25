#ifndef _XRDSYSATOMICS_
#define _XRDSYSATOMICS_
/******************************************************************************/
/*                                                                            */
/*                      X r d S y s A t o m i c s . h h                       */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* The following instruction acronyms are used:
   AtomicCAS() -> Compare And [if equal] Set
   AtomicDTZ() -> Decrease To Zero
   AtomicFAZ() -> Fetch And Zero
   AtomicISM() -> Increase and Set Maximum
*/
  
#ifdef HAVE_ATOMICS
#define AtomicBeg(Mtx)
#define AtomicEnd(Mtx)
#define AtomicAdd(x, y)     __sync_fetch_and_add(&x, y)
#define AtomicCAS(x, y, z)  __sync_bool_compare_and_swap(&x, y, z)
#define AtomicDec(x)        __sync_fetch_and_sub(&x, 1)
#define AtomicDTZ(x)        if (!(__sync_fetch_and_sub(&x, 1))) AtomicFAZ(x)
#define AtomicFAZ(x)        __sync_fetch_and_and(&x, 0)
#define AtomicGet(x)        __sync_fetch_and_or(&x, 0)
#define AtomicInc(x)        __sync_fetch_and_add(&x, 1)
#define AtomicISM(x, y)     AtomicCAS(y, AtomicInc(x), x)
#define AtomicSub(x, y)     __sync_fetch_and_sub(&x, y)
#else
#define AtomicBeg(Mtx)      Mtx.Lock()
#define AtomicEnd(Mtx)      Mtx.UnLock()
#define AtomicAdd(x, y)     x += y
#define AtomicCAS(x, y, z)  if (x == y) x = z
#define AtomicDTZ(x)        if (!(x--)) x = 0
#define AtomicDec(x)        x--
#define AtomicFAZ(x)        x; x = 0
#define AtomicGet(x)        x
#define AtomicInc(x)        x++
#define AtomicISM(x, y)     if (y == x++) y = x
#define AtomicSub(x, y)     x -= y
#endif
#endif
