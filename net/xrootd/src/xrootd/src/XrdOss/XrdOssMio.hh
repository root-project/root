#ifndef __XRDOSSMIO_H__
#define __XRDOSSMIO_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d o s s M i o . h h                           */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOss/XrdOssMioFile.hh"

// The following are options passed to Map()
//
#define OSSMIO_MLOK 0x0001
#define OSSMIO_MMAP 0x0002
#define OSSMIO_MPRM 0x0004
  
class XrdOssMio
{
public:
static void           Display(XrdSysError &Eroute);

static char           isAuto() {return MM_chk;}

static char           isOn()   {return MM_on;}

static XrdOssMioFile *Map(char *path, int fd, int opts);

static void          *preLoad(void *arg);

static void           Recycle(XrdOssMioFile *mp);

static void           Set(int V_off, int V_preld, int V_check);

static void           Set(long long V_max);

private:
static int  Reclaim(off_t amount);
static int  Reclaim(XrdOssMioFile *mp);

static XrdOucHash<XrdOssMioFile> MM_Hash;

static XrdSysMutex    MM_Mutex;
static XrdOssMioFile *MM_Perm;
static XrdOssMioFile *MM_Idle;
static XrdOssMioFile *MM_IdleLast;

static char       MM_on;
static char       MM_chk;
static char       MM_okmlock;
static char       MM_preld;
static long long  MM_max;
static long long  MM_pagsz;
static long long  MM_pages;
static long long  MM_inuse;
};
#endif
