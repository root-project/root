#ifndef __OLB_METER__H
#define __OLB_METER__H
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b M e t e r . h h                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"

class XrdOlbMeterFS;
  
class XrdOlbMeter
{
public:

int   calcLoad(int pcpu, int pio, int pload, int pmem, int ppag);

int   FreeSpace(int &tutil);

int   isOn() {return Running;}

int   Monitor(char *pgm, int itv);

char *Report();

void *Run();

void *RunFS();

int   numFS() {return fs_nums;}

void  setParms(XrdOucTList *tlp, int warnDups);

       XrdOlbMeter();
      ~XrdOlbMeter();

private:
      void calcSpace();
      void informLoad(void);
      int  isDup(struct stat &buf, XrdOlbMeterFS *baseFS);
const char Scale(long long inval, long &outval);
      void SpaceMsg(int why);

XrdOucStream  myMeter;
XrdSysMutex   cfsMutex;
XrdSysMutex   repMutex;
XrdOucTList  *fs_list;
long long     MinFree;
long long     HWMFree;
long long     dsk_tot;  // Calculated only once
long long     dsk_free;
long long     dsk_maxf;
int           dsk_util;
int           dsk_calc;
int           fs_nums;  // Calculated only once
int           noSpace;
int           Running;

char          ubuff[64];
time_t        rep_tod;
time_t        rep_todfs;
char         *monpgm;
int           monint;
pthread_t     montid;

unsigned int  xeq_load;
unsigned int  cpu_load;
unsigned int  mem_load;
unsigned int  pag_load;
unsigned int  net_load;
};

namespace XrdOlb
{
extern    XrdOlbMeter Meter;
}
#endif
