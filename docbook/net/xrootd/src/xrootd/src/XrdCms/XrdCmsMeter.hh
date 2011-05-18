#ifndef __CMS_METER__H
#define __CMS_METER__H
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s M e t e r . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"

class XrdCmsMeter
{
public:

int   calcLoad(int pcpu, int pio, int pload, int pmem, int ppag);

int   calcLoad(int xload,int pdsk);

int   FreeSpace(int &tutil);

void  Init();

int   isOn() {return Running;}

int   Monitor(char *pgm, int itv);

void  Record(int pcpu, int pnet, int pxeq,
             int pmem, int ppag, int pdsk);

int   Report(int &pcpu, int &pnet, int &pxeq,
             int &pmem, int &ppag, int &pdsk);

void *Run();

void *RunFS();

int   numFS() {return fs_nums;}

unsigned int TotalSpace(unsigned int &minfree);

enum  vType {manFS = 1, peerFS = 2};

void  setVirtual(vType vVal) {Virtual = vVal;}

void  setVirtUpdt() {cfsMutex.Lock(); VirtUpdt = 1; cfsMutex.UnLock();}

       XrdCmsMeter();
      ~XrdCmsMeter();

private:
      void calcSpace();
      char Scale(long long inval, long &outval);
      void SpaceMsg(int why);
      void UpdtSpace();

XrdOucStream  myMeter;
XrdSysMutex   cfsMutex;
XrdSysMutex   repMutex;
long long     MinFree;  // Calculated only once
long long     HWMFree;  // Calculated only once
long long     dsk_lpn;  // Calculated only once
long long     dsk_tot;  // Calculated only once
long long     dsk_free;
long long     dsk_maxf;
int           dsk_util;
int           dsk_calc;
int           fs_nums;  // Calculated only once
int           lastFree;
int           lastUtil;
int           noSpace;
int           Running;
long          MinShow;  // Calculated only once
long          HWMShow;  // Calculated only once
char          MinStype; // Calculated only once
char          HWMStype; // Calculated only once
char          Virtual;  // This is a virtual filesystem
char          VirtUpdt; // Data changed for the virtul FS

time_t        rep_tod;
char         *monpgm;
int           monint;
pthread_t     montid;

unsigned int  xeq_load;
unsigned int  cpu_load;
unsigned int  mem_load;
unsigned int  pag_load;
unsigned int  net_load;
};

namespace XrdCms
{
extern    XrdCmsMeter Meter;
}
#endif
