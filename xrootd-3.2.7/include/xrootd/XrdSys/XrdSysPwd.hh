#ifndef __XRDSYSPWD_HH__
#define __XRDSYSPWD_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d S y s P w d . h h                           */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include <sys/types.h>
#include <pwd.h>

class XrdSysPwd
{
public:

int   rc;

struct passwd *Get(const char *Usr)
                  {rc = getpwnam_r(Usr,&pwStruct,pwBuff,sizeof(pwBuff),&Ppw);
                   return Ppw;
                  }

struct passwd *Get(uid_t       Uid)
                  {rc = getpwuid_r(Uid,&pwStruct,pwBuff,sizeof(pwBuff),&Ppw);
                   return Ppw;
                  }

               XrdSysPwd() : rc(2) {}

               XrdSysPwd(const char *Usr, struct passwd **pwP)
                  {rc = getpwnam_r(Usr,&pwStruct,pwBuff,sizeof(pwBuff),pwP);}

               XrdSysPwd(uid_t       Uid, struct passwd **pwP)
                  {rc = getpwuid_r(Uid,&pwStruct,pwBuff,sizeof(pwBuff),pwP);}

              ~XrdSysPwd() {}

private:

struct passwd pwStruct, *Ppw;
char          pwBuff[4096];
};
#endif
