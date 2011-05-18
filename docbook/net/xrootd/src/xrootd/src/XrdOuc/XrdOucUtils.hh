#ifndef __XRDOUCUTILS_HH__
#define __XRDOUCUTILS_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c U t i l s . h h                         */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

#include <sys/types.h>
#include <sys/stat.h>
  
class XrdSysError;
class XrdOucStream;

class XrdOucUtils
{
public:

static const mode_t pathMode = S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH;

static char *eText(int rc, char *eBuff, int eBlen, int AsIs=0);

static int   doIf(XrdSysError *eDest, XrdOucStream &Config,
                  const char *what, const char *hname, 
                                    const char *nname, const char *pname);
 
static int   fmtBytes(long long val, char *buff, int bsz);

static char *genPath(const char *path, const char *inst, const char *psfx=0);

static int   genPath(char *buff, int blen, const char *path, const char *psfx=0);

static const char *InstName(int TranOpt=0);

static const char *InstName(const char *name, int Fillit=1);

static int   is1of(char *val, const char **clist);

static void  makeHome(XrdSysError &eDest, const char *inst);

static int   makePath(char *path, mode_t mode);
 
static char *subLogfn(XrdSysError &eDest, const char *inst, char *logfn);

static void  Undercover(XrdSysError &eDest, int noLog);

       XrdOucUtils() {}
      ~XrdOucUtils() {}
};
#endif
