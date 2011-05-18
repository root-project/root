#ifndef __ACC_AUTHFILE__
#define __ACC_AUTHFILE__
/******************************************************************************/
/*                                                                            */
/*                     X r d A c c A u t h F i l e . h h                      */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <limits.h>
#include <netdb.h>
#include <sys/param.h>
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdAcc/XrdAccAuthDB.hh"

// This class is provided for obtaining capability information from a file.
//
class XrdAccAuthFile : public XrdAccAuthDB
{
public:

int      Open(XrdSysError &eroute, const char *path=0);

char     getRec(char **recname);

int      getPP(char **path, char **priv);

int      Close();

int      Changed(const char *dbpath);

         XrdAccAuthFile(XrdSysError *erp);
        ~XrdAccAuthFile();

private:

int  Bail(int retc, const char *txt1, const char *txt2=0);
char *Copy(char *dp, char *sp, int dplen);

enum DBflags {Noflags=0, inRec=1, isOpen=2, dbError=4}; // Values combined

XrdSysError      *Eroute;
DBflags           flags;
XrdOucStream      DBfile;
char             *authfn;
char              rectype;
time_t            modtime;
XrdSysMutex       DBcontext;

char recname_buff[MAXHOSTNAMELEN+1];   // Max record name by default
char path_buff[PATH_MAX+2];          // Max path   name
};
#endif
