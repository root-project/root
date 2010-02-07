/******************************************************************************/
/*                                                                            */
/*                        X r d F r m U t i l s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <errno.h>
#include <unistd.h>
#include <utime.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdSys/XrdSysError.hh"

const char *XrdFrmUtilsCVSID = "$Id$";

using namespace XrdFrm;
  
/******************************************************************************/
/*                                   A s k                                    */
/******************************************************************************/
  
char XrdFrmUtils::Ask(char dflt, const char *Msg1, const char *Msg2,
                                 const char *Msg3)
{
   const char *Hint;
   char Answer[8];
   int n;

   Hint = (dflt == 'y' ? " (y | n | a): " : " (n | y | a): ");

   do {cerr <<"frm_admin: " <<Msg1 <<Msg2 <<Msg3 <<Hint;
       cin.getline(Answer, sizeof(Answer));
       if (!*Answer) return dflt;

       n = strlen(Answer);
       if (!strncmp("yes",  Answer, n)) return 'y';
       if (!strncmp("no",   Answer, n)) return 'n';
       if (!strncmp("abort",Answer, n)) return 'a';
      } while(1);
   return 'a';
}
  
/******************************************************************************/
/*                                 U t i m e                                  */
/******************************************************************************/
  
int XrdFrmUtils::Utime(const char *Path, time_t tVal)
{
   struct utimbuf tbuf = {tVal, tVal};
   int rc;

// Set the time
//
   do {rc = utime(Path, &tbuf);} while(rc && errno == EINTR);
   if (rc) Say.Emsg("Utils", errno, "set utime for pfn", Path);

// All done
//
   return rc == 0;
}
