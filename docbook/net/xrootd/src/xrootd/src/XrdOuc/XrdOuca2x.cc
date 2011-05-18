/******************************************************************************/
/*                                                                            */
/*                          X r d O u c a 2 x . c c                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdOuca2xCVSID = "$Id$";

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef WIN32
#include "XrdSys/XrdWin32.hh"
#endif
#include "XrdOuc/XrdOuca2x.hh"

/******************************************************************************/
/*                                   a 2 i                                    */
/******************************************************************************/

int XrdOuca2x::a2i(XrdSysError &Eroute, const char *emsg, const char *item,
                                             int *val, int minv, int maxv)
{
    char *eP;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

    errno = 0;
    *val  = strtol(item, &eP, 10);
    if (errno || *eP)
       {Eroute.Emsg("a2x", emsg, item, "is not a number");
        return -1;
       }
    if (*val < minv) 
       return Emsg(Eroute, emsg, item, "may not be less than %d", minv);
    if (maxv >= 0 && *val > maxv)
       return Emsg(Eroute, emsg, item, "may not be greater than %d", maxv);
    return 0;
}
 
/******************************************************************************/
/*                                  a 2 l l                                   */
/******************************************************************************/

int XrdOuca2x::a2ll(XrdSysError &Eroute, const char *emsg, const char *item,
                                long long *val, long long minv, long long maxv)
{
    char *eP;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

    errno = 0;
    *val  = strtoll(item, &eP, 10);
    if (errno || *eP)
       {Eroute.Emsg("a2x", emsg, item, "is not a number");
        return -1;
       }
    if (*val < minv) 
       return Emsg(Eroute, emsg, item, "may not be less than %lld", minv);
    if (maxv >= 0 && *val > maxv)
       return Emsg(Eroute, emsg, item, "may not be greater than %lld", maxv);
    return 0;
}

/******************************************************************************/
/*                                  a 2 f m                                   */
/******************************************************************************/

int XrdOuca2x::a2fm(XrdSysError &Eroute, const char *emsg, const char *item,
                                              int *val, int minv, int maxv)
{  int rc, num;
   if ((rc = a2fm(Eroute, emsg, item, &num, minv))) return rc;
   if ((*val | maxv) != maxv) 
      {Eroute.Emsg("a2fm", emsg, item, "is too inclusive.");
       return -1;
      }

   *val = 0;
   if (num & 0100) *val |= S_IXUSR; // execute permission: owner
   if (num & 0200) *val |= S_IWUSR; // write permission:   owner
   if (num & 0400) *val |= S_IRUSR; // read permission:    owner
   if (num & 0010) *val |= S_IXGRP; // execute permission: group
   if (num & 0020) *val |= S_IWGRP; // write permission:   group
   if (num & 0040) *val |= S_IRGRP; // read permission:    group
   if (num & 0001) *val |= S_IXOTH; // execute permission: other
   if (num & 0002) *val |= S_IWOTH; // write permission:   other
   if (num & 0004) *val |= S_IROTH; // read permission:    other
   return 0;
}

int XrdOuca2x::a2fm(XrdSysError &Eroute, const char *emsg, const char *item,
                                              int *val, int minv)
{
    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

    errno = 0;
    *val  = strtol(item, (char **)NULL, 8);
    if (errno)
       {Eroute.Emsg("a2x", emsg, item, "is not an octal number");
        return -1;
       }
    if (!(*val & minv))
       {Eroute.Emsg("a2x", emsg, item, "is too exclusive");;
        return -1;
       }
    return 0;
}
 
/******************************************************************************/
/*                                  a 2 s p                                   */
/******************************************************************************/

int XrdOuca2x::a2sp(XrdSysError &Eroute, const char *emsg, const char *item,
                                long long *val, long long minv, long long maxv)
{
    char *pp, buff[120];
    int i;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

    i = strlen(item);
    if (item[i-1] != '%') return a2sz(Eroute, emsg, item, val, minv, maxv);

    errno = 0;
    *val  = strtoll(item, &pp, 10);

    if (errno || *pp != '%')
       {Eroute.Emsg("a2x", emsg, item, "is not a number");
        return -1;
       }

    if (maxv < 0) maxv = 100;

    if (*val > maxv)
       {sprintf(buff, "may not be greater than %lld%%", maxv);
        Eroute.Emsg("a2x", emsg, item, buff);
        return -1;
       }

    if (minv < 0) minv = 0;

    if (*val > maxv)
       {sprintf(buff, "may not be less than %lld%%", minv);
        Eroute.Emsg("a2x", emsg, item, buff);
        return -1;
       }

    *val = -*val;
    return 0;
}

/******************************************************************************/
/*                                  a 2 s z                                   */
/******************************************************************************/

int XrdOuca2x::a2sz(XrdSysError &Eroute, const char *emsg, const char *item,
                                long long *val, long long minv, long long maxv)
{   long long qmult;
    char *eP, *fP = (char *)item + strlen(item) - 1;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

         if (*fP == 'k' || *fP == 'K') qmult = 1024LL;
    else if (*fP == 'm' || *fP == 'M') qmult = 1024LL*1024LL;
    else if (*fP == 'g' || *fP == 'G') qmult = 1024LL*1024LL*1024LL;
    else if (*fP == 't' || *fP == 'T') qmult = 1024LL*1024LL*1024LL*1024LL;
    else                              {qmult = 1; fP++;}
    errno = 0;
    *val  = strtoll(item, &eP, 10) * qmult;
    if (errno || eP != fP)
       {Eroute.Emsg("a2x", emsg, item, "is not a number");
        return -1;
       }
    if (*val < minv) 
       return Emsg(Eroute, emsg, item, "may not be less than %lld", minv);
    if (maxv >= 0 && *val > maxv)
       return Emsg(Eroute, emsg, item, "may not be greater than %lld", maxv);
    return 0;
}
 
/******************************************************************************/
/*                                  a 2 t m                                   */
/******************************************************************************/

int XrdOuca2x::a2tm(XrdSysError &Eroute, const char *emsg, const char *item, int *val,
                          int minv, int maxv)
{   int qmult;
    char *eP, *fP = (char *)item + strlen(item) - 1;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

         if (*fP == 's' || *fP == 'S') qmult = 1;
    else if (*fP == 'm' || *fP == 'M') qmult = 60;
    else if (*fP == 'h' || *fP == 'H') qmult = 60*60;
    else if (*fP == 'd' || *fP == 'D') qmult = 60*60*24;
    else                              {qmult = 1; fP++;}

    errno = 0;
    *val  = strtoll(item, &eP, 10) * qmult;
    if (errno || eP != fP)
       {Eroute.Emsg("a2x", emsg, item, "is not a number");
        return -1;
       }
    if (*val < minv) 
       return Emsg(Eroute, emsg, item, "may not be less than %d", minv);
    if (maxv >= 0 && *val > maxv)
       return Emsg(Eroute, emsg, item, "may not be greater than %d", maxv);
    return 0;
}

/******************************************************************************/
/*                                  a 2 v p                                   */
/******************************************************************************/

int XrdOuca2x::a2vp(XrdSysError &Eroute, const char *emsg, const char *item,
                                             int *val, int minv, int maxv)
{
    char *pp;

    if (!item || !*item)
       {Eroute.Emsg("a2x", emsg, "value not specified"); return -1;}

    errno = 0;
    *val  = strtol(item, &pp, 10);

    if (!errno && *pp == '%')
       {if (*val < 0)
           {Eroute.Emsg("a2x", emsg, item, "may not be negative.");
            return -1;
           }
        if (*val > 100)
           {Eroute.Emsg("a2x", emsg, item, "may not be greater than 100%.");
            return -1;
           }
           else {*val = -*val; return 0;}
       }

    if (*val < minv) 
       return Emsg(Eroute, emsg, item, "may not be less than %d", minv);
    if (maxv >= 0 && *val > maxv)
       return Emsg(Eroute, emsg, item, "may not be greater than %d", maxv);
    return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
  
int XrdOuca2x::Emsg(XrdSysError &Eroute, const char *etxt1, const char *item,
                                         const char *etxt2, int val)
{char buff[256];
 sprintf(buff, etxt2, val);
 Eroute.Emsg("a2x", etxt1, item, buff);
 return -1;
}

int XrdOuca2x::Emsg(XrdSysError &Eroute, const char *etxt1, const char *item,
                                         const char *etxt2, long long val)
{char buff[256];
 sprintf(buff, etxt2, val);
 Eroute.Emsg("a2x", etxt1, item, buff);
 return -1;
}
