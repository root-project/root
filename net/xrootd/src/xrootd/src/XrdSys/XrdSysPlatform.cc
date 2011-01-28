/******************************************************************************/
/*                                                                            */
/*                     X r d S y s P l a t f o r m . c c                      */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//        $Id$

const char *XrdSysPlatformCVSID = "$Id$";

#include <stdio.h>
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#include <netinet/in.h>
#endif
#include <sys/types.h>

#if defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__) || \
    defined(__IEEE_LITTLE_ENDIAN) || \
   (defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN)
#if !defined(__GNUC__) || defined(__macos__) || defined(__solaris__)
extern "C"
{
unsigned long long Swap_n2hll(unsigned long long x)
{
    unsigned long long ret_val;
    *( (unsigned int  *)(&ret_val) + 1) = ntohl(*( (unsigned int  *)(&x)));
    *(((unsigned int  *)(&ret_val)))    = ntohl(*(((unsigned int  *)(&x))+1));
    return ret_val;
}
}
#endif

#endif

#ifndef HAVE_STRLCPY
extern "C"
{
size_t strlcpy(char *dst, const char *src, size_t sz)
{
    size_t slen = strlen(src);
    size_t tlen =sz-1;

    if (slen <= tlen) strcpy(dst, src);
       else if (tlen > 0) {strncpy(dst, src, tlen); dst[tlen] = '\0';}
               else if (tlen == 0) dst[0] = '\0';

    return slen;
}
}
#endif
#ifdef __macos__
#include <pwd.h>
// This is not re-enetrant or mt-safe but it's all we have
//
char *cuserid(char *buff)
{
  static char myBuff[33];
  char *theBuff = (buff ? buff : myBuff);

  struct passwd *thePWD = getpwuid(getuid());
  if (!thePWD)
    {if (buff) *buff = '\0';
     return buff;
    }

  strlcpy(theBuff, thePWD->pw_name, 33);
  return theBuff;
}
#endif
