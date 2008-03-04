/******************************************************************************/
/*                                                                            */
/*                        X r d O u c T r a c e . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdOucTraceCVSID = "$Id$";

#include "XrdOuc/XrdOucTrace.hh"

/******************************************************************************/
/*                               b i n 2 h e x                                */
/******************************************************************************/
  
char *XrdOucTrace::bin2hex(char *inbuff, int dlen, char *buff)
{
    static char hv[] = "0123456789abcdef";
    static char xbuff[56];
    char *outbuff = (buff ? buff : xbuff);
    int i;
    if (dlen > 24) dlen = 24;
    for (i = 0; i < dlen; i++) {
        *outbuff++ = hv[(inbuff[i] >> 4) & 0x0f];
        *outbuff++ = hv[ inbuff[i]       & 0x0f];
        if ((i & 0x03) == 0x03 || i+1 == dlen) *outbuff++ = ' ';
        }
     *outbuff = '\0';
     return xbuff;
}
