#ifndef __LINK_MATCH__
#define __LINK_MATCH__
/******************************************************************************/
/*                                                                            */
/*                       X r d L i n k M a t c h . h h                        */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <strings.h>
#include <stdlib.h>
  
class XrdLinkMatch
{
public:


int                Match(const char *uname, int unlen,
                         const char *hname, int hnlen);
inline int         Match(const char *uname, int unlen,
                         const char *hname)
                        {return Match(uname, unlen, hname, strlen(hname));}

// Target: [<user>][*][@[<hostpfx>][*][<hostsfx>]]
//
       void        Set(const char *target);

             XrdLinkMatch(const char *target=0)
                         {Uname = HnameL = HnameR = 0;
                          Unamelen = Hnamelen = 0;
                          if (target) Set(target);
                         }

            ~XrdLinkMatch() {}

private:

char               Buff[256];
int                Unamelen;
char              *Uname;
int                HnamelenL;
char              *HnameL;
int                HnamelenR;
char              *HnameR;
int                Hnamelen;
};
#endif
