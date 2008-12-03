#ifndef __OUC_TLIST__
#define __OUC_TLIST__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c T L i s t . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdlib.h>
#include <string.h>
#include <strings.h>
  
class XrdOucTList
{
public:

XrdOucTList *next;
char        *text;
int          val;

             XrdOucTList(const char *tval=0, int num=0, XrdOucTList *np=0)
                        {text = (tval ? strdup(tval) : 0); val=num; next=np;}

            ~XrdOucTList() {if (text) free(text);}
};

class XrdOucTListHelper
{
public:

XrdOucTList **Anchor;

      XrdOucTListHelper(XrdOucTList **p) : Anchor(p) {}
     ~XrdOucTListHelper() {XrdOucTList *tp;
                           while((tp = *Anchor))
                                {*Anchor = tp->next; delete tp;}
                          }
};
#endif
