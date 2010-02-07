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
union
{
long long    dval;
int          ival[2];
short        sval[4];
char         cval[8];
int          val;
};

             XrdOucTList(const char *tval, long long *dv,XrdOucTList *np=0)
                        {next=np; text = (tval ? strdup(tval) : 0); dval=*dv;}

             XrdOucTList(const char *tval=0, int num=0, XrdOucTList *np=0)
                        {next=np; text = (tval ? strdup(tval) : 0); val=num;}

             XrdOucTList(const char *tval, int   iv[2], XrdOucTList *np=0)
                        {next=np; text = (tval ? strdup(tval) : 0);
                         memcpy(sval, iv, sizeof(ival));}

             XrdOucTList(const char *tval, short sv[4], XrdOucTList *np=0)
                        {next=np; text = (tval ? strdup(tval) : 0);
                         memcpy(sval, sv, sizeof(sval));}

             XrdOucTList(const char *tval, char  cv[8], XrdOucTList *np=0)
                        {text = (tval ? strdup(tval) : 0); next=np;
                         memcpy(cval, cv, sizeof(cval));}

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
