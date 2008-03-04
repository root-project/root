#ifndef __OLB_PLIST__H
#define __OLB_PLIST__H
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b P L i s t . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>
#include <strings.h>
#include <stdlib.h>

#include "XrdOlb/XrdOlbTypes.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                     S t r u c t   o o l b _ P I n f o                      */
/******************************************************************************/
  
class XrdOlbPInfo
{
public:
   SMask_t rovec;
   SMask_t rwvec;
   SMask_t ssvec;

inline int  And(const SMask_t mask)
               {return ((rovec &= mask)|(rwvec &= mask)|(ssvec &= mask)) != 0;}

inline void Or(const XrdOlbPInfo *pi)
               {rovec |=  pi->rovec; rwvec |=  pi->rwvec; ssvec |=  pi->ssvec;}

inline void Set(const XrdOlbPInfo *pi)
               {rovec  =  pi->rovec; rwvec  =  pi->rwvec; ssvec  =  pi->ssvec;}

           XrdOlbPInfo() {rovec = rwvec = ssvec = 0;}
          ~XrdOlbPInfo() {}
           XrdOlbPInfo   &operator =(const XrdOlbPInfo &rhs)
                        {Set(&rhs); return *this;}
};
 
/******************************************************************************/
/*                      C l a s s   o o l b _ P L i s t                       */
/******************************************************************************/
  
class XrdOlbPList
{
public:
friend class XrdOlbPList_Anchor;

inline XrdOlbPList    *Next() {return next;}
inline char          *Path() {return pathname;}
const  char          *PType();

       XrdOlbPList(const char *pname="", XrdOlbPInfo *pi=0)
                 {next     = 0;
                  pathlen  = strlen(pname);
                  pathname = strdup(pname);
                  if (pi) pathmask.Set(pi);
                 }

      ~XrdOlbPList() {if (pathname) free(pathname);}

private:

XrdOlbPList     *next;
int             pathlen;
char           *pathname;
XrdOlbPInfo      pathmask;
};

class XrdOlbPList_Anchor
{
public:

inline void        Lock() {mutex.Lock();}
inline void      UnLock() {mutex.UnLock();}

inline void        Empty(XrdOlbPList *newlist=0)
                   {Lock();
                    XrdOlbPList *p = next;
                    while(p) {next = p->next; delete p; p = next;}
                    next = newlist;
                    UnLock();
                   }

       int         Find(const char *pname, XrdOlbPInfo &masks);

inline XrdOlbPList *First() {return next;}

       SMask_t     Insert(const char *pname, XrdOlbPInfo *pinfo);

inline int         NotEmpty() {return next != 0;}

       void        Remove(SMask_t mask);

const  char       *Type(const char *pname);

inline XrdOlbPList *Zorch(XrdOlbPList *newlist=0)
                   {Lock();
                    XrdOlbPList *p = next;
                    next = newlist;
                    UnLock();
                    return p;
                   }

       XrdOlbPList_Anchor() {next = 0;}

      ~XrdOlbPList_Anchor() {Empty();}

private:

XrdSysMutex   mutex;
XrdOlbPList  *next;
};
#endif
