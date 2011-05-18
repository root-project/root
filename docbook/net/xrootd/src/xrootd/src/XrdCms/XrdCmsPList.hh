#ifndef XRDCMSPLIST__H
#define XRDCMSPLIST__H
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s P L i s t . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>
#include <strings.h>
#include <stdlib.h>

#include "XrdCms/XrdCmsTypes.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                     C l a s s   X r d C m s P I n f o                      */
/******************************************************************************/
  
class XrdCmsPInfo
{
public:
   SMask_t rovec;
   SMask_t rwvec;
   SMask_t ssvec;

inline int  And(const SMask_t mask)
               {return ((rovec &= mask)|(rwvec &= mask)|(ssvec &= mask)) != 0;}

inline void Or(const XrdCmsPInfo *pi)
               {rovec |=  pi->rovec; rwvec |=  pi->rwvec; ssvec |=  pi->ssvec;}

inline void Set(const XrdCmsPInfo *pi)
               {rovec  =  pi->rovec; rwvec  =  pi->rwvec; ssvec  =  pi->ssvec;}

           XrdCmsPInfo() {rovec = rwvec = ssvec = 0;}
          ~XrdCmsPInfo() {}
           XrdCmsPInfo   &operator =(const XrdCmsPInfo &rhs)
                        {Set(&rhs); return *this;}
};
 
/******************************************************************************/
/*                     C l a s s   X r d C m s P L i s t                      */
/******************************************************************************/
  
class XrdCmsPList
{
public:
friend class XrdCmsPList_Anchor;

inline XrdCmsPList   *Next() {return next;}
inline char          *Path() {return pathname;}
const  char          *PType();

       XrdCmsPList(const char *pname="", XrdCmsPInfo *pi=0)
                  : next(0), pathname(strdup(pname)), pathlen(strlen(pname)),
                    pathtype(0) {if (pi) pathmask.Set(pi);}

      ~XrdCmsPList() {if (pathname) free(pathname);}

private:

XrdCmsPInfo     pathmask;
XrdCmsPList    *next;
char           *pathname;
int             pathlen;
char            pathtype;
char            reserved[3];
};

class XrdCmsPList_Anchor
{
public:

inline void         Lock() {mutex.Lock();}
inline void       UnLock() {mutex.UnLock();}

       int          Add(const char *pname, XrdCmsPInfo *pinfo);

inline void         Empty(XrdCmsPList *newlist=0)
                    {Lock();
                     XrdCmsPList *p = next;
                     while(p) {next = p->next; delete p; p = next;}
                     next = newlist;
                     UnLock();
                    }

       int          Find(const char *pname, XrdCmsPInfo &masks);

inline XrdCmsPList *First() {return next;}

       SMask_t      Insert(const char *pname, XrdCmsPInfo *pinfo);

inline int          NotEmpty() {return next != 0;}

       void         Remove(SMask_t mask);

const  char        *Type(const char *pname);

inline XrdCmsPList *Zorch(XrdCmsPList *newlist=0)
                   {Lock();
                    XrdCmsPList *p = next;
                    next = newlist;
                    UnLock();
                    return p;
                   }

       XrdCmsPList_Anchor() {next = 0;}

      ~XrdCmsPList_Anchor() {Empty();}

private:

XrdSysMutex   mutex;
XrdCmsPList  *next;
};
#endif
