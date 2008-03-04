#ifndef __OOUC_NLIST__
#define __OOUC_NLIST__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c N L i s t . h h                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#ifndef WIN32
#include <strings.h>
#else
#include "XrdSys/XrdWin32.hh"
#endif
#include <stdlib.h>
#include "XrdSys/XrdSysPthread.hh"
  
class XrdOucNList
{
public:

inline int         Flag() {return flags;}
inline XrdOucNList *Next() {return next;}

       int         NameOK(const char *pd, const int pl);
inline int         NameOK(const char *pd)
                         {return NameOK(pd, strlen(pd));}

inline void        Set(int fval) {flags = fval;}

             XrdOucNList(const char *name="", int nvals=0);

            ~XrdOucNList()
                  {if (nameL) free(nameL);}

friend class XrdOucNList_Anchor;

private:

XrdOucNList        *next;
int                namelenL;
char              *nameL;
int                namelenR;
char              *nameR;
int                flags;
};

class XrdOucNList_Anchor : public XrdOucNList
{
public:

inline void        Lock() {mutex.Lock();}
inline void      UnLock() {mutex.UnLock();}

inline void        Empty(XrdOucNList *newlist=0)
                   {Lock();
                    XrdOucNList *p = next;
                    while(p) {next = p->next; delete p; p = next;}
                    next = newlist;
                    UnLock();
                   }

inline XrdOucNList *Find(const char *name)
                   {int nlen = strlen(name);
                    Lock();
                    XrdOucNList *p = next;
                    while(p) {if (p->NameOK(name, nlen)) break;
                              p=p->next;
                             }
                    UnLock();
                    return p;
                   }

inline XrdOucNList *First() {return next;}

inline void        Insert(XrdOucNList *newitem)
                   {Lock();
                    newitem->next = next; next = newitem; 
                    UnLock();
                   }

inline int         NotEmpty() {return next != 0;}

inline XrdOucNList *Pop()
                   {XrdOucNList *np;
                    Lock();
                    if ((np = next)) next = np->next;
                    UnLock();
                    return np;
                   }

       void        Replace(const char *name, int nval);

       void        Replace(XrdOucNList *item);

                   // Warning: You must manually lock the object before swap
inline void        Swap(XrdOucNList_Anchor &other)
                       {XrdOucNList *savenext = next;
                        next = other.First();
                        other.Zorch(savenext);
                       }

inline void        Zorch(XrdOucNList *newnext=0) {next = newnext;}

private:

XrdSysMutex         mutex;
};
#endif
