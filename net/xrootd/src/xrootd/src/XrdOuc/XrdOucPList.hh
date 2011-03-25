#ifndef __OUC_PLIST__
#define __OUC_PLIST__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c P L i s t . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <strings.h>
#include <stdlib.h>
  
class XrdOucPList
{
public:

inline int                 Attr() {return attrs;}
inline unsigned long long  Flag() {return flags;}
inline XrdOucPList        *Next() {return next;}
inline char               *Path() {return path;}
inline int                 Plen() {return pathlen;}

inline int          PathOK(const char *pd, const int pl)
                          {return pl >= pathlen && !strncmp(pd, path, pathlen);}

inline void         Set(int                aval) {attrs = aval;}
inline void         Set(unsigned long long fval) {flags = fval;}

             XrdOucPList(const char *pd="", unsigned long long fv=0)
                        : flags(fv), next(0),  path(strdup(pd)),
                          pathlen(strlen(pd)), attrs(0) {}
            ~XrdOucPList()
                  {if (path) free(path);}

friend class XrdOucPListAnchor;

private:

unsigned long long flags;
XrdOucPList       *next;
char              *path;
int                pathlen;
int                attrs;
};

class XrdOucPListAnchor : public XrdOucPList
{
public:

inline XrdOucPList *About(const char *pathname)
                   {int plen = strlen(pathname); 
                    XrdOucPList *p = next;
                    while(p) {if (p->PathOK(pathname, plen)) break;
                              p=p->next;
                             }
                    return p;
                   }

inline void        Default(unsigned long long x) {dflts = x;}

inline void        Empty(XrdOucPList *newlist=0)
                   {XrdOucPList *p = next;
                    while(p) {next = p->next; delete p; p = next;}
                    next = newlist;
                   }

inline unsigned long long  Find(const char *pathname)
                   {int plen = strlen(pathname); 
                    XrdOucPList *p = next;
                    while(p) {if (p->PathOK(pathname, plen)) break;
                              p=p->next;
                             }
                    return (p ? p->flags : dflts);
                   }

inline XrdOucPList *Match(const char *pathname)
                   {int plen = strlen(pathname); 
                    XrdOucPList *p = next;
                    while(p) {if (p->pathlen == plen 
                              &&  !strcmp(p->path, pathname)) break;
                              p=p->next;
                             }
                    return p;
                   }

inline XrdOucPList *First() {return next;}

inline void        Insert(XrdOucPList *newitem)
                   {XrdOucPList *pp = 0, *cp = next;
                    while(cp && newitem->pathlen < cp->pathlen) {pp=cp;cp=cp->next;}
                    if (pp) {newitem->next = pp->next; pp->next = newitem;}
                       else {newitem->next = next;         next = newitem;}
                   }

inline int         NotEmpty() {return next != 0;}

                   XrdOucPListAnchor(unsigned long long dfx=0) {dflts = dfx;}
                  ~XrdOucPListAnchor() {}

private:

unsigned long long dflts;
};
#endif
