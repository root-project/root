#ifndef _ACC_GROUPS_H
#define _ACC_GROUPS_H
/******************************************************************************/
/*                                                                            */
/*                       X r d A c c G r o u p s . h h                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

#include <grp.h>
#include <limits.h>

#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                       X r d A c c G r o u p L i s t                        */
/******************************************************************************/
  
class XrdAccGroupList
{
public:

const char *First() {return grouptab[0];}

const char *Next()  {if (grouptab[nextgroup]) return grouptab[nextgroup++];
                     return (const char *)0;
                    }

      void  Reset() {nextgroup = 0;}

      XrdAccGroupList(const int cnt=0, const char **gtable=0)
                     {int j = (cnt > NGROUPS_MAX ? NGROUPS_MAX : cnt);
             if (cnt) memcpy((void *)grouptab, (const void *)gtable,
                             (size_t)(j * sizeof(char *)));
                      memset((void *)&grouptab[cnt], 0,
                             (size_t)((NGROUPS_MAX-j+1)*sizeof(char *)));
                      nextgroup = 0;
                     }

      XrdAccGroupList(XrdAccGroupList & rv)
            {memcpy((void *)grouptab,(const void *)rv.grouptab,sizeof(grouptab));
             nextgroup = 0;
            }

     ~XrdAccGroupList() {}

private:
const char  *grouptab[NGROUPS_MAX+1];
      int    nextgroup;
};

/******************************************************************************/
/*                        G r o u p s   O p t i o n s                         */
/******************************************************************************/

enum XrdAccGroups_Options { Primary_Only   = 0x0001,
                            Groups_Debug   = 0x8000,
                            No_Group_Opt   = 0x0000
                          };

/******************************************************************************/
/*                           G r o u p   T y p e s                            */
/******************************************************************************/
  
enum XrdAccGroupType      {XrdAccNoGroup = 0, XrdAccUnixGroup, XrdAccNetGroup};
  
/******************************************************************************/
/*                          X r d A c c G r o u p s                           */
/******************************************************************************/
  
class XrdAccGroups
{
public:

// Domain() returns whatever we have for the NIS domain.
//
const char       *Domain() {return domain;}

// AddName() registers a name in the static name table. This allows us to
// avoid copying the strings a table points to when returning a table copy.
// If the name was added successfully, a pointer to the name is returned.
// Otherwise, zero is returned.
//
char             *AddName(const XrdAccGroupType gtype, const char *name);

// FindName() looks up a name in the static name table.
//
char             *FindName(const XrdAccGroupType gtype, const char *name);

// Groups() returns all of the relevant groups that a user belongs to. A
// null pointer may be returned if no groups are applicable.
//
XrdAccGroupList *Groups(const char *user);

// NetGroups() returns all of the relevant netgroups that the user/host
// combination belongs to. A null pointer may be returned is no netgroups
// are applicable.
//
XrdAccGroupList *NetGroups(const char *user, const char *host);

// PurgeCache() removes all entries in the various caches. It is called
// whenever a new set of access tables has been instantiated.
//
void             PurgeCache();

// Use by the configuration object to set group id's that must be looked up.
//
int              Retran(const gid_t gid);

// Use by the configuration object to establish the netgroup domain.
//
void             SetDomain(const char *dname) {domain = dname;}

// Used by the configuration object to set the cache lifetime.
//
void             SetLifetime(const int seconds) {LifeTime = (int)seconds;}

// Used by the configuration object to set various options
//
void             SetOptions(XrdAccGroups_Options opts) {options = opts;}

      XrdAccGroups();

     ~XrdAccGroups() {}  // The group object never gets deleted!!

private:

int addGroup(const char *user, const gid_t gid, char *gname,
                   char **Gtab, int gtabi);
char *Dotran(const gid_t gid, char *gname);

gid_t       retrangid[128];  // Up to 128 retranslatable gids
int         retrancnt;       // Number of used entries
time_t      LifeTime;        // Seconds we can keep something in the cache
const char *domain;          // NIS netgroup domain to use

XrdAccGroups_Options options;// Various option values.
int         HaveGroups;
int         HaveNetGroups;

XrdSysMutex  Group_Build_Context, Group_Name_Context;
XrdSysMutex  Group_Cache_Context, NetGroup_Cache_Context;

XrdOucHash<XrdAccGroupList> NetGroup_Cache;
XrdOucHash<XrdAccGroupList>    Group_Cache;
XrdOucHash<char>               Group_Names;
XrdOucHash<char>            NetGroup_Names;
};
#endif
