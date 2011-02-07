/******************************************************************************/
/*                                                                            */
/*                       X r d A c c G r o u p s . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdAccGroupsCVSID = "$Id$";

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <pwd.h>
#include <string.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/types.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdAcc/XrdAccCapability.hh"
#include "XrdAcc/XrdAccGroups.hh"
#include "XrdAcc/XrdAccPrivs.hh"

// This routine uses non mt-safe routines such as getpwnam, getgrent, etc.
// We do so because we are the only ones using these routines and thet are
// protected by the Group_Build_Context mutex. Anyway, the re-enterant
// version of the same routines are not mt-safe in any case, sigh.

// Additionally, this routine does not support a user in more than
// NGROUPS_MAX groups. This is a standard unix limit defined in limits.h.
  
/******************************************************************************/
/*                  G l o b a l   G r o u p s   O b j e c t                   */
/******************************************************************************/

// There is only one Groups object that handles group memberships. Others
// needing access to this object should declare an extern to this object.
//
XrdAccGroups XrdAccGroupMaster;
  
/******************************************************************************/
/*          G r o u p   C o n s t r u c t i o n   A r g u m e n t s           */
/******************************************************************************/
  
struct XrdAccGroupArgs {const char   *user;
                        const char   *host;
                        int           gtabi;
                        const char   *Gtab[NGROUPS_MAX];
                       };

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdAccGroups::XrdAccGroups()
{

// Do standard initialization
//
   retrancnt = 0; 
   HaveGroups = 0;
   HaveNetGroups = 0;
   options = No_Group_Opt;
   domain = 0;
   LifeTime = 60*60*12;
}

/******************************************************************************/
/*                               A d d N a m e                                */
/******************************************************************************/
  
char *XrdAccGroups::AddName(const XrdAccGroupType gtype, const char *name)
{
   char *np;
   XrdOucHash<char> *hp;

// Prepare to add a group name
//
   if (gtype == XrdAccNetGroup) {hp = &NetGroup_Names; HaveNetGroups = 1;}
      else {hp = &Group_Names; HaveGroups = 1;}

// Lock the Name hash table
//
   Group_Name_Context.Lock();

// Add a name into the name hash table. We need to only keep a single
// read/only copy of the group name to speed multi-threading.
//
   if (!(np = hp->Find(name)))
      {hp->Add(name, 0, 0, Hash_data_is_key);
       if (!(np = hp->Find(name)))
           cerr <<"XrdAccGroups: Unable to add group " <<name <<endl;
      }

// All done.
//
   Group_Name_Context.UnLock();
   return np;
}

/******************************************************************************/
/*                              F i n d N a m e                               */
/******************************************************************************/
  
char *XrdAccGroups::FindName(const XrdAccGroupType gtype, const char *name)
{
   char *np;

// Lock the Name hash table
//
   Group_Name_Context.Lock();

// Lookup the actual name in the hash table
//
   if (gtype == XrdAccNetGroup) np = NetGroup_Names.Find(name);
      else np = Group_Names.Find(name);

// All done.
//
   Group_Name_Context.UnLock();
   return np;
}
  
/******************************************************************************/
/*                                                                            */
/*                          G r o u p s ( u s e r )                           */
/*                                                                            */
/******************************************************************************/
  
XrdAccGroupList *XrdAccGroups::Groups(const char *user)
{
struct group  *gr;
struct passwd *pw;
char **cp;
XrdAccGroupList *glist;
int   gtabi;
char *Gtab[NGROUPS_MAX];

// Check if we have any referenced groups
//
   if (!HaveGroups) return (XrdAccGroupList *)0;


// Check if we already have this user in the group cache. Since we may be
// modifying the cache, we need to have exclusive control over it. We must
// copy the group cache because the original may be deleted at any time.
//
   Group_Cache_Context.Lock();
   if ((glist = Group_Cache.Find(user)))
      {if (glist->First()) glist = new XrdAccGroupList(*glist);
          else glist = 0;
       Group_Cache_Context.UnLock();
       return glist;
      }
   Group_Cache_Context.UnLock();

// If the user has no password file entry, then we have no groups for user.
// All code that tries to construct a group list is protected by the
// Group_Build_Context mutex.
//
   Group_Build_Context.Lock();
   if ( (pw = getpwnam(user)) == NULL)
      {Group_Build_Context.UnLock();
       return (XrdAccGroupList *)0;
      }

// Build first entry for the primary group. We will ignore the primary group
// listing later. We do this to ensure that the user has at least one group
// regardless of what the groups file actually says.
//
   gtabi = addGroup(user, pw->pw_gid, 0, Gtab, 0);

// Now run through all of the group entries getting the list of user's groups
// Do this only when Primary_Only is not turned on (i.e., SVR5 semantics)
//
   if (!(options & Primary_Only))
      {
       setgrent() ;
       while ((gr = getgrent()))
            {
             if (pw->pw_gid == gr->gr_gid) continue; /*Already have this one.*/
             for (cp = gr->gr_mem; cp && *cp; cp++)
                 if (strcmp(*cp, user) == 0)
                    gtabi = addGroup(user, gr->gr_gid,
                               Dotran(gr->gr_gid,gr->gr_name),
                                     Gtab, gtabi);
            }
       endgrent();
      }

// All done with non mt-safe routines
//
   Group_Build_Context.UnLock();

// Allocate a new GroupList object
//
   glist = new XrdAccGroupList(gtabi, (const char **)Gtab);

// Add this user to the group cache to speed things up the next time
//
   Group_Cache_Context.Lock();
   Group_Cache.Add(user, glist, LifeTime);
   Group_Cache_Context.UnLock();

// Return a copy of the group list since the original may be deleted
//
   if (!gtabi) return (XrdAccGroupList *)0;
   return new XrdAccGroupList(gtabi, (const char **)Gtab);
}
 
/******************************************************************************/
/*                 N e t G r o u p s ( u s e r ,   h o s t )                  */
/******************************************************************************/
  
XrdAccGroupList *XrdAccGroups::NetGroups(const char *user, const char *host)
{
XrdAccGroupList *glist;
int   i, j;
char uh_key[MAXHOSTNAMELEN+96];
struct XrdAccGroupArgs GroupTab;
int XrdAccCheckNetGroup(const char *netgroup, char *key, void *Arg);

// Check if we have any Netgroups
//
   if (!HaveNetGroups) return (XrdAccGroupList *)0;

// Construct the key for this user
//
   i = strlen(user); j = strlen(host);
   if (i+j+2 > (int)sizeof(uh_key)) return (XrdAccGroupList *)0;
   strcpy(uh_key, user);
   uh_key[i] = '@';
   strcpy(&uh_key[i+1], host);

// Check if we already have this user in the group cache. Since we may be
// modifying the cache, we need to have exclusive control over it. We must
// copy the group cache entry because the original may be deleted at any time.
//
   NetGroup_Cache_Context.Lock();
   if ((glist = NetGroup_Cache.Find(uh_key)))
      {if (glist->First()) glist = new XrdAccGroupList(*glist);
          else glist = 0;
       NetGroup_Cache_Context.UnLock();
       return glist;
      }
   NetGroup_Cache_Context.UnLock();

// For each known netgroup, check to see if the user is in the netgroup.
//
   GroupTab.user  = user;
   GroupTab.host  = host;
   GroupTab.gtabi = 0;
   Group_Name_Context.Lock();
   NetGroup_Names.Apply(XrdAccCheckNetGroup, (void *)&GroupTab);
   Group_Name_Context.UnLock();

// Allocate a new GroupList object
//
   glist = new XrdAccGroupList(GroupTab.gtabi,
                           (const char **)GroupTab.Gtab);

// Add this user to the group cache to speed things up the next time
//
   NetGroup_Cache_Context.Lock();
   NetGroup_Cache.Add((const char *)uh_key, glist, LifeTime);
   NetGroup_Cache_Context.UnLock();

// Return a copy of the group list
//
   if (!GroupTab.gtabi) return (XrdAccGroupList *)0;
   return new XrdAccGroupList(GroupTab.gtabi,
                          (const char **)GroupTab.Gtab);
}

/******************************************************************************/
/*                            P u r g e C a c h e                             */
/******************************************************************************/

void XrdAccGroups::PurgeCache()
{

// Purge the group cache
//
   Group_Cache_Context.Lock();
   Group_Cache.Purge();
   Group_Cache_Context.UnLock();

// Purge the netgroup cache
//
   NetGroup_Cache_Context.Lock();
   NetGroup_Cache.Purge();
   NetGroup_Cache_Context.UnLock();
}
  
/******************************************************************************/
/*                                R e t r a n                                 */
/******************************************************************************/
  
int XrdAccGroups::Retran(const gid_t gid)
{
    if ((int)gid < 0) retrancnt = 0;
       else {if (retrancnt > (int)(sizeof(retrangid)/sizeof(gid_t))) return -1;
             retrangid[retrancnt++] = gid;
            }
    return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
  
/******************************************************************************/
/*                              a d d G r o u p                               */
/******************************************************************************/
  
int XrdAccGroups::addGroup(const char *user, const gid_t gid, char *gname,
                           char **Gtab, int gtabi)
{
   char *gp;

// Check if we have room to add another group. We can squeek by such errors
// because all it means is that the user normally has fewer privs (which is
// not always true, sigh).
//
   if (gtabi >= NGROUPS_MAX)
      {if (gtabi == NGROUPS_MAX)
          cerr <<"XrdAccGroups: More than " <<gtabi <<"groups for " <<user <<endl;
       return gtabi;
      }

// See if we should lookup the group name. The caller had better be holding the
// Group_Build_Context mutex.
//
if (!gname || !gname[0])
   {struct group *gp;
    if ((gp = getgrgid(gid)) == NULL) return gtabi;
             else gname = gp->gr_name;
   }

// Check if we have this group registered. Only a handful of groups are
// actually relevant. Ignore the unreferenced groups. If registered, we
// need the persistent name because of multi-threading issues.
//
   if (!(gp = Group_Names.Find(gname)) ) return gtabi;

// Add the groupname to the table of groups for the user
//
   Gtab[gtabi++] = gp;
   return gtabi;
}

/******************************************************************************/
/*                                D o t r a n                                 */
/******************************************************************************/
  
char *XrdAccGroups::Dotran(const gid_t gid, char *gname)
{
     int i;

    // See if the groupname needs to be retranslated. This is necessary
    // When multiple groups share the same gid due to NIS constraints.
    //
    for (i = 0; i < retrancnt; i++) if (retrangid[i] == gid) return (char *)0;
    return gname;
}

/******************************************************************************/
/*                    E x t e r n a l   F u n c t i o n s                     */
/******************************************************************************/

/******************************************************************************/
/*                   o o a c c _ C h e c k N e t G r o u p                    */
/******************************************************************************/
  
int XrdAccCheckNetGroup(const char *netgroup, char *key, void *Arg)
{
    struct XrdAccGroupArgs *grp = static_cast<struct XrdAccGroupArgs *>(Arg);

    // Check if this netgroup, user, host, domain combination exists.
    //
    if (innetgr(netgroup, (const char *)grp->host, (const char *)grp->user,
                XrdAccGroupMaster.Domain()))
       {if (grp->gtabi >= NGROUPS_MAX) 
           {if (grp->gtabi == NGROUPS_MAX)
               cerr <<"XrdAccGroups: More than " <<grp->gtabi <<"netgroups for " <<grp->user <<endl;
            return 1;
           }

        // Add the groupname into the groupname hash table. We have already
        // been passed the read/only copy of the name.
        //
        grp->Gtab[grp->gtabi] = netgroup; grp->gtabi++;
       }
    return 0;
}
