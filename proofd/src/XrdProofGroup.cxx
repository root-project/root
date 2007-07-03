// @(#)root/proofd:$Name:  $:$Id: XrdProofGroup.cxx,v 1.4 2007/06/21 17:30:21 brun Exp $
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofGroup                                                        //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class describing groups                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdPlatform.h"

#include "XrdProofGroup.h"
#include "XrdProofdTrace.h"

static const char *gTraceID = " ";
extern XrdOucTrace *XrdProofdTrace;

// Local definitions
#define TRACEID gTraceID

// Functions used in scanning hash tables

//__________________________________________________________________________
static int CheckUser(const char *, XrdProofGroup *g, void *u)
{
   // Check if user 'u' is memmebr of group 'grp'

   const char *usr = (const char *)u;

   if (g && usr && g->HasMember(usr))
      // Found the group
      return 1;

   // Check next
   return 0;
}

//__________________________________________________________________________
static int ExportGroup(const char *, XrdProofGroup *g, void *u)
{
   // Add a string describing group 'g' to a global string

   XrdOucString *msg = (XrdOucString *)u;

   if (msg->length() > 0)
      *msg += '\n';

   *msg = g->Name(); *msg += ": ";
   *msg += ", size: ";
   *msg += g->Size();
   *msg += ", members(s): ";
   *msg += g->Members();

   return 0;
}

//__________________________________________________________________________
static int PrintGroup(const char *, XrdProofGroup *g, void *)
{
   // Print info describing group 'g' to stdout

   if (g)
      g->Print();

   return 0;
}

//__________________________________________________________________________
static int AuxFunc(const char *, XrdProofGroup *g, void *s)
{
   // Generic function used for auxiliary purpose

   XrdOucString *opt = (XrdOucString *)s;

   if (!opt || opt->length() <= 0 || (*opt) == "getfirst")
      // Stop going through the table
      return 1;

   if (opt->beginswith("getnextgrp:")) {
      XrdOucString grp("||");
      grp.insert(g->Name(),1);
      if (opt->find(grp) == STR_NPOS) {
         *opt += grp;
         return 1;
      }
   }

   // Process next
   return 0;
}

//__________________________________________________________________________
XrdProofGroup::XrdProofGroup(const char *n, const char *m)
              : fName(n), fMembers(m)
{
   // Constructor

   fSize = 0;
   fActive = 0;
   fPriority = -1;
   fFraction = -1;
   fFracEff = 0;
   fMutex = new XrdOucRecMutex;
}
//__________________________________________________________________________
XrdProofGroup::~XrdProofGroup()
{
   // Destructor

   if (fMutex)
      delete fMutex;
   fMutex = 0;
}

//__________________________________________________________________________
void XrdProofGroup::Print()
{
   // Dump group content

   XrdOucMutexHelper mhp(fMutex); 

   if (fName != "default") {
      XPDPRT("+++ Group: "<<fName<<", size "<<fSize<<" member(s) ("<<fMembers<<")");
      XPDPRT("+++ Priority: "<<fPriority<<", fraction: "<<fFraction);
      XPDPRT("+++ End of Group: "<<fName);
   } else {
      XPDPRT("+++ Group: "<<fName);
      XPDPRT("+++ Priority: "<<fPriority<<", fraction: "<<fFraction);
      XPDPRT("+++ End of Group: "<<fName);
   }
}

//__________________________________________________________________________
void XrdProofGroup::Count(const char *usr, int n)
{ 
   // Modify the active count

   // A username must be defined and an action required
   if (!usr || strlen(usr) == 0 || n == 0)
      return;

   // Reference string
   XrdOucString u(usr);
   u += ",";

   XrdOucMutexHelper mhp(fMutex);

   // If we are named, the user must be a member
   if (fName != "unnamed" && fMembers.find(u) == STR_NPOS)
      return;

   if (n > 0) {

      // Don't count it if already done
      if (fActives.find(u) != STR_NPOS)
         return;

      fActives += u;


   } else {

      // Don't remove it if not active
      if (fActives.find(u) == STR_NPOS)
         return;

      fActives.replace(u,"");
   }

   // Count
   fActive += n;
}

//__________________________________________________________________________
bool XrdProofGroup::HasMember(const char *usr)
{
   // Check if 'usr' is member of this group

   XrdOucMutexHelper mhp(fMutex);
   XrdOucString u(usr); u += ",";
   int iu = fMembers.find(u);
   if (iu != STR_NPOS)
      if (iu == 0 || fMembers[iu-1] == ',')
         return 1;
   return 0;
}

//__________________________________________________________________________
XrdProofGroupMgr::XrdProofGroupMgr(const char *fn)
{
   // Constructor

   ResetIter(); 
   Config(fn);
}

//__________________________________________________________________________
XrdProofGroup *XrdProofGroupMgr::Apply(int (*f)(const char *, XrdProofGroup *,
                                                void *), void *arg)
{
   // Apply function 'f' to the hash table of groups; 'arg' is passed to 'f'
   // in the last argument. After applying 'f', the action depends on the
   // return value with the following rule:
   //         < 0 - the hash table item is deleted.
   //         = 0 - the next hash table item is processed.
   //         > 0 - processing stops and the hash table item is returned.

   return (fGroups.Num() > 0 ? fGroups.Apply(f,arg) : (XrdProofGroup *)0);
}

//__________________________________________________________________________
XrdOucString XrdProofGroupMgr::Export(const char *grp)
{
   // Return a string describing the group

   XrdOucMutexHelper mhp(fMutex); 

   XrdOucString msg;

   if (!grp) {
      fGroups.Apply(ExportGroup, (void *) &msg);
   } else {
      XrdProofGroup *g = fGroups.Find(grp);
      ExportGroup(grp, g, (void *) &msg);
   }

   return msg;
}

//__________________________________________________________________________
void XrdProofGroupMgr::Print(const char *grp)
{
   // Return a string describing the group

   XrdOucMutexHelper mhp(fMutex); 

   if (!grp) {
      fGroups.Apply(PrintGroup, 0);
   } else {
      XrdProofGroup *g = fGroups.Find(grp);
      PrintGroup(grp, g, 0);
   }

   return;
}

//__________________________________________________________________________
XrdProofGroup *XrdProofGroupMgr::GetGroup(const char *grp)
{
   // Returns the instance of for group 'grp.
   // Return 0 in the case the group does not exist

   // If the group is defined and exists, check it 
   if (grp && strlen(grp) > 0)
      return fGroups.Find(grp);
   return (XrdProofGroup *)0;
}

//__________________________________________________________________________
XrdProofGroup *XrdProofGroupMgr::GetUserGroup(const char *usr, const char *grp)
{
   // Returns the instance of the first group to which this user belongs;
   // if grp != 0, return the instance corresponding to group 'grp', if
   // existing and the // user belongs to it.
   // Return 0 in the case the user does not belong to any group or does not
   // belong to 'grp'.

   XrdProofGroup *g = 0;

   // Check inputs
   if (!usr || strlen(usr) <= 0)
      return g;

   // If the group is defined and exists, check it 
   if (grp && strlen(grp) > 0) {
      g = fGroups.Find(grp);
      if (g && (!strncmp(g->Name(),"default",7) || g->HasMember(usr)))
         return g;
      else
         return (XrdProofGroup *)0;
   }

   // Scan the table
   g = fGroups.Apply(CheckUser, (void *)usr);

   // Assign to "default" group if nothing was found
   return ((!g) ? fGroups.Find("default") : g);
}

//__________________________________________________________________________
XrdProofGroup *XrdProofGroupMgr::Next()
{
   // Returns the instance of next group in the pseudo-iterator
   // functionality. To scan over all the groups do the following:
   //         ResetIter();
   //         while ((g = Next())) {
   //            // ... Process group
   //         }
   // Return 0 when there are no more groups

   return fGroups.Apply(AuxFunc,&fIterator);
}

//__________________________________________________________________________
int XrdProofGroupMgr::Config(const char *fn)
{
   // (Ri-)configure the group info using the file 'fn'.
   // Return the number of active groups or -1 in case of error.

   if (!fn || strlen(fn) <= 0) {
      // This call is to reset existing info and remain with
      // the 'default' group only
      XrdOucMutexHelper mhp(fMutex);
      // Reset existing info
      fGroups.Purge();
      // Create "default" group
      fGroups.Add("default", new XrdProofGroup("default"));
      return fGroups.Num();;
   }

   // Did the file changed ?
   if (fCfgFile.fName != fn) {
      fCfgFile.fName = fn;
      XrdProofdAux::Expand(fCfgFile.fName);
      fCfgFile.fMtime = 0;
   }

   // Get the modification time
   struct stat st;
   if (stat(fCfgFile.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "Config: enter: time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fCfgFile.fMtime)
      return 0;

   // Save the modification time
   fCfgFile.fMtime = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fCfgFile.fName.c_str(), "r"))) {
      TRACE(XERR, "Config: cannot open file: "<<fCfgFile.fName<<" (errno:"<<errno<<")");
      return -1;
   }

   // This part must be modified in atomic way
   XrdOucMutexHelper mhp(fMutex);

   // Reset existing info
   fGroups.Purge();

   // Create "default" group
   fGroups.Add("default", new XrdProofGroup("default"));

   // Read now the directives
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      // Remove trailing '\n'
      if (lin[strlen(lin)-1] == '\n') lin[strlen(lin)-1] = '\0';
      // Skip comments or empty lines
      if (lin[0] == '#' || strlen(lin) <= 0) continue;
      // Good line: parse it
      bool gotkey = 0, gotgrp = 0;
      XrdOucString gl(lin), tok, key, group;
      gl.replace(" ",",");
      int from = 0;
      while ((from = gl.tokenize(tok, from, ',')) != -1) {
         if (tok.length() > 0) {
            if (!gotkey) {
               key = tok;
               gotkey = 1;
            } else if (!gotgrp) {
               group = tok;
               gotgrp = 1;
               break;
            }
         }
      }
      // Check consistency
      if (!gotkey || !gotgrp) {
         // Insufficient info
         TRACE(DBG, "Config: incomplete line: " << lin);
         continue;
      }

      // Get linked to the group, if any
      XrdProofGroup *g = fGroups.Find(group.c_str());

      // Action depends on key
      if (key == "group") {
         if (!g)
            // Create new group container
            fGroups.Add(group.c_str(), (g = new XrdProofGroup(group.c_str())));
         while ((from = gl.tokenize(tok, from, ',')) != -1) {
            if (tok.length() > 0)
               // Add group member
               g->AddMember(tok.c_str());
         }
      } else if (key == "property") {
         // Property definition: format of property is
         // property <group> <property_name> <nominal_value> [<effective_value>]
         XrdOucString name;
         int nom=0;
         bool gotname = 0, gotnom = 0;
         while ((from = gl.tokenize(tok, from, ',')) != -1) {
            if (tok.length() > 0) {
               if (!gotname) {
                  name = tok;
                  gotname= 1;
               } else if (!gotnom) {
                  nom = atoi(tok.c_str());
                  gotnom = 1;
                  break;
               }
            }
         }
         if (!gotname || !gotnom) {
            // Insufficient info
            TRACE(DBG, "Config: incomplete property line: " << lin);
            continue;
         }
         if (!g)
            // Create new group container
            fGroups.Add(group.c_str(), (g = new XrdProofGroup(group.c_str())));
         if (name == "priority")
            g->SetPriority(nom);
         if (name == "fraction")
            g->SetFraction(nom);
      }
   }

   // Return the number of active groups
   return fGroups.Num();
}

