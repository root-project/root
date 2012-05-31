// @(#)root/proofd:$Id$
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
   fPriority = -1;
   fFraction = -1;
   fFracEff = 0;
   fMutex = new XrdSysRecMutex;
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
   XPDLOC(GMGR, "Group::Print")

   XrdSysMutexHelper mhp(fMutex); 

   if (fName != "default") {
      TRACE(ALL, "+++ Group: "<<fName<<", size "<<fSize<<" member(s) ("<<fMembers<<")");
      TRACE(ALL, "+++ Priority: "<<fPriority<<", fraction: "<<fFraction);
      TRACE(ALL, "+++ End of Group: "<<fName);
   } else {
      TRACE(ALL, "+++ Group: "<<fName);
      TRACE(ALL, "+++ Priority: "<<fPriority<<", fraction: "<<fFraction);
      TRACE(ALL, "+++ End of Group: "<<fName);
   }
}

//__________________________________________________________________________
void XrdProofGroup::Count(const char *usr, int n)
{
   // Modify the active count

   // A username must be defined and an action required
   if (!usr || strlen(usr) == 0 || n == 0)
      return;

   XrdSysMutexHelper mhp(fMutex);

   XrdProofGroupMember *m = fActives.Find(usr);
   if (!m) {
      // Create a new active user
      m = new XrdProofGroupMember(usr);
      fActives.Add(usr, m);
   }

   // Count
   if (m) {
      m->Count(n);
      // If no active sessions left, remove from active
      if (m->Active() <= 0) {
         fActives.Del(usr);
         delete m;
      }
   }
}

//__________________________________________________________________________
int XrdProofGroup::Active(const char *usr)
{
   // Return the number of active groups (usr = 0) or the number of
   // active sessions for user 'usr'

   XrdSysMutexHelper mhp(fMutex);

   int na = 0;
   if (!usr || strlen(usr) == 0) {
      na = fActives.Num();
   } else {
      XrdProofGroupMember *m = fActives.Find(usr);
      if (m) na = m->Active();
   }
   // Done
   return na;
}

//__________________________________________________________________________
bool XrdProofGroup::HasMember(const char *usr)
{
   // Check if 'usr' is member of this group

   XrdSysMutexHelper mhp(fMutex);
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

   XrdSysMutexHelper mhp(fMutex); 

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

   XrdSysMutexHelper mhp(fMutex); 

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
   if (grp && strlen(grp) > 0) {
      XrdSysMutexHelper mhp(fMutex);
      return fGroups.Find(grp);
   }
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

   XrdSysMutexHelper mhp(fMutex);

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
   // (Re-)configure the group info using the file 'fn'.
   // Return the number of active groups or -1 in case of error.
   XPDLOC(GMGR, "GroupMgr::Config")

   if ((!fn || strlen(fn) <= 0)) {
      if (fCfgFile.fName != fn) {
         // This call is to reset existing info and remain with
         // the 'default' group only
         XrdSysMutexHelper mhp(fMutex);
         // Reset existing info
         fGroups.Purge();
         // Create "default" group
         fGroups.Add("default", new XrdProofGroup("default"));
         // Reset fCfgFile
         fCfgFile.fName = "";
         fCfgFile.fMtime = 0;
      }
      return fGroups.Num();
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
   TRACE(DBG, "enter: time of last modification: " << st.st_mtime);

   // Nothing to do if the file did not change
   if (st.st_mtime <= fCfgFile.fMtime) return fGroups.Num();
   
   // Save the modification time
   fCfgFile.fMtime = st.st_mtime;

   // This part must be modified in atomic way
   XrdSysMutexHelper mhp(fMutex);

   // Reset existing info
   fGroups.Purge();

   // Create "default" group
   fGroups.Add("default", new XrdProofGroup("default"));

   // Read now the directives (recursive processing of 'include sub-file'
   // in here)
   if (ParseInfoFrom(fCfgFile.fName.c_str()) != 0) {
      TRACE(XERR, "problems parsing config file "<<fCfgFile.fName);
   }
   
   // Notify the content
   Print(0);

   // Return the number of active groups
   return fGroups.Num();
}

//__________________________________________________________________________
int XrdProofGroupMgr::ParseInfoFrom(const char *fn)
{
   // Parse config information from the open file 'fin'. Can be called
   // recursively following 'include sub-file' lines.
   // Return 0 or -1 in case of error.
   XPDLOC(GMGR, "GroupMgr::ParseInfoFrom")

   // Check input
   if (!fn || strlen(fn) <= 0) {
      TRACE(XERR, "file name undefined!");
      return -1;
   }

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fn, "r"))) {
      TRACE(XERR, "cannot open file: "<<fn<<" (errno:"<<errno<<")");
      return -1;
   }

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
         TRACE(DBG, "incomplete line: " << lin);
         continue;
      }

      if (key == "include") {
         // File to be included in the parsing
         XrdOucString subfn = group;
         // Expand the path
         XrdProofdAux::Expand(subfn);
         // Process it
         if (ParseInfoFrom(subfn.c_str()) != 0) {
            TRACE(XERR, "problems parsing included file "<<subfn);
         }
         continue;
      }

      if (key == "priorityfile") {
         // File from which (updated) priorities are read
         fPriorityFile.fName = group;
         XrdProofdAux::Expand(fPriorityFile.fName);
         fPriorityFile.fMtime = 0;
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
            TRACE(DBG, "incomplete property line: " << lin);
            continue;
         }
         if (!g)
            // Create new group container
            fGroups.Add(group.c_str(), (g = new XrdProofGroup(group.c_str())));
         if (name == "priority")
            g->SetPriority((float)nom);
         if (name == "fraction")
            g->SetFraction(nom);
      }
   }
   // Close this file
   fclose(fin);
   // Done
   return 0;
}

//__________________________________________________________________________
int XrdProofGroupMgr::ReadPriorities()
{
   // Read update priorities from the file defined at configuration time.
   // Return 1 if the file did not change, 0 if the file has been read
   // correctly, or -1 in case of error.
   XPDLOC(GMGR, "GroupMgr::ReadPriorities")

   // Get the modification time
   struct stat st;
   if (stat(fPriorityFile.fName.c_str(), &st) != 0)
      return -1;
   TRACE(DBG, "time of last modification: " << st.st_mtime);

   // File should be loaded only once
   if (st.st_mtime <= fPriorityFile.fMtime) {
      TRACE(DBG, "file unchanged since last reading - do nothing ");
      return 1;
   }

   // Save the modification time
   fPriorityFile.fMtime = st.st_mtime;

   // Open the defined path.
   FILE *fin = 0;
   if (!(fin = fopen(fPriorityFile.fName.c_str(), "r"))) {
      TRACE(XERR, "cannot open file: "<<fPriorityFile.fName<<" (errno:"<<errno<<")");
      return -1;
   }

   // This part must be modified in atomic way
   XrdSysMutexHelper mhp(fMutex);

   // Read now the directives
   char lin[2048];
   while (fgets(lin,sizeof(lin),fin)) {
      // Remove trailing '\n'
      if (lin[strlen(lin)-1] == '\n') lin[strlen(lin)-1] = '\0';
      // Skip comments or empty lines
      if (lin[0] == '#' || strlen(lin) <= 0) continue;
      // Good line candidate: parse it
      XrdOucString gl(lin), group, value;
      // It must contain a '='
      int from = 0;
      if ((from = gl.tokenize(group, 0, '=')) == -1)
         continue;
      // Get linked to the group, if any
      XrdProofGroup *g = fGroups.Find(group.c_str());
      if (!g) {
         TRACE(XERR, "found info for unknown group: "<<group<<" - ignoring");
         continue;
      }
      gl.tokenize(value, from, '=');
      if (value.length() <= 0) {
         TRACE(XERR, "value missing: read line is: '"<<gl<<"'");
         continue;
      }
      // Transform it in a usable value 
      if (value.find('.') == STR_NPOS)
         value += '.';
      // Save it
      g->SetPriority((float)strtod(value.c_str(),0));
   }

   // Close the file
   fclose(fin);

   // Done
   return 0;
}

//__________________________________________________________________________
static int GetGroupsInfo(const char *, XrdProofGroup *g, void *s)
{
   // Fill the global group structure

   XpdGroupGlobal_t *glo = (XpdGroupGlobal_t *)s;

   if (glo) {
      if (g->Active() > 0) {
         // Set the min/max priorities
         if (glo->prmin == -1 || g->Priority() < glo->prmin)
            glo->prmin = g->Priority();
         if (glo->prmax == -1 || g->Priority() > glo->prmax)
            glo->prmax = g->Priority();
         // Set the draft fractions
         if (g->Fraction() > 0) {
            g->SetFracEff((float)(g->Fraction()));
            glo->totfrac += (float)(g->Fraction());
         } else {
            glo->nofrac += 1;
         }
      }
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

//__________________________________________________________________________
static int SetGroupFracEff(const char *, XrdProofGroup *g, void *s)
{
   // Check if user 'u' is memmebr of group 'grp'

   XpdGroupEff_t *eff = (XpdGroupEff_t *)s;

   if (eff && eff->glo) {
      XpdGroupGlobal_t *glo = eff->glo;
      if (g->Active() > 0) {
         if (eff->opt == 0) {
            float ef = g->Priority() / glo->prmin;
            g->SetFracEff(ef);
         } else if (eff->opt == 1) {
            if (g->Fraction() < 0) {
               float ef = ((100. - glo->totfrac) / glo->nofrac);
               g->SetFracEff(ef);
            }
         } else if (eff->opt == 2) {
            if (g->FracEff() < 0) {
               // Share eff->cut (default 5%) between those with undefined fraction
               float ef = (eff->cut / glo->nofrac);
               g->SetFracEff(ef);
            } else {
               // renormalize
               float ef = g->FracEff() * eff->norm;
               g->SetFracEff(ef);
            }
         }
      }
   } else {
      // Not enough info: stop
      return 1;
   }

   // Check next
   return 0;
}

//______________________________________________________________________________
int XrdProofGroupMgr::SetEffectiveFractions(bool opri)
{
   // Go through the list of active groups (those having at least a non-idle
   // member) and determine the effective resource fraction on the base of
   // the scheduling option and of priorities or nominal fractions.
   // Return 0 in case of success, -1 in case of error, 1 if every group
   // has the same priority so that the system scheduler should do the job.

   // Loop over groupd
   XpdGroupGlobal_t glo = {-1., -1., 0, 0.};
   Apply(GetGroupsInfo, &glo);

   XpdGroupEff_t eff = {0, &glo, 0.5, 1.};
   if (opri) {
      // Set effective fractions
      ResetIter();
      eff.opt = 0;
      Apply(SetGroupFracEff, &eff);

   } else {
      // In the fraction scheme we need to fill up with the remaining resources
      // if at least one lower bound was found. And of course we need to restore
      // unitarity, if it was broken

      if (glo.totfrac < 100. && glo.nofrac > 0) {
         eff.opt = 1;
         Apply(SetGroupFracEff, &eff);
      } else if (glo.totfrac > 100) {
         // Leave 5% for unnamed or low priority groups
         eff.opt = 2;
         eff.norm = (glo.nofrac > 0) ? (100. - eff.cut)/glo.totfrac : 100./glo.totfrac ;
         Apply(SetGroupFracEff, &eff);
      }
   }

   // Done
   return 0;
}
