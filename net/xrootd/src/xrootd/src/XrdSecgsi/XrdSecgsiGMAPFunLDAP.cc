// $Id$

const char *XrdSecgsiGMAPFunLDAPCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*             X r d S e c g s i G M A P F u n L D A P . c c                  */
/*                                                                            */
/* (c) 2008, G. Ganis / CERN                                                  */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* GMAP function implementation querying a LDAP database                      */
/*                                                                            */
/* ************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static char *ldapsrv    = 0;
static char *searchbase = 0;
static char *attribute  = 0;

int XrdSecgsiGMAPInit(const char *cfg);

//
// Main function
//
extern "C"
{
char *XrdSecgsiGMAPFun(const char *dn, int now)
{
   // Implementation of XrdSecgsiGMAPFun querying an LDAP server
   // for the distinguished name 'dn'; the unused argument is the time at
   // which the function is called.

   // Init the relevant fields (only once)
   if (now <= 0) {
      if (XrdSecgsiGMAPInit(dn) != 0)
         return (char *)-1;
      return (char *)0;
   }

   // Output
   char *name = 0;

   // Prepare the command to be executed
   char cmd[4096];
   sprintf(cmd, "ldapsearch -H %s -x -b \"%s\" \"subject=%s\" %s",
                 ldapsrv, searchbase, dn, attribute);

   // Execute the command into a pipe
   FILE *fp = popen(cmd, "r");
   if (fp) {
      char line[1024], att[40], uname[256];
      sprintf(att, "%s: ", attribute);
      while (fgets(line, sizeof(line), fp)) {
         // Look for a line starting with "uid: "
         if (!strncmp(line, att, strlen(att))) {
            sscanf(line, "%s %s", att, uname);
            name = strdup(uname);
            break;
         }
      }
      pclose(fp);
   }

   // Done
   return name;
}}

//
// Init the relevant parameters from a dedicated config file
//
int XrdSecgsiGMAPInit(const char *cfg)
{
   // Initialize the relevant parameters from the file 'cfg' or
   // from the one defined by XRDGSIGMAPLDAPCF.
   // Return 0 on success, -1 otherwise

   if (!cfg) cfg = getenv("XRDGSIGMAPLDAPCF");
   if (!cfg || strlen(cfg) <= 0) {
      fprintf(stderr, " +++ XrdSecgsiGMAPInit (LDAP): error: undefined config file path +++\n");
      return -1;
   }

   FILE *fcf = fopen(cfg, "r");
   if (fcf) {
      char l[4096], k[20], val[4096];
      while (fgets(l, sizeof(l), fcf)) {
         int len = strlen(l);
         if (len < 2) continue;
         if (l[0] == '#') continue;
         if (l[len-1] == '\n') l[len-1] = '\0';
         sscanf(l, "%s %s", k, val);
         if (!strcmp(k, "srv:")) {
            ldapsrv = strdup(val);
         } else if (!strcmp(k, "base:")) {
            searchbase = strdup(val);
         } else if (!strcmp(k, "attr:")) {
            attribute = strdup(val);
         } else {
            fprintf(stderr, "XrdSecgsiGMAPInit (LDAP): warning: unknown key: '%s' - ignoring\n", k);
         }
      }
      fclose(fcf);
   } else {
      fprintf(stderr, " +++ XrdSecgsiGMAPInit (LDAP): error: config file '%s'"
                      " could not be open (errno: %d) +++\n", cfg, errno);
      return -1;
   }
   // Done
   return 0;
}
