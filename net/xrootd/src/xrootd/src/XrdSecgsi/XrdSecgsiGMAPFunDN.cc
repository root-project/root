// $Id$

/******************************************************************************/
/*                                                                            */
/*             X r d S e c g s i G M A P F u n D N . c c                      */
/*                                                                            */
/* (c) 2011, G. Ganis / CERN                                                  */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* GMAP function implementation extracting info from the DN                   */
/*                                                                            */
/* ************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <XrdOuc/XrdOucHash.hh>
#include <XrdOuc/XrdOucString.hh>

enum XrdSecgsi_Match {kFull     = 0,
                      kBegins   = 1,
                      kEnds     = 2,
                      kContains = 4
                     };

class XrdSecgsiMapEntry_t
{
public:
   XrdSecgsiMapEntry_t(const char *v, const char *u, int t) : val(v), user(u), type(t) { }

   XrdOucString  val;
   XrdOucString  user;
   int           type;
};

static XrdOucHash<XrdSecgsiMapEntry_t> gMappings;

//__________________________________________________________________________
static int FindMatchingCondition(const char *, XrdSecgsiMapEntry_t *mc, void *xmp)
{
   // Print content of entry 'ui' and go to next

   XrdSecgsiMapEntry_t *mpe = (XrdSecgsiMapEntry_t *)xmp;

   bool match = 0;
   if (mc && mpe) {
      if (mc->type == kContains) {
         if (mpe->val.find(mc->val) != STR_NPOS) match = 1;
      } else if (mc->type == kBegins) {
         if (mpe->val.beginswith(mc->val)) match = 1;
      } else if (mc->type == kEnds) {
         if (mpe->val.endswith(mc->val)) match = 1;
      } else {
         if (mpe->val.matches(mc->val.c_str())) match = 1;
      }
      if (match) mpe->user = mc->user;
   }

   // We stop if matched, otherwise we continue
   return (match) ? 1 : 0;
}


int XrdSecgsiGMAPInit(const char *cfg);

//
// Main function
//
extern "C"
{
char *XrdSecgsiGMAPFun(const char *dn, int now)
{
   // Implementation of XrdSecgsiGMAPFun extracting the information from the 
   // distinguished name 'dn'

   // Init the relevant fields (only once)
   if (now <= 0) {
      if (XrdSecgsiGMAPInit(dn) != 0)
         return (char *)-1;
      return (char *)0;
   }

   // Output
   char *name = 0;

   XrdSecgsiMapEntry_t *mc = 0;
   // Try the full match first
   if ((mc = gMappings.Find(dn))) {
      // Get the associated user
      name = strdup(mc->val.c_str());
   } else {
      // Else scan the avaulable mappings
      mc = new XrdSecgsiMapEntry_t(dn, "", kFull);
      gMappings.Apply(FindMatchingCondition, (void *)mc);
      if (mc->user.length() > 0) name = strdup(mc->user.c_str());
   }
             fprintf(stderr, " +++ XrdSecgsiGMAPFun (DN): mapping DN '%s' to '%s' \n", dn, name);
  
   // Done
   return name;
}}

//
// Init the relevant parameters from a dedicated config file
//
int XrdSecgsiGMAPInit(const char *cfg)
{
   // Initialize the relevant parameters from the file 'cfg' or
   // from the one defined by XRDGSIGMAPDNCF.
   // Return 0 on success, -1 otherwise

   if (!cfg) cfg = getenv("XRDGSIGMAPDNCF");
   if (!cfg || strlen(cfg) <= 0) {
      fprintf(stderr, " +++ XrdSecgsiGMAPInit (DN): error: undefined config file path +++\n");
      return -1;
   }

   FILE *fcf = fopen(cfg, "r");
   if (fcf) {
      char l[4096], val[4096], usr[256];
      while (fgets(l, sizeof(l), fcf)) {
         int len = strlen(l);
         if (len < 2) continue;
         if (l[0] == '#') continue;
         if (l[len-1] == '\n') l[len-1] = '\0';
         if (sscanf(l, "%4096s %256s", val, usr) >= 2) {
            XrdOucString stype = "matching";
            char *p = &val[0];
            int type = kFull;
            if (val[0] == '^') {
               // Starts-with
               type = kBegins;
               p = &val[1];
               stype = "beginning with";
            } else {
               int vlen = strlen(val);
               if (val[vlen-1] == '$') {
                  // Ends-with
                  type = kEnds;
                  val[vlen-1] = '\0';
                  stype = "ending with";
               } else if (val[vlen-1] == '+') {
                  // Contains
                  type = kContains;
                  val[vlen-1] = '\0';
                  stype = "containing";
               }
            }
            // Register
            gMappings.Add(p, new XrdSecgsiMapEntry_t(p, usr, type));
            //
            fprintf(stderr, " +++ XrdSecgsiGMAPInit (DN): mapping DNs %s '%s' to '%s' \n", stype.c_str(), p, usr);
         }
      }
      fclose(fcf);
   } else {
      fprintf(stderr, " +++ XrdSecgsiGMAPInit (DN): error: config file '%s'"
                      " could not be open (errno: %d) +++\n", cfg, errno);
      return -1;
   }
   // Done
   return 0;
}
