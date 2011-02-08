/*
   Andrew McNab and Shiv Kaushal, University of Manchester.
   Copyright (c) 2002-3. All rights reserved.

   Redistribution and use in source and binary forms, with or
   without modification, are permitted provided that the following
   conditions are met:

     o Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.
     o Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
   BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
/*------------------------------------------------------------------------*
 * For more information about GridSite: http://www.gridpp.ac.uk/gridsite/ *
 *------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <ctype.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fnmatch.h>

#include <libxml/xmlmemory.h>
#include <libxml/tree.h>
#include <libxml/parser.h>

#include "gridsite.h"

//#define XACML_DEBUG

#ifdef XACML_DEBUG
  #define XACML_DEBUG_FILE "/tmp/grstxacmldebug.out"
#endif


/*                                                                      *
 * Global variables, shared by all GACL functions by private to libgacl *
 *                                                                      */

extern char     *grst_perm_syms[];
extern GRSTgaclPerm grst_perm_vals[];


FILE* debugfile;

GRSTgaclAcl *GRSTgaclAclParse(xmlDocPtr, xmlNodePtr, GRSTgaclAcl *);
GRSTgaclAcl *GRSTxacmlAclParse(xmlDocPtr, xmlNodePtr, GRSTgaclAcl *);
int          GRSTxacmlPermPrint(GRSTgaclPerm perm, FILE *fp);

/*                                                     *
 * Functions to read in XACML 1.1 compliant format ACL *
 * Functions based on method for opening GACL format   *
 *                                                     */

// need to check these for libxml memory leaks? - what needs to be freed?


static GRSTgaclCred *GRSTxacmlCredParse(xmlNodePtr cur)
/*
    GRSTxacmlCredParse - parse a credential stored in the libxml structure cur,
    returning it as a pointer or NULL on error.
*/
{
  xmlNodePtr  attr_val;
  xmlNodePtr  attr_des;
  GRSTgaclCred   *cred;

  // cur points to <Subject> or <AnySubjects/>, loop done outside this function.

  if ( (xmlStrcmp(cur->name, (const xmlChar *) "AnySubject") == 0)) cred = GRSTgaclCredNew("any-user");

  else{

  attr_val=cur->xmlChildrenNode->xmlChildrenNode;
  attr_des=attr_val->next;

  cred = GRSTgaclCredNew((char *) xmlNodeGetContent(attr_des->properties->children));

  cred->next      = NULL;

  //Assumed that there is only one name/value pair per credential
  GRSTgaclCredAddValue(cred, (char *) xmlNodeGetContent(attr_des->properties->next->children),
                             (char *) xmlNodeGetContent(attr_val));
  }

  return cred;
}

static GRSTgaclEntry *GRSTxacmlEntryParse(xmlNodePtr cur)
/*
    GRSTxacmlEntryParse - parse an entry stored in the libxml structure cur,
    returning it as a pointer or NULL on error. Also checks to see if the following
    <Rule> tag refers to the same <Target> by checking the <RuleId> of both
*/
{
  int        i, check=0;
  xmlNodePtr cur2;
  xmlNodePtr rule_root=cur;
  GRSTgaclEntry *entry;
  GRSTgaclCred  *cred;


  // Next line not needed as function only called if <Rule> tag found
  // if (xmlStrcmp(cur->name, (const xmlChar *) "Rule") != 0) return NULL;
  // cur and rule_root point to the <Rule> tag

  cur = cur->xmlChildrenNode->xmlChildrenNode;
  // cur should now be pointing at <Subjects> tag
#ifdef XACML_DEBUG
  fprintf (debugfile, "Starting to Parse Entry\n");
#endif
  entry = GRSTgaclEntryNew();

  while (cur!=NULL){

    if (xmlStrcmp(cur->name, (const xmlChar *) "Subjects") == 0){
#ifdef XACML_DEBUG
      fprintf (debugfile, "Starting to Parse Credentials\n");
#endif
      if (check==0){
        // cur still pointing at <Subjects> tag make cur2 point to <Subject> and loop over them.
	cur2=cur->xmlChildrenNode;
	while (cur2!=NULL){
          if ( ((cred = GRSTxacmlCredParse(cur2)) != NULL) && (!GRSTgaclEntryAddCred(entry, cred))){
            GRSTgaclCredFree(cred);
            GRSTgaclEntryFree(entry);
            return NULL;
	  }
	  cur2=cur2->next;
        }
      }
    }

    else if (xmlStrcmp(cur->name, (const xmlChar *) "Actions") == 0){
#ifdef XACML_DEBUG
      fprintf (debugfile, "Starting to Parse Permissions\n");
#endif
      if (xmlStrcmp(xmlNodeGetContent(rule_root->properties->next->children), (const xmlChar *) "Permit") == 0 ){
#ifdef XACML_DEBUG
	fprintf (debugfile, "\tPermit-ed actions: ");
#endif
        for (cur2 = cur->xmlChildrenNode; cur2 != NULL; cur2=cur2->next) //cur2-><Action>
          for (i=0; grst_perm_syms[i] != NULL; ++i)
            if (xmlStrcmp(xmlNodeGetContent(cur2->xmlChildrenNode->xmlChildrenNode), (const xmlChar *) grst_perm_syms[i]) == 0)
            {
#ifdef XACML_DEBUG
              fprintf (debugfile, "%s ", grst_perm_syms[i]);
#endif
	      GRSTgaclEntryAllowPerm(entry, grst_perm_vals[i]);
	    }
      }

      if (xmlStrcmp(xmlNodeGetContent(rule_root->properties->next->children), (const xmlChar *) "Deny") == 0 ) {
#ifdef XACML_DEBUG
	fprintf (debugfile, "\tDeny-ed actions: ");
#endif
        for (cur2 = cur->xmlChildrenNode; cur2 != NULL; cur2=cur2->next) //cur2-><Action>
          for (i=0; grst_perm_syms[i] != NULL; ++i)
            if (xmlStrcmp(xmlNodeGetContent(cur2->xmlChildrenNode->xmlChildrenNode), (const xmlChar *) grst_perm_syms[i]) == 0)
            {
              
#ifdef XACML_DEBUG
	      fprintf (debugfile, "%s ", grst_perm_syms[i]);
#endif
	      GRSTgaclEntryDenyPerm(entry, grst_perm_vals[i]);
	    }
      }

    }
    else{ // I cannot parse this - give up rather than get it wrong
#ifdef XACML_DEBUG
      fprintf (debugfile, "OOOPSIE\n");
#endif
      GRSTgaclEntryFree(entry);
      return NULL;
    }

    cur=cur->next;

    // Check if next Rule should be included when end of current rule reached
    // If RuleId are from the same entry (eg Entry1A and Entry1D)
    // make cur point to the next Rule's <Subjects> tag
    if (cur==NULL)
      if (check==0)
        if (rule_root->next!=NULL)
	  if ( strncmp((const char*)xmlNodeGetContent(rule_root->properties->children), // RuleId of this Rule
		       (const char*)xmlNodeGetContent(rule_root->next->properties->children), // RuleId of next Rule
		 6) == 0){
#ifdef XACML_DEBUG
	    fprintf (debugfile, "End of perms and creds, next is %s \n", xmlNodeGetContent(rule_root->next->properties->children));
#endif
	    rule_root=rule_root->next;
	    cur=rule_root->xmlChildrenNode->xmlChildrenNode;
#ifdef XACML_DEBUG
	    fprintf (debugfile, "skipped to <%s> tag of next Rule\n", cur->name);
#endif
	    check++;
	  }
  }

  return entry;
}

GRSTgaclAcl *GRSTxacmlAclLoadFile(char *filename)
{
  xmlDocPtr   doc;
  xmlNodePtr  cur;
  GRSTgaclAcl    *acl=NULL;

  doc = xmlParseFile(filename);
  if (doc == NULL) return NULL;

  cur = xmlDocGetRootElement(doc);
  if (cur == NULL) return NULL;

  if (!xmlStrcmp(cur->name, (const xmlChar *) "Policy")) { acl=GRSTxacmlAclParse(doc, cur, acl);}
  else if (!xmlStrcmp(cur->name, (const xmlChar *) "gacl")) {acl=GRSTgaclAclParse(doc, cur, acl);}
  else /* ACL format not recognised */
    {
      xmlFreeDoc(doc);
      free(cur);
      return NULL;
    }

  xmlFreeDoc(doc);
  return acl;
}

GRSTgaclAcl *GRSTxacmlAclParse(xmlDocPtr doc, xmlNodePtr cur, GRSTgaclAcl *acl)
{
  GRSTgaclEntry  *entry;

  #ifdef XACML_DEBUG
  debugfile=fopen(XACML_DEBUG_FILE, "w");
  fprintf (debugfile, "ACL loaded..\n");
  fprintf (debugfile, "Parsing XACML\n");
  #endif

  // Have an XACML policy file.
  // Skip <Target> tag and set cur to first <Rule> tag
  cur = cur->xmlChildrenNode->next;

  acl = GRSTgaclAclNew();

  while (cur != NULL){

    if ( !xmlStrcmp(cur->name, (const xmlChar *)"Rule") )
    { // IF statement not needed?
      #ifdef XACML_DEBUG
      fprintf (debugfile, "Rule %s found\n", xmlNodeGetContent(cur->properties->children) );
      fprintf (debugfile, "Parsing Entry for this rule\n");
      #endif
      entry = GRSTxacmlEntryParse(cur);

      if (entry == NULL)
      {
        GRSTgaclAclFree(acl);
        return NULL;
      }
      else GRSTgaclAclAddEntry(acl, entry);

      #ifdef XACML_DEBUG
      fprintf (debugfile, "Entry read in\n\n");
      #endif
    }

    // If the current and next Rules are part of the same entry then advance two Rules
    // If not then advance 1
    if (cur->next != NULL)
    {
      if ( strncmp((const char*)xmlNodeGetContent(cur->properties->children),       // RuleId of this Rule
                   (const char*)xmlNodeGetContent(cur->next->properties->children), // RuleId of next Rule
                   6) == 0)
      {
        #ifdef XACML_DEBUG
	fprintf (debugfile, "skipping next rule %s, should have been caught previously\n\n", xmlNodeGetContent(cur->next->properties->children) );
	#endif
	cur=cur->next;
      } // Check first 6 characters i.e. Entry1**/
    }

    cur=cur->next;

  }

  #ifdef XACML_DEBUG
  fprintf (debugfile, "Finished loading ACL - Fanfare!\n");
  fclose(debugfile);
  #endif

  return acl;
}


int GRSTxacmlFileIsAcl(char *pathandfile)
/* Return 1 if filename in *pathandfile starts GRST_ACL_FILE
   Return 0 otherwise. */
{
  char *filename;

  filename = rindex(pathandfile, '/');
  if (filename == NULL) filename = pathandfile;
  else                  filename++;

  return (strncmp((const char*)filename, GRST_ACL_FILE, sizeof(GRST_ACL_FILE) - 1) == 0);
}

char *GRSTxacmlFileFindAclname(char *pathandfile)
/* Return malloc()ed ACL filename that governs the given file or directory
   (for directories, the ACL file is in the directory itself), or NULL if none
   can be found. */
{
  char        *path, *p;
  struct stat  statbuf;

  path = malloc(strlen(pathandfile) + sizeof(GRST_ACL_FILE) + 1);
  strcpy(path, pathandfile);

  if (stat(path, &statbuf) == 0)
    {
      if (!S_ISDIR(statbuf.st_mode)) /* can strip this / off straightaway */
        {
          p = rindex(path, '/');
          if (p != NULL) *p = '\0';
        }
    }

  while (path[0] != '\0')
       {
         strcat(path, "/");
         strcat(path, GRST_ACL_FILE);

         if (stat(path, &statbuf) == 0) return path;

         p = rindex(path, '/');
         *p = '\0';     /* strip off the / we added for ACL */

         p = rindex(path, '/');
         if (p == NULL) break; /* must start without / and we there now ??? */

         *p = '\0';     /* strip off another layer of / */                 
       }
       
  free(path);
  return NULL;
}

GRSTgaclAcl *GRSTxacmlAclLoadforFile(char *pathandfile)
/* Return ACL that governs the given file or directory (for directories,
   the ACL file is in the directory itself.) */
{
  char        *path;
  GRSTgaclAcl     *acl;

  path = GRSTxacmlFileFindAclname(pathandfile);
  
  if (path != NULL)
    {
      acl = GRSTxacmlAclLoadFile(path);
      free(path);
      return acl;
    }

  return NULL;
}



/*                                                     *
 * Functions to save ACL in XACML 1.1 compliant format *
 * Functions based on method for saving to GACL format *
 *                                                     */


int GRSTxacmlCredPrint(GRSTgaclCred *cred, FILE *fp)
/*
   GRSTxacmlCredPrint - print a credential and any name-value pairs is contains in XACML form
*/
{
  char *q;

  if (cred->auri != NULL)
    {
	   fputs("\t\t\t\t<Subject>\n", fp);
	   fputs("\t\t\t\t\t<SubjectMatch MatchId=\"urn:oasis:names:tc:xacml:1.0:function:string-equal\">\n", fp);
	   fputs("\t\t\t\t\t\t<AttributeValue DataType=\"http://www.w3.org/2001/XMLSchema#string\">", fp);
           for (q=cred->auri; *q != '\0'; ++q)
              if      (*q == '<')  fputs("&lt;",   fp);
              else if (*q == '>')  fputs("&gt;",   fp);
              else if (*q == '&')  fputs("&amp;" , fp);
              else if (*q == '\'') fputs("&apos;", fp);
              else if (*q == '"')  fputs("&quot;", fp);
              else                 fputc(*q, fp);


	   fputs("</AttributeValue>\n", fp);

	   fputs("\t\t\t\t\t\t<SubjectAttributeDesignator\n", fp);
	   fputs("\t\t\t\t\t\t\tAttributeId=", fp);
	   fprintf(fp, "\"cred\"\n");
	   fputs("\t\t\t\t\t\t\tDataType=", fp);
           fprintf(fp, "\"auri\"/>\n");
	   fputs("\t\t\t\t\t</SubjectMatch>\n", fp);
	   fputs("\t\t\t\t</Subject>\n", fp);
    }
    else fputs("\t\t\t\t<AnySubject/>\n", fp);

  return 1;
}


int GRSTxacmlEntryPrint(GRSTgaclEntry *entry, FILE *fp, int rule_number)
{
  GRSTgaclCred  *cred;
  GRSTgaclPerm  i;

  if (entry->allowed){

  fprintf(fp, "\t<Rule RuleId=\"Entry%dA\" Effect=\"Permit\">\n", rule_number);
  fputs("\t\t<Target>\n", fp);
  fputs("\t\t\t<Subjects>\n", fp);

  for (cred = entry->firstcred; cred != NULL; cred = cred->next)
                                            GRSTxacmlCredPrint(cred, fp);

  fputs("\t\t\t</Subjects>\n", fp);
  fputs("\t\t\t<Actions>\n", fp);

      for (i=GRST_PERM_READ; i <= GRST_PERM_ADMIN; ++i)
       if ((entry->allowed) & i) GRSTxacmlPermPrint(i, fp);

  fputs("\t\t\t</Actions>\n", fp);
  fputs("\t\t</Target>\n", fp);
  fputs("\t</Rule>\n", fp);
  }

  if (entry->denied){

  fprintf(fp, "\t<Rule RuleId=\"Entry%dD\" Effect=\"Deny\">\n", rule_number);
  fputs("\t\t<Target>\n", fp);
  fputs("\t\t\t<Subjects>\n", fp);

  for (cred = entry->firstcred; cred != NULL; cred = cred->next)
                                            GRSTxacmlCredPrint(cred, fp);

  fputs("\t\t\t</Subjects>\n", fp);
  fputs("\t\t\t<Actions>\n", fp);

      for (i=GRST_PERM_READ; i <= GRST_PERM_ADMIN; ++i)
       if (entry->denied & i) GRSTxacmlPermPrint(i, fp);

  fputs("\t\t\t</Actions>\n", fp);
  fputs("\t\t</Target>\n", fp);
  fputs("\t</Rule>\n", fp);
  }
  return 1;
}


int GRSTxacmlPermPrint(GRSTgaclPerm perm, FILE *fp)
{
  GRSTgaclPerm i;

  for (i=GRST_PERM_READ; grst_perm_syms[i] != NULL; ++i)
       if (perm == grst_perm_vals[i])
         {

	   fputs("\t\t\t\t<Action>\n", fp);
	   fputs("\t\t\t\t\t<ActionMatch MatchId=\"urn:oasis:names:tc:xacml:1.0:function:string-equal\">\n", fp);
	   fputs("\t\t\t\t\t\t<AttributeValue DataType=\"http://www.w3.org/2001/XMLSchema#string\">", fp);
	   fprintf(fp, "%s", grst_perm_syms[i]);
	   fputs("</AttributeValue>\n", fp);
	   fputs("\t\t\t\t\t\t<ActionAttributeDesignator\n", fp);
	   fputs("\t\t\t\t\t\t\tAttributeId=\"urn:oasis:names:tc:xacml:1.0:action:action-id\"\n", fp);
	   fputs("\t\t\t\t\t\t\tDataType=\"http://www.w3.org/2001/XMLSchema#string\"/>\n", fp);
	   fputs("\t\t\t\t\t</ActionMatch>\n", fp);
	   fputs("\t\t\t\t</Action>\n",fp);

           return 1;
         }

  return 0;
}

int GRSTxacmlAclPrint(GRSTgaclAcl *acl, FILE *fp, char* dir_uri)
{
  GRSTgaclEntry *entry;
  int rule_number=1;

  fputs("<Policy", fp);
  fputs("\txmlns=\"urn:oasis:names:tc:xacml:1.0:policy\"\n", fp);
  fputs("\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n", fp);
  fputs("\txsi:schemaLocation=\"urn:oasis:names:tc:xacml:1.0:policy cs-xacml-schema-policy-01.xsd\"\n", fp);
  fputs("\tPolicyId=\"GridSitePolicy\"\n", fp);
  fputs("\tRuleCombiningAlgId=\"urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:deny-overrides\">\n\n", fp);

  fputs("\t<Target>\n\t\t<Resources>\n\t\t\t<Resource>\n", fp);
  fputs("\t\t\t\t<ResourceMatch MatchId=\"urn:oasis:names:tc:xacml:1.0:function:string-equal\">\n", fp);
  fputs("\t\t\t\t\t<AttributeValue DataType=\"http://www.w3.org/2001/XMLSchema#string\">", fp);
  fprintf(fp, "%s", dir_uri);
  fputs("</AttributeValue>\n", fp);
  fputs("\t\t\t\t\t<ResourceAttributeDesignator\n", fp);
  fputs("\t\t\t\t\t\tAttributeId=\"urn:oasis:names:tc:xacml:1.0:resource:resource-id\"\n", fp);
  fputs("\t\t\t\t\t\tDataType=\"http://www.w3.org/2001/XMLSchema#string\"/>\n", fp);

  fputs("\t\t\t\t</ResourceMatch>\n\t\t\t</Resource>\n\t\t</Resources>\n\t\t<Subjects>\n\t\t\t<AnySubject/>\n\t\t</Subjects>", fp);
  fputs("\n\t\t<Actions>\n\t\t\t<AnyAction/>\n\t\t</Actions>\n\t</Target>\n\n", fp);

  for (entry = acl->firstentry; entry != NULL; entry = entry->next){

	GRSTxacmlEntryPrint(entry, fp, rule_number);
	rule_number++;
  }

  fputs("</Policy>\n", fp);

  return 1;
}

int GRSTxacmlAclSave(GRSTgaclAcl *acl, char *filename, char* dir_uri)
{
  int   ret;
  FILE *fp;

  fp = fopen(filename, "w");
  if (fp == NULL) return 0;

  fprintf(fp,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

  ret = GRSTxacmlAclPrint(acl, fp, dir_uri);

  fclose(fp);

  return ret;
}




