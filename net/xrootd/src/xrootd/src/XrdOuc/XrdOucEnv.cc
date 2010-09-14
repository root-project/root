/******************************************************************************/
/*                                                                            */
/*                          X r d O u c E n v . c c                           */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOucEnvCVSID = "$Id$";

#include "string.h"
#include "stdio.h"

#include "XrdOuc/XrdOucEnv.hh"
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOucEnv::XrdOucEnv(const char *vardata, int varlen, 
                     const XrdSecEntity *secent)
                    : env_Hash(8,13), secEntity(secent)
{
   char *vdp, varsave, *varname, *varvalu;

   if (!vardata) {global_env = 0; global_len = 0; return;}

// Copy the the global information (don't rely on its being correct)
//
   if (!varlen) varlen = strlen(vardata);
   global_env = (char *)malloc(varlen+2); global_len = varlen;
   if (*vardata == '&') vdp = global_env;
      else {*global_env = '&'; vdp = global_env+1;}
   memcpy((void *)vdp, (const void *)vardata, (size_t)varlen);
   *(vdp+varlen) = '\0';
   vdp = global_env;

// scan through the string looking for '&'
//
   if (vdp) while(*vdp)
        {if (*vdp != '&') {vdp++; continue;}    // &....
         varname = ++vdp;

         while(*vdp && *vdp != '=') vdp++;  // &....=
         if (!*vdp) break;
         *vdp = '\0';
         varvalu = ++vdp;

         while(*vdp && *vdp != '&') vdp++;  // &....=....&
         varsave = *vdp; *vdp = '\0';

         if (*varname && *varvalu)
            env_Hash.Rep(varname, strdup(varvalu), 0, Hash_dofree);

         *vdp = varsave; *(--varvalu) = '=';
        }
   return;
}

/******************************************************************************/
/*                               D e l i m i t                                */
/******************************************************************************/

char *XrdOucEnv::Delimit(char *value)
{
     while(*value) if (*value == ',') {*value = '\0'; return ++value;}
                      else value++;
     return (char *)0;
}
 
/******************************************************************************/
/*                                E x p o r t                                 */
/******************************************************************************/

int XrdOucEnv::Export(const char *Var, const char *Val)
{
   int vLen = strlen(Var);
   char *eBuff;

// If this is a null value then substitute a null string
//
   if (!Val) Val = "";

// Allocate memory. Note that this memory will appear to be lost.
//
   eBuff = (char *)malloc(vLen+strlen(Val)+2); // +2 for '=' and '\0'

// Set up envar
//
   strcpy(eBuff, Var);
   *(eBuff+vLen) = '=';
   strcpy(eBuff+vLen+1, Val);
   return putenv(eBuff);
}

/******************************************************************************/

int XrdOucEnv::Export(const char *Var, int Val)
{
   char buff[32];
   sprintf(buff, "%d", Val);
   return Export(Var, buff);
}

/******************************************************************************/
/*                                G e t I n t                                 */
/******************************************************************************/

long XrdOucEnv::GetInt(const char *varname) 
{
// Retrieve a char* value from the Hash table and convert it into a long.
// Return -999999999 if the varname does not exist
//
  if (env_Hash.Find(varname) == NULL) {
    return -999999999;
  } else {
    return atol(env_Hash.Find(varname));
  }
}


/******************************************************************************/
/*                                P u t I n t                                 */
/******************************************************************************/

void XrdOucEnv::PutInt(const char *varname, long value) 
{
// Convert the long into a char* and the put it into the hash table
//
  char stringValue[24];
  sprintf(stringValue, "%ld", value);
  env_Hash.Rep(varname, strdup(stringValue), 0, Hash_dofree);
}
