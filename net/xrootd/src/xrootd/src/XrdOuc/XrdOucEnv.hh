#ifndef __OUC_ENV__
#define __OUC_ENV__
/******************************************************************************/
/*                                                                            */
/*                          X r d O u c E n v . h h                           */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>
#ifndef WIN32
#include <strings.h>
#endif
#include "XrdOuc/XrdOucHash.hh"

class XrdSecEntity;

class XrdOucEnv
{
public:

// Env() returns the full environment string and length passed to the
//       constructor.
//
inline char *Env(int &envlen) {envlen = global_len; return global_env;}

// Export() sets an external environmental variable to the desired value
//          using dynamically allocated fixed storage.
//
static int   Export(const char *Var, const char *Val);
static int   Export(const char *Var, int         Val);

// Get() returns the address of the string associated with the variable
//       name. If no association exists, zero is returned.
//
       char *Get(const char *varname) {return env_Hash.Find(varname);}

// GetInt() returns a long integer value. If the variable varname is not found
//           in the hash table, return -999999999.       
//
       long  GetInt(const char *varname);

// Put() associates a string value with the a variable name. If one already
//       exists, it is replaced. The passed value and variable strings are
//       duplicated (value here, variable by env_Hash).
//
       void  Put(const char *varname, const char *value)
                {env_Hash.Rep((char *)varname, strdup(value), 0, Hash_dofree);}

// PutInt() puts a long integer value into the hash. Internally, the value gets
//          converted into a char*
//
       void  PutInt(const char *varname, long value);

// Delimit() search for the first occurrence of comma (',') in value and
//           replaces it with a null byte. It then returns the address of the
//           remaining string. If no comma was found, it returns zero.
//
       char *Delimit(char *value);

// secEnv() returns the security environment; which may be a null pointer.
//
inline const XrdSecEntity *secEnv() {return secEntity;}

// Use the constructor to define the initial variable settings. The passed
// string is duplicated and the copy can be retrieved using Env().
//
       XrdOucEnv(const char *vardata=0, int vardlen=0, 
                 const XrdSecEntity *secent=0);

      ~XrdOucEnv() {if (global_env) free((void *)global_env);}

private:

XrdOucHash<char> env_Hash;
const XrdSecEntity *secEntity;
char *global_env;
int   global_len;
};
#endif
