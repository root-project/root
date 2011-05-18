//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientEnv                                                         // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Singleton used to handle the default parameter values                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef XRD_CENV_H
#define XRD_CENV_H

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysPthread.hh"

#include <string.h>

using namespace std;


#define EnvGetLong(x) XrdClientEnv::Instance()->ShellGetInt(x)
#define EnvGetString(x) XrdClientEnv::Instance()->ShellGet(x)
#define EnvPutString(name, val) XrdClientEnv::Instance()->Put(name, val)
#define EnvPutInt(name, val) XrdClientEnv::Instance()->PutInt(name, val)

class XrdClientEnv {
 private:

   XrdOucEnv           *fOucEnv;
   XrdSysRecMutex       fMutex;
   static XrdClientEnv *fgInstance;
   XrdOucEnv           *fShellEnv;

 protected:
   XrdClientEnv();
   ~XrdClientEnv();

   //---------------------------------------------------------------------------
   //! Import the variables from the shell environment, the variable names
   //! are capitalized and prefixed with "XRD_"
   //---------------------------------------------------------------------------
   bool ImportStr( const char *varname );
   bool ImportInt( const char *varname );

 public:

   const char *          Get(const char *varname) {
      const char *res;
      XrdSysMutexHelper m(fMutex);

      res = fOucEnv->Get(varname);
      return res;
   }

   long                   GetInt(const char *varname) {
      long res;
      XrdSysMutexHelper m(fMutex);

      res = fOucEnv->GetInt(varname);
      return res;
   }

   //---------------------------------------------------------------------------
   //! Get a string variable from the environment, the same as Get, but
   //! checks the shell environment first
   //---------------------------------------------------------------------------
   const char *ShellGet( const char *varname );

   //---------------------------------------------------------------------------
   //! Get an integet variable from the environment, the same as GetInt, but
   //! checks the shell environment first
   //---------------------------------------------------------------------------
   long        ShellGetInt( const char *varname );


   void                   Put(const char *varname, const char *value) {
      XrdSysMutexHelper m(fMutex);

      fOucEnv->Put(varname, value);
   }

   void  PutInt(const char *varname, long value) {
      XrdSysMutexHelper m(fMutex);

      fOucEnv->PutInt(varname, value);
   }

   static XrdClientEnv    *Instance();

};

#endif
