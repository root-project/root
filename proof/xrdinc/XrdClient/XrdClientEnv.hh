#ifndef XRD_CENV_H
#define XRD_CENV_H
/******************************************************************************/
/*                                                                            */
/*                     X r d C l i e n t E n v . h h                          */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/*  Alvise Dorigo, Fabrizio Furano                                            */
/*          INFN Padova, 2003                                                 */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Singleton used to handle the default parameter values                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysPthread.hh"

#include <string.h>

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
   void Lock()
   {
     fMutex.Lock();
   }

   void UnLock()
   {
     fMutex.UnLock();
   }

   int ReInitLock()
   {
     return fMutex.ReInitRecMutex();
   }

  static XrdClientEnv *Instance();
};
#endif
