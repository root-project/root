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

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientConn.hh"
#include "XrdClient/XrdClientConnMgr.hh"
#include <string>
#include <algorithm>
#include <ctype.h>

XrdClientEnv *XrdClientEnv::fgInstance = 0;

XrdClientEnv *XrdClientEnv::Instance() {
   // Create unique instance

   if (!fgInstance) {
      fgInstance = new XrdClientEnv;
      if (!fgInstance) {
	 std::cerr << "XrdClientEnv::Instance: fatal - couldn't create XrdClientEnv" << std::endl;
         abort();
      }
   }
   return fgInstance;
}

//_____________________________________________________________________________
XrdClientEnv::XrdClientEnv() {
   // Constructor
   fOucEnv   = new XrdOucEnv();
   fShellEnv = new XrdOucEnv();

   PutInt(NAME_CONNECTTIMEOUT, DFLT_CONNECTTIMEOUT);
   PutInt(NAME_REQUESTTIMEOUT, DFLT_REQUESTTIMEOUT);
   PutInt(NAME_MAXREDIRECTCOUNT, DFLT_MAXREDIRECTCOUNT);
   PutInt(NAME_DEBUG, DFLT_DEBUG);
   PutInt(NAME_RECONNECTWAIT, DFLT_RECONNECTWAIT);
   PutInt(NAME_REDIRCNTTIMEOUT, DFLT_REDIRCNTTIMEOUT);
   PutInt(NAME_FIRSTCONNECTMAXCNT, DFLT_FIRSTCONNECTMAXCNT);
   PutInt(NAME_READCACHESIZE, DFLT_READCACHESIZE);
   PutInt(NAME_READCACHEBLKREMPOLICY, DFLT_READCACHEBLKREMPOLICY);
   PutInt(NAME_READAHEADSIZE, DFLT_READAHEADSIZE);
   PutInt(NAME_MULTISTREAMCNT, DFLT_MULTISTREAMCNT);
   PutInt(NAME_DFLTTCPWINDOWSIZE, DFLT_DFLTTCPWINDOWSIZE);
   PutInt(NAME_DATASERVERCONN_TTL, DFLT_DATASERVERCONN_TTL);
   PutInt(NAME_LBSERVERCONN_TTL, DFLT_LBSERVERCONN_TTL);
   PutInt(NAME_PURGEWRITTENBLOCKS, DFLT_PURGEWRITTENBLOCKS);
   PutInt(NAME_READAHEADSTRATEGY, DFLT_READAHEADSTRATEGY);
   PutInt(NAME_READTRIMBLKSZ, DFLT_READTRIMBLKSZ);
   PutInt(NAME_TRANSACTIONTIMEOUT, DFLT_TRANSACTIONTIMEOUT);
   PutInt(NAME_REMUSEDCACHEBLKS, DFLT_REMUSEDCACHEBLKS);
   PutInt(NAME_ENABLE_FORK_HANDLERS, DFLT_ENABLE_FORK_HANDLERS);
   PutInt(NAME_ENABLE_TCP_KEEPALIVE, DFLT_ENABLE_TCP_KEEPALIVE);
   PutInt(NAME_TCP_KEEPALIVE_TIME,     DFLT_TCP_KEEPALIVE_TIME);
   PutInt(NAME_TCP_KEEPALIVE_INTERVAL, DFLT_TCP_KEEPALIVE_INTERVAL);
   PutInt(NAME_TCP_KEEPALIVE_PROBES,   DFLT_TCP_KEEPALIVE_PROBES);
   PutInt(NAME_XRDCP_SIZE_HINT,        DFLT_XRDCP_SIZE_HINT);

   ImportInt( NAME_CONNECTTIMEOUT );
   ImportInt( NAME_REQUESTTIMEOUT );
   ImportInt( NAME_MAXREDIRECTCOUNT );
   ImportInt( NAME_DEBUG );
   ImportInt( NAME_RECONNECTWAIT );
   ImportInt( NAME_REDIRCNTTIMEOUT );
   ImportInt( NAME_FIRSTCONNECTMAXCNT );
   ImportInt( NAME_READCACHESIZE );
   ImportInt( NAME_READCACHEBLKREMPOLICY );
   ImportInt( NAME_READAHEADSIZE );
   ImportInt( NAME_MULTISTREAMCNT );
   ImportInt( NAME_DFLTTCPWINDOWSIZE );
   ImportInt( NAME_DATASERVERCONN_TTL );
   ImportInt( NAME_LBSERVERCONN_TTL );
   ImportInt( NAME_PURGEWRITTENBLOCKS );
   ImportInt( NAME_READAHEADSTRATEGY );
   ImportInt( NAME_READTRIMBLKSZ );
   ImportInt( NAME_TRANSACTIONTIMEOUT );
   ImportInt( NAME_REMUSEDCACHEBLKS );
   ImportInt( NAME_ENABLE_FORK_HANDLERS );
   ImportInt( NAME_ENABLE_TCP_KEEPALIVE );
   ImportInt( NAME_TCP_KEEPALIVE_TIME );
   ImportInt( NAME_TCP_KEEPALIVE_INTERVAL );
   ImportInt( NAME_TCP_KEEPALIVE_PROBES );
   ImportInt( NAME_XRDCP_SIZE_HINT );
}

//------------------------------------------------------------------------------
// Import a string variable from the shell environment
//------------------------------------------------------------------------------
bool XrdClientEnv::ImportStr( const char *varname )
{
  std::string name = "XRD_";
  name += varname;
  std::transform( name.begin(), name.end(), name.begin(), ::toupper );

  char *value;
  if( !XrdOucEnv::Import( name.c_str(), value ) )
     return false;

  fShellEnv->Put( varname, value );
  return true;
}

//------------------------------------------------------------------------------
// Import an int variable from the shell environment
//------------------------------------------------------------------------------
bool XrdClientEnv::ImportInt( const char *varname )
{
  std::string name = "XRD_";
  name += varname;
  std::transform( name.begin(), name.end(), name.begin(), ::toupper );

  long value;
  if( !XrdOucEnv::Import( name.c_str(), value ) )
     return false;

  fShellEnv->PutInt( varname, value );
  return true;
}

//------------------------------------------------------------------------------
// Get a string from the shell environment
//------------------------------------------------------------------------------
const char *XrdClientEnv::ShellGet(const char *varname)
{
  XrdSysMutexHelper m( fMutex );
  const char *res = fShellEnv->Get( varname );
  if( res )
    return res;

  res = fOucEnv->Get( varname );
  return res;
}

//------------------------------------------------------------------------------
// Get an integer from the shell environment
//------------------------------------------------------------------------------
long XrdClientEnv::ShellGetInt(const char *varname)
{
  XrdSysMutexHelper m( fMutex );
  const char *res = fShellEnv->Get( varname );

  if( res )
    return fShellEnv->GetInt( varname );

  return fOucEnv->GetInt( varname );
}


//_____________________________________________________________________________
XrdClientEnv::~XrdClientEnv() {
   // Destructor
   delete fOucEnv;
   delete fShellEnv;
   delete fgInstance;

   fgInstance = 0;
}

//------------------------------------------------------------------------------
// The fork handlers need to have C linkage (no symbol name mangling)
//------------------------------------------------------------------------------
extern "C"
{

//------------------------------------------------------------------------------
// To be called prior to forking
//------------------------------------------------------------------------------
static void prepare()
{
  if( EnvGetLong( NAME_ENABLE_FORK_HANDLERS ) && ConnectionManager )
  {
    ConnectionManager->ShutDown();
    SessionIDRepo.Purge();
  }
}

//------------------------------------------------------------------------------
// To be called in the parent just after forking
//------------------------------------------------------------------------------
static void parent()
{
  if( EnvGetLong( NAME_ENABLE_FORK_HANDLERS ) && ConnectionManager )
  {
    ConnectionManager->BootUp();
  }
}

//------------------------------------------------------------------------------
// To be called in the child just after forking
//------------------------------------------------------------------------------
static void child()
{
  if( EnvGetLong( NAME_ENABLE_FORK_HANDLERS ) && ConnectionManager )
  {
    ConnectionManager->BootUp();
  }
}

} // extern "C"

//------------------------------------------------------------------------------
// Install the fork handlers on application startup
//------------------------------------------------------------------------------
namespace
{
  static struct Initializer
  {
    Initializer()
    {
      //------------------------------------------------------------------------
      // Install the fork handlers
      //------------------------------------------------------------------------
#ifndef WIN32
      if( pthread_atfork( prepare, parent, child ) != 0 )
      {
        std::cerr << "Unable to install the fork handlers - safe forking not ";
        std::cerr << "possible" << std::endl;
      }
#endif
    }
  } initializer;
}
