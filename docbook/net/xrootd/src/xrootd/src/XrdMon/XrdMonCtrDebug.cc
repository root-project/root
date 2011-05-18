/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrDebug.cc                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCtrDebug.hh"
#include "XrdMon/XrdMonCommon.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <stdlib.h> /* getenv */

#include <fstream>
#include <string>

XrdMonCtrDebug::Verbosity XrdMonCtrDebug::_verbose = XrdMonCtrDebug::Receiving;
XrdSysMutex      XrdMonCtrDebug::_mutex;

void
XrdMonCtrDebug::initialize()
{
    const char* env = getenv("DEBUG");
    if ( 0 == env ) {
        return;
    }
    if ( 0 == strcmp(env, "all") ) {
        _verbose = XrdMonCtrDebug::All;
        return;
    }
    _verbose = (XrdMonCtrDebug::Verbosity) atoi(env);
}
