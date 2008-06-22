/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndTraceCache.cc                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndTraceCache.hh"
#include "XrdMon/XrdMonSndTraceEntry.hh"

#include "XrdSys/XrdSysHeaders.hh"
using std::cerr;
using std::cout;
using std::endl;

const unsigned int XrdMonSndTraceCache::PACKETSIZE  = 8*1024;
const unsigned int XrdMonSndTraceCache::NODATAELEMS = (PACKETSIZE-HDRLEN)/TRACEELEMLEN;

XrdMonSndTraceCache::XrdMonSndTraceCache()
{}

int
XrdMonSndTraceCache::add(const XrdMonSndTraceEntry& de)
{
    if ( _entries.size() > NODATAELEMS ) {
        cerr << "Internal error: buffer too large (" 
             << _entries.size() << " > " << NODATAELEMS 
             << ")." << endl;
        return 1;
    }
    _entries.push_back(de);
    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SCache) ) {
        cout << "Cache:: added " << de << ", size now " 
             << _entries.size() << endl;
    }
    
    return 0;
}

