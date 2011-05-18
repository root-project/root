/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndTraceCache.hh                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSNDTRACECACHE_HH
#define XRDMONSNDTRACECACHE_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonSndDebug.hh"
#include "XrdMon/XrdMonSndTraceEntry.hh"
#include <vector>
using std::vector;

// The class responsible for caching data before it is sent

class XrdMonSndTraceCache {

public:
    static const unsigned int PACKETSIZE;
    static const unsigned int NODATAELEMS;

    XrdMonSndTraceCache();

    bool bufferFull() const {
        return _entries.size() >= NODATAELEMS-3; // save 3 spots for time entries
    }

    int add(const XrdMonSndTraceEntry& de);
    const vector<XrdMonSndTraceEntry>& getVector() { return _entries; }
    void clear() { _entries.clear(); }
    
private:
    vector<XrdMonSndTraceEntry> _entries;
};

#endif /* XRDMONSNDTRACECACHE_HH */
