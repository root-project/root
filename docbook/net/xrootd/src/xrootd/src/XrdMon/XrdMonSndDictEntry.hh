/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndDictEntry.hh                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef DICTENTRY_HH
#define DICTENTRY_HH

#include "XrdMon/XrdMonTypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <string>
using std::ostream;
using std::string;

// <user>.<pid>:<fd>@<host>\npath
class XrdMonSndDictEntry {
public:
    struct CompactEntry {
        kXR_int32 id;
        string  others;  // <user>.<pid>:<fd>@<host>\n<path>
        kXR_int16 size() const {return 4 + others.size();}
    };
    
    XrdMonSndDictEntry(string u, 
                       kXR_int16 pid,
                       kXR_int16 fd,
                       string host,
                       string path,
                       kXR_int32 id);

    CompactEntry code();
    
private:
    string  _user;
    kXR_int16 _pid;
    kXR_int16 _fd;
    string  _host;
    string  _path;

    kXR_int32 _myId;

    friend ostream& operator<<(ostream& o, 
                               const XrdMonSndDictEntry& m);
};

#endif /* DICTENTRY_HH */
