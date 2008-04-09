/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndDictEntry.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndDictEntry.hh"
#include <sstream>
using std::ostream;
using std::stringstream;

XrdMonSndDictEntry::XrdMonSndDictEntry(string u, 
                                       kXR_int16 pid,
                                       kXR_int16 fd,
                                       string host,
                                       string path,
                                       kXR_int32 myId)
    : _user(u),
      _pid(pid),
      _fd(fd),
      _host(host),
      _path(path),
      _myId(myId)
{}

XrdMonSndDictEntry::CompactEntry
XrdMonSndDictEntry::code()
{
    stringstream ss(stringstream::out);
    ss << _user << '.' << _pid << ':' << _fd << '@' << _host
       << '\n' << _path;
    CompactEntry ce;
    ce.id     = _myId;
    ce.others = ss.str();
    return ce;
}

ostream& 
operator<<(ostream& o, const XrdMonSndDictEntry& m)
{
    o << m._user << " " << m._pid  << " " << m._fd << " "
      << m._host << " " << m._path << " " << m._myId;
    return o;
}

