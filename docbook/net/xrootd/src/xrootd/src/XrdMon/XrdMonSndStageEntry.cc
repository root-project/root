/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndStageEntry.cc                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndStageEntry.hh"
#include <sstream>
using std::ostream;
using std::stringstream;

XrdMonSndStageEntry::XrdMonSndStageEntry(string u, 
                                       kXR_int16 pid,
                                       kXR_int16 fd,
                                       string host,
                                       string path,
                                       kXR_int32 bytes,
                                       kXR_int32 secs,
                                       kXR_int32 tod,
                                       kXR_int32 myId)
    : _user(u),
      _pid(pid),
      _fd(fd),
      _host(host),
      _path(path),
      _bytes(bytes),
      _secs(secs),
      _tod(tod),
      _myId(myId)
{}

XrdMonSndStageEntry::CompactEntry
XrdMonSndStageEntry::code()
{
    stringstream ss(stringstream::out);
    ss << _user << '.' << _pid << ':' << _fd << '@' << _host
       << '\n' << _path << "\n&sz=" << _bytes << "&tm=" << _secs
       << "&tod=" << _tod;
    //    std::cout << "coded this: \n---\n" << ss.str() << "\n---\n" << std::endl;
    
    CompactEntry ce;
    ce.id     = _myId;
    ce.others = ss.str();
    return ce;
}

ostream& 
operator<<(ostream& o, const XrdMonSndStageEntry& m)
{
    o << m._user << " " << m._pid  << " " << m._fd << " "
      << m._host << " " << m._path << " " << m._bytes << " "
      << m._secs << " " << m._tod  << " " << m._myId;
    return o;
}

