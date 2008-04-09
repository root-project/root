/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecOnePacket.cc                           */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonDecOnePacket.hh"
#include "XrdMon/XrdMonHeader.hh"
#include <iomanip>
#include <netinet/in.h>
#include <string.h>

using std::setw;
using std::streampos;

long     XrdMonDecOnePacket::_nextNr    = 0;
dictid_t XrdMonDecOnePacket::_minDictId = 0;
dictid_t XrdMonDecOnePacket::_maxDictId = 0;

XrdMonDecOnePacket::XrdMonDecOnePacket(bool)
    : _myNr(_nextNr++),
      _fPos(-1),
      _seq(LOST),
      _dictId(-1),
      _len(0)
{}

XrdMonDecOnePacket::XrdMonDecOnePacket(int errType, kXR_int64 pos)
    : _myNr(_nextNr++),
      _fPos(pos),
      _seq(errType),
      _dictId(-1),
      _len(0)
{}

int
XrdMonDecOnePacket::init(const char* buf, int bytesLeft, kXR_int64 fPos)
{
    if ( bytesLeft <= HDRLEN ) {
        return -1;
    }
        
    XrdMonHeader header;
    header.decode(buf);

    if ( header.packetLen() > bytesLeft ) {
        return -1;
    }

    _myNr = _nextNr++;
    
    
    _fPos   = fPos;
    _seq    = header.seqNo();
    _dictId = -1;
    _len    = header.packetLen();
    _stod   = header.stod();
    
    if ( header.packetType() == PACKET_TYPE_DICT ) {
        kXR_int32 x32;
        memcpy(&x32, buf+HDRLEN, sizeof(kXR_int32));
        _dictId = ntohl(x32);
    }

    if ( _dictId > 0 && _dictId < _minDictId ) {
        _minDictId = _dictId;
    }
    if ( _dictId > _maxDictId ) {
        _maxDictId = _dictId;
    }

    // skip the rest of the packet
    return header.packetLen();
}

ostream&
operator<<(ostream& o, const XrdMonDecOnePacket& p)
{
    o << setw(6)     << setw(5)  << p._myNr
      << ", fpos="   << setw(12) << p._fPos
      << ", seq="    << setw(3)  << p._seq 
      << ", dictId=" << setw(5)  << p._dictId;

    return o;
}
