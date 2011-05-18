/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSndPacket.cc                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndPacket.hh"
#include <string.h>


XrdMonSndPacket::XrdMonSndPacket()
    : _size(0), _data(0)
{}

XrdMonSndPacket::XrdMonSndPacket(const XrdMonSndPacket& p)
    : _size(p._size)
{
    if ( p._data == 0 ) {
        _data = 0;
    } else {
        _data = new char [p.size()];
        memcpy(_data, p._data, p.size());
    }
}

XrdMonSndPacket::~XrdMonSndPacket()
{
    delete [] _data;
}

int
XrdMonSndPacket::init(packetlen_t newSize)
{
    _data = new char[newSize];
    if ( 0 == _data ) {
        return 1; // error
    }
    _size = newSize;
    return 0;
}

void
XrdMonSndPacket::reset()
{
    delete [] _data;
    _size = 0;
}
