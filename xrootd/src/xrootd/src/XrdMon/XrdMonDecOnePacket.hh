/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecOnePacket.hh                           */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECONEPACKET_HH
#define XRDMONDECONEPACKET_HH

#include "XrdMon/XrdMonCommon.hh"

#include <fstream>
#include <sys/time.h>
using std::fstream;
using std::ostream;

class XrdMonDecOnePacket {
public:
    XrdMonDecOnePacket() {}
    XrdMonDecOnePacket(bool); // LOST
    XrdMonDecOnePacket(int errType, kXR_int64 pos);
    int init(const char* buf, int bytesLeft, kXR_int64 fPos);

    static dictid_t minDictId() { return _minDictId; }
    static dictid_t maxDictId() { return _maxDictId; }
    static void resetNextNr()   { _nextNr = 0;       }

    long myNr()       const { return _myNr; }
    kXR_int64 fPos()    const { return _fPos; }
    kXR_int16 seq()     const { return _seq;  }
    dictid_t dictId() const { return _dictId; }
    packetlen_t len() const { return _len;  }    
    kXR_int32 stod()     const { return _stod; }
    
    bool isLost()     const { return _seq == LOST; }
    
    void setOOOStatus() { _seq = OOO; }
        
  
    enum { REGULAR    = 257, // not lost, not out of order
           LOST       =  -1, // lost packet
           OOO        =  -2, // out of order
           INVALID    =  -3  // just ignore this slot
    };

private:
    static long     _nextNr;
    static dictid_t _minDictId;
    static dictid_t _maxDictId;
    
    long        _myNr;   // id to identify this packet

    kXR_int64   _fPos;   // offset of this packet in the file
    kXR_int16   _seq;    // seqNo, or info: lost/outoforder/emptyslot
    dictid_t    _dictId; // dict id, or -1
    packetlen_t _len;    // packet size
    kXR_int32      _stod;   // when xrd server was started
    
    friend ostream& operator<<(ostream& o, const XrdMonDecOnePacket& p);
};

#endif /* XRDMONDECONEPACKET_HH */
