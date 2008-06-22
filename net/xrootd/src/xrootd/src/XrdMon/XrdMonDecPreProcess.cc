/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecPreProcess.cc                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonDecPreProcess.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonUtils.hh"

#include <iomanip>
#include "XrdSys/XrdSysHeaders.hh"
#include <sstream>
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::setprecision;
using std::setw;
using std::stringstream;

XrdMonDecPreProcess::XrdMonDecPreProcess(fstream& theFile, 
                             kXR_int64 fSize, 
                             sequen_t lastSeq,
                             kXR_int32 ignoreIfBefore,
                             vector< pair<packetlen_t, kXR_int64> >& allPackets)
    : _file(theFile),
      _fSize(fSize),
      _tempBufPos(-1),
      _markNextSlotAsSpecial(false),
      _ignoreIfBefore(ignoreIfBefore),
      _allPackets(allPackets),
      _lastSeq(lastSeq)
{}

void
XrdMonDecPreProcess::operator()()
{
    _allPackets.reserve(64*1024);
    _lostPackets.reserve(12);
    
    _file.seekg(0, ios::beg);

    checkFile();
    reportAndThrowIfTooBad();

    // prepare file for reading
    _file.seekg(0, ios::beg);
}

void
XrdMonDecPreProcess::checkFile()
{
    kXR_int32 xrdStartTime = 0;
    
    enum { RBUFSIZE = 1024*1024 };
    char rBuf[RBUFSIZE];    

    while ( _fSize > _file.tellg() ) {
        // fill buffer
        kXR_int64 fPos = _file.tellg(); // tellg of this read buffer

        int no2Read = _fSize-fPos > RBUFSIZE ? RBUFSIZE : _fSize-fPos;
        
        _file.read(rBuf, no2Read);
        cout << "Read " << no2Read << " bytes" << endl;
        
        kXR_int64 fPosEnd = _file.tellg();
        int rBufMax = fPosEnd - fPos;        
        
        // process packets until rbuf's size at least MAXPACKETSIZE
        int rPos = 0;
        while ( rPos < rBufMax ) {
            int noBytesRead = processOnePacket(rBuf+rPos, rBufMax-rPos, 
                                               fPos+rPos, xrdStartTime);
            if ( noBytesRead == -1 ) { // only a piece of a packet 
                break;                 // left in buffer
            }
            rPos += noBytesRead;
        }
        _file.seekg(fPos+rBufMax - (rBufMax-rPos));
    }
}

int
XrdMonDecPreProcess::processOnePacket(const char* buf, 
                                      int bytesLeft, 
                                      kXR_int64 fPos, 
                                      kXR_int32& xrdStartTime)
{
    XrdMonDecOnePacket packet;
    int noBytesRead = packet.init(buf, bytesLeft, fPos);
    if ( noBytesRead == -1 ) {
        return -1;
    }
    
    if ( packet.stod() < _ignoreIfBefore ) {
        cout << "Ignoring " << packet 
             << ", timestamp " << packet.stod() << endl;
        XrdMonDecOnePacket::resetNextNr();
        return noBytesRead;
    }
    if ( packet.myNr() == 0 ) {
        xrdStartTime = packet.stod();
        cout << "xrd start time " << xrdStartTime
             << " --> " << timestamp2string(xrdStartTime) << endl;
    } else if ( packet.stod() != xrdStartTime ) {
        // BTW, FIXME memory leak in stringstream::str()
        stringstream ss(stringstream::out);
        ss << "xrd start time changed " << packet.stod() 
           << ", this is not supported";
        throw XrdMonException(ERR_INTERNALERR, ss.str());
    }   

    cout << "XrdMonDecPreprocessing " << packet << endl;
        
    if ( packet.myNr() == 64*1024 ) {
        double perc = (100*(double)_file.tellg())/_fSize;
        int todo = (int) ((100-perc) * 64 * 1024 / perc);
        cout << "Processed 64K packets, currently at " 
             << _file.tellg() << ", total size " << _fSize
             << " (" << perc << "%), ~" << todo 
             << " packets todo" << endl;
        _allPackets.reserve(todo + 64*1024 + 8 * 1024);
    }
        
    kXR_char expected = previousSeq() + 1;
    
    int gap = packet.seq() - expected;

    if ( 0 == gap ) {
        keepPacket(packet);          
        return noBytesRead;
    }
    if ( gap < 0 || ((255 + expected - packet.seq()) < TBUFSIZE) ) {
        // likely to be an out of order packet, do some studies to check
        if ( outOfOrder(packet) ) { // if ooo, all done inside
            return noBytesRead;
        }
    }
    if ( gap < 0 ) {
        gap = 256 - expected + packet.seq();
    }
    while ( gap-- ) {
        XrdMonDecOnePacket lostPacket(XrdMonDecOnePacket::LOST);
        keepPacket(lostPacket);
        cout << "Possibly lost packet" << endl;
    }
    keepPacket(packet);
    return noBytesRead;
}

void
XrdMonDecPreProcess::keepPacket(XrdMonDecOnePacket& packet)
{
    _allPackets.push_back(
             pair<packetlen_t, kXR_int64>(packet.len(), packet.fPos())
                         );
    add2TempBuf(packet);
}

void
XrdMonDecPreProcess::add2TempBuf(XrdMonDecOnePacket& packet)
{
    if ( _tempBufPos < MAXTBUFELEM ) {
        _tempBuf[++_tempBufPos] = packet;
    } else {
        if ( _tempBuf[0].isLost() ) {
            _lostPackets.push_back(_tempBuf[0].myNr());
        }
        int i = 1;
        while ( i<TBUFSIZE ) {
            _tempBuf[i-1] = _tempBuf[i];
            ++i;
        }
        _tempBuf[MAXTBUFELEM] = packet;
    }
}

kXR_char
XrdMonDecPreProcess::previousSeq() const
{
    if ( _tempBufPos == -1 ) {
        return _lastSeq;
    }
    for (int i=_tempBufPos ; i>=0 ; --i) {
        int s = _tempBuf[i].seq();
        if ( s >= 0 ) {
            return s;
        }
    }
    return 0xFF;
}

// returns position in the _packets vector where this packet belongs
bool
XrdMonDecPreProcess::outOfOrder(XrdMonDecOnePacket& packet)
{
    int pos = _tempBufPos; // "pos" - position of ooo packet in the tempBuf
    while ( pos > 0 ) {
        --pos;
        if ( _tempBuf[pos].seq() == XrdMonDecOnePacket::LOST ) {
            if ( _tempBuf[pos-1].seq() + 1 == packet.seq() &&
                 _tempBuf[pos+1].seq() - 1 == packet.seq()   ) {
                cout << "Out of order packet arrived, seq " 
                     << packet.seq() << endl;

                _tempBuf[pos] = packet;

                int apPos = _allPackets.size() - (MAXTBUFELEM - pos) - 1;
                _allPackets[apPos].first  = packet.len();
                _allPackets[apPos].second = packet.fPos();
                cout << "set length " << packet.len()
                     << " for position " << apPos 
                     << ", seq is " << packet.seq() << endl;
                
                _oooPackets.push_back(apPos);
                return true; // yes, ooo packet
            }
        }
    }
    return false; // not an ooo packet
}

void
XrdMonDecPreProcess::reportAndThrowIfTooBad()
{
    int noLost = _lostPackets.size();
    int noOOO  = _oooPackets.size();
    int i, totalNoPackets = _allPackets.size();
    
    double perc = (100*(double)noLost)/totalNoPackets;
    cout << noLost << " packets lost out of " << totalNoPackets
         << " (" << setprecision(2) << perc << "%)"
         << ", " << noOOO << " out-of-order packets" << endl;
    if ( noLost > 0 ) {
        cout << "Lost: ";
        for (i=0 ; i<noLost ; ++i) {
            cout << _lostPackets[i] << ", ";
        }
        cout << endl;
    }
    if ( noOOO > 0 ) {
        cout << "OOO: ";
        for (i=0 ; i<noOOO ; ++i) {
            cout << _oooPackets[i] << ", ";
        }
        cout << endl;
    }

    cout << "dictIds: min=" << XrdMonDecOnePacket::minDictId() 
         << ", max=" << XrdMonDecOnePacket::maxDictId() << endl;

    if ( perc > 5 ) {
        stringstream ss(stringstream::out);
        ss << "Too many lost packets: " << noLost
           << " (" << perc << "%)";
        throw XrdMonException(ERR_TOOMANYLOST, ss.str());
    }

    //cout << " ================" << endl;
    //cout << "allPackets:\n";
    //
    //int s = _allPackets.size();
    //for (i=0 ; i<s ; ++i) {
    //    cout << setw(3) << i 
    //         << setw(5) << ") len = " << setw(4) << _allPackets[i].first
    //         << ", fPos = " << setw(7) << _allPackets[i].second << endl;
    //}
}
