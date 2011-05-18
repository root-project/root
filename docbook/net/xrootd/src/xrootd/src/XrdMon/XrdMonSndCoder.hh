/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSndCoder.hh                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSNDCODER_HH
#define XRDMONSNDCODER_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonSndAdminEntry.hh"
#include "XrdMon/XrdMonSndDebug.hh"
#include "XrdMon/XrdMonSndDictEntry.hh"
#include "XrdMon/XrdMonSndStageEntry.hh"
#include "XrdMon/XrdMonSndPacket.hh"
#include "XrdMon/XrdMonSndTraceEntry.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include <assert.h>
#include <netinet/in.h>
#include <utility> // for pair
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::pair;
using std::vector;

// The class responsible for coding data into a binary packet

class XrdMonSndCoder {
public:
    XrdMonSndCoder();

    int prepare2Transfer(const XrdMonSndAdminEntry& ae);
    int prepare2Transfer(const vector<XrdMonSndTraceEntry>& vector);
    int prepare2Transfer(const vector<kXR_int32>& vector);
    int prepare2Transfer(const XrdMonSndDictEntry::CompactEntry& ce);
    int prepare2Transfer(const XrdMonSndStageEntry::CompactEntry& ce);

    const XrdMonSndPacket& packet() { return _packet; }
    void reset() { _packet.reset(); }
    void printStats() const ;
    
private:
    char* writeHere() { return _packet.offset(_putOffset); }
    int reinitXrdMonSndPacket(packetlen_t newSize, char packetCode);
    pair<char, kXR_unt32> generateBigNumber(const char* descr);
    
    inline void add_int08_t(int8_t value) {
        memcpy(writeHere(), &value, sizeof(int8_t));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored int08_t value " << (int) value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(int8_t);
    }
    inline void add_kXR_int16(kXR_int16 value) {
        kXR_int16 v = htons(value);
        memcpy(writeHere(), &v, sizeof(kXR_int16));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored kXR_int16 value " << value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(kXR_int16);
    }
    inline void add_kXR_unt16(kXR_unt16 value) {
        kXR_unt16 v = htons(value);
        memcpy(writeHere(), &v, sizeof(kXR_unt16));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored kXR_unt16 value " << value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(kXR_unt16);
    }
    inline void add_kXR_int32(kXR_int32 value) {
        kXR_int32 v = htonl(value);
        memcpy(writeHere(), &v, sizeof(kXR_int32));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored kXR_int32 value " << value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(kXR_int32);
    }
    inline void add_kXR_unt32(kXR_unt32 value) {
        kXR_unt32 v = htonl(value);
        memcpy(writeHere(), &v, sizeof(kXR_unt32));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored kXR_unt32 value " << value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(kXR_unt32);
    }
    inline void add_kXR_int64(kXR_int64 value) {
        kXR_int64 v = htonll(value);
        memcpy(writeHere(), &v, sizeof(kXR_int64));
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored kXR_int64 value " << value 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sizeof(kXR_int64);
    }
    inline void add_Mark(char mark, int noChars=8) {
        assert(noChars<=8);
        char x[8];
        memset(x, 0, 8);
        x[0] = mark;
        memcpy(writeHere(), x, 1);
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored mark " << mark 
                 << ", _putOffset " << _putOffset << endl;
        }

        _putOffset += noChars;
    }
    inline void add_string(const string& s) {
        kXR_int16 sLen = s.size();
        if ( 0 == sLen ) {
            cerr << "Error in add_string, size 0" << endl;
            return;
        }
        memcpy(writeHere(), s.c_str(), sLen);
        if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
            cout << "stored string " << s 
                 << ", _putOffset " << _putOffset << endl;
        }
        _putOffset += sLen;
    }

private:
    XrdMonSndPacket  _packet;
    kXR_int32 _putOffset; // tracks where to write inside packet
    sequen_t _sequenceNo;

    static kXR_int32 _serverStartTime;
    
    // statistics
    kXR_int32 _noDict;
    kXR_int32 _noOpen;
    kXR_int32 _noClose;
    kXR_int32 _noTrace;
    kXR_int32 _noTime;
};

#endif /* XRDMONSNDCODER_HH */
