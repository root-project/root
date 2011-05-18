/*****************************************************************************/
/*                                                                           */
/*                          XrdMonDumpPacketsApp.cc                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonDecArgParser.hh"
#include <fstream>
#include <iomanip>
#include "XrdSys/XrdSysHeaders.hh"
#include <sstream>
using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::setfill;
using std::setw;
using std::stringstream;

void 
dumpOnePacket(kXR_int64 uniqueId, fstream& _file)
{
    // prepare file for writing
    stringstream ss(stringstream::out);
    ss << "/tmp/ap.dump." << setw(6) << setfill('0') << uniqueId;
    string filePath = ss.str();
    fstream ofs(filePath.c_str(), ios::out|ios::binary);

    // read header, dump to file
    char hBuffer[HDRLEN];
    _file.read(hBuffer, HDRLEN);
    ofs.write(hBuffer, HDRLEN);

    //decode header
    XrdMonHeader header;
    header.decode(hBuffer);
    cout << "Dumping packet " << uniqueId << " to " << filePath.c_str() << ", "
         << "inputfile tellg: " << setw(10) << (kXR_int64) _file.tellg()-HDRLEN 
         << ", header: " << header << endl;
    
    // read packet, dump to file
    char packet[MAXPACKETSIZE];
    _file.read(packet, header.packetLen()-HDRLEN);
    ofs.write(packet, header.packetLen()-HDRLEN);

    // close output file
    ofs.close();
}

int main(int argc, char* argv[])
{
    try {
        XrdMonDecArgParser::parseArguments(argc, argv);
        const char* fName = XrdMonDecArgParser::_fPath.c_str();

        fstream _file;
        _file.open(fName, ios::in|ios::binary);
        _file.seekg(0, ios::end);
        kXR_int64 fSize = _file.tellg();
        _file.seekg(XrdMonDecArgParser::_offset2Dump, ios::beg);
    
        if (XrdMonDecArgParser::_offset2Dump != 0 ) {
            dumpOnePacket(XrdMonDecArgParser::_offset2Dump, _file);
        } else {
            kXR_int64 packetNo = 0;
            while ( fSize > _file.tellg() ) {
                dumpOnePacket(++packetNo, _file);
            }
        }
        } catch (XrdMonException& e) {
        e.printIt();
        return 1;
    }

    return 0;
}
