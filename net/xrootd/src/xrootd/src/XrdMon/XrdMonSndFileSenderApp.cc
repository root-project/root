/*****************************************************************************/
/*                                                                           */
/*                        XrdMonSndDummyXrootdApp.cc                         */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonArgParserConvert.hh"
#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonSndDebug.hh"
#include "XrdMon/XrdMonSndTransmitter.hh"

#include <assert.h>
#include <unistd.h>  /* usleep */
#include <sys/time.h>
#include <fstream>
#include <iomanip>
#include "XrdSys/XrdSysHeaders.hh"
#include <strings.h>
#include <sstream>
using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::setfill;
using std::setw;
using std::stringstream;


using namespace XrdMonArgParserConvert;

// known problems with 2 and 4
//const kXR_int64 NOCALLS = 8640000;   24h worth
const kXR_int64 NOCALLS = 1000000000;
const kXR_int16 maxNoXrdMonSndPackets = 5;
const char*     DEFAULT_FILE  = "/tmp/active/rcv";
const kXR_int16 DEFAULT_SLEEP = 0;

void
printHelp()
{
    cout << "\nxrdmonFileSender\n"
         << "    [-host <hostName>]\n"
         << "    [-port <portNr>]\n"
         << "\n"
         << "-host <hostName>         Name of the receiver's host.\n"
         << "                         Default value is \"" << DEFAULT_HOST << "\".\n"
         << "-port <portNr>           Port number of the receiver's host\n"
         << "                         Default valus is \"" << DEFAULT_PORT << "\".\n"
         << "-inputFile <fileName>    Input file name (with path or without).\n"
         << "                         Default value is \"" << DEFAULT_FILE << "\".\n"
         << "-sleep <value>           number of miliseconds to sleep between each packet.\n"
         << "                         Default value is \"" << DEFAULT_SLEEP << "\".\n"
         << endl;
}



int main(int argc, char* argv[])
{
    XrdMonArgParser::ArgImpl<const char*, Convert2String>
        arg_host("-host", DEFAULT_HOST);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_port("-port", DEFAULT_PORT);
    XrdMonArgParser::ArgImpl<const char*, Convert2String>
        arg_file("-inputFile", "/tmp/active.rcv");
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_sleep("-sleep", DEFAULT_SLEEP);

    try {
        XrdMonArgParser argParser;
        argParser.registerExpectedArg(&arg_host);
        argParser.registerExpectedArg(&arg_port);
        argParser.registerExpectedArg(&arg_file);
        argParser.registerExpectedArg(&arg_sleep);
        argParser.parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        printHelp();
        return 1;
    }

    const char* inFName = arg_file.myVal();
    if ( 0 != access(inFName, R_OK) ) {
        cerr << "Invalid input file name \"" << inFName << "\"" << endl;
        return 2;
    }

    kXR_int16 msecSleep = arg_sleep.myVal();

    XrdMonSndDebug::initialize();

    XrdMonSndTransmitter transmitter;

    assert ( !transmitter.initialize(arg_host.myVal(), arg_port.myVal()) );

    fstream _file;
    _file.open(inFName, ios::in|ios::binary);
    _file.seekg(0, ios::end);
    kXR_int64 fSize = _file.tellg();
    _file.seekg(0, ios::beg);

    int uniqueId = 0;
    
    while ( _file && fSize > _file.tellg() ) {
        // read header
        char hBuffer[HDRLEN];
        _file.read(hBuffer, HDRLEN);

        //decode header
        XrdMonHeader header;
        header.decode(hBuffer);
        cout << setw(10) << ++uniqueId << " Sending " << header << endl;

        // read the rest of the packet
        char packetData[MAXPACKETSIZE];
        _file.read(packetData, header.packetLen()-HDRLEN);

        // assemble the packet
        XrdMonSndPacket packet;
        packet.init(header.packetLen());
        bcopy(hBuffer,     packet.offset(0),      HDRLEN);
        bcopy(packetData,  packet.offset(HDRLEN), header.packetLen()-HDRLEN);

        transmitter(packet);

        usleep(msecSleep);
    }
    
    transmitter.shutdown();
    
    return 0;
}
