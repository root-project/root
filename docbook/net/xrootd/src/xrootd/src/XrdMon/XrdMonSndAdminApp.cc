/*****************************************************************************/
/*                                                                           */
/*                           XrdMonSndAdminApp.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonArgParserConvert.hh"
#include "XrdMon/XrdMonSndAdminEntry.hh"
#include "XrdMon/XrdMonSndCoder.hh"
#include "XrdMon/XrdMonSndTransmitter.hh"
#include <assert.h>
using namespace XrdMonArgParserConvert;

void
printHelp()
{
    cout << "\nxrdmonAdmin\n"
         << "    [-host <hostName>]\n"
         << "    [-port <portNr>]\n"
         << "\n"
         << "-host <hostName>         Name of the receiver's host.\n"
         << "                         Default value is \"" << DEFAULT_HOST << "\".\n"
         << "-port <portNr>           Port number of the receiver's host\n"
         << "                         Default valus is \"" << DEFAULT_PORT << "\".\n"
         << endl;
}

int main(int argc, char* argv[])
{
    XrdMonArgParser::ArgImpl<const char*, Convert2String>
        arg_host("-host", DEFAULT_HOST);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_port("-port", DEFAULT_PORT);

    try {
        XrdMonArgParser argParser;
        argParser.registerExpectedArg(&arg_host);
        argParser.registerExpectedArg(&arg_port);
        argParser.parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        printHelp();
        return 1;
    }

    XrdMonSndDebug::initialize();

    XrdMonSndCoder coder;
    XrdMonSndTransmitter transmitter;
    assert ( !transmitter.initialize(arg_host.myVal(), arg_port.myVal()) );

    XrdMonSndAdminEntry ae;
    ae.setShutdown();
    coder.prepare2Transfer(ae);
    transmitter(coder.packet());

    return 0;
}
