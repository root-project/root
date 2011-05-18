/*****************************************************************************/
/*                                                                           */
/*                            XrdMonArgParser.cc                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"

XrdMonArgParser::XrdMonArgParser()
{}

XrdMonArgParser::~XrdMonArgParser()
{}

void
XrdMonArgParser::registerExpectedArg(Arg* arg)
{
    _args.push_back(arg);
}

void
XrdMonArgParser::parseArguments(int argc, char* argv[])
{
    int curArg = 1; // skip the first (program name)
    int i, argSize = _args.size();
    
    while ( curArg < argc ) {
        bool claimed = false;
        for ( i=0 ; i<argSize ; ++i) {
            int x = _args[i]->parseArgs(argc, argv, curArg);
            if ( x > 0 ) {
                curArg += x;
                claimed = true;
                break;
            }
        }
        if ( ! claimed ) {
            string ss("Unexpected argument ");
            ss += argv[curArg];
            throw XrdMonException(ERR_INVALIDARG, ss);
        }
    }

    // now make sure that all the arguments that are 
    // required has been set
    for ( i=0 ; i<argSize ; ++i) {
        _args[i]->throwIfRequiredButNotSet();
    }    
}

