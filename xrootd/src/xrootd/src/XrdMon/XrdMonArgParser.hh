/*****************************************************************************/
/*                                                                           */
/*                            XrdMonArgParser.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONARGPARSER_HH
#define XRDMONARGPARSER_HH

#include <stdio.h>
#include <string>
#include <vector>
using std::string;
using std::vector;

class XrdMonArgParser {
public:
    class Arg {
    public:
        virtual ~Arg() {}
        virtual int parseArgs(int argc, char* argv[], int curArg) = 0;
        virtual void throwIfRequiredButNotSet() = 0;
    };

    template <typename T, class C>
    class ArgImpl : public Arg {
    public:
        ArgImpl(const char* theSwitch,    // leading "-blablabla"
                T defaultValue,
                bool required = false);   // required/optional
        virtual ~ArgImpl() {}
        virtual int parseArgs(int argc, char* argv[], int curArg);
        virtual void throwIfRequiredButNotSet();
        T myVal() { return _value; }

    private:
        T            _value;    // the value of the arg
        const string _switch;   // leading switch
        bool         _done;     // arg has been found
        bool         _required; // required/optional
    };

    XrdMonArgParser();
    ~XrdMonArgParser();
    
    void registerExpectedArg(Arg* arg);
    void parseArguments(int argc, char* argv[]);

private:
    vector<Arg*> _args;
};

#include "XrdMon/XrdMonArgParser.icc"

#endif /* XRDMONARGPARSER_HH */
