/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndAdminEntry.hh                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef ADMINENTRY_HH
#define ADMINENTRY_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTypes.hh"

class XrdMonSndAdminEntry {
public:    
    void setShutdown() {
        _command = c_shutdown;
        _arg = 0;
    }
    kXR_int16 size() const         { return 2*sizeof(kXR_int16); }
    AdminCommand command() const { return _command; }
    kXR_int16 arg() const          { return _arg; }
    
private:
    AdminCommand _command;
    kXR_int16 _arg;
};

#endif /* ADMINENTRY_HH */
