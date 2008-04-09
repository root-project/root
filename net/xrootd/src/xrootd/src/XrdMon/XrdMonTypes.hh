/*****************************************************************************/
/*                                                                           */
/*                              XrdMonTypes.hh                               */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XProtocol/XPtypes.hh"
#include <algorithm>

typedef kXR_int32 length_t;
typedef kXR_int32 dictid_t;
typedef kXR_char  packet_t;
typedef kXR_char  sequen_t;
typedef kXR_int16 packetlen_t;
typedef kXR_unt16 senderid_t;
typedef std::pair<char*, senderid_t> hp_t;

