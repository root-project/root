// $Id: PayloadMode.h,v 1.3 2012-07-08 20:02:33 avalassi Exp $
#ifndef RELATIONALCOOL_PAYLOADMODE_H
#define RELATIONALCOOL_PAYLOADMODE_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#ifdef COOL290VP
#include "CoolKernel/PayloadMode.h"
#else

namespace cool 
{

  /** @file PayloadMode.h
   *
   *  Enum definition for the payload mode of a COOL folder.
   *
   *  Currently three different modes for storing payload exist in COOL:
   *  - In the default mode, the payload is stored inline in the IOV table, 
   *    leading to an overhead from multiple payload copies in system objects.
   *  - The separate payload table mode was introduced to reduce this overhead:
   *    system object rows originating from the same user-inserted IOV contain
   *    multiple copies of the same payload ID, but not of the payload.
   *  - Later also the possibility to add a vector of payloads to one IOV was 
   *    added (vector payload mode). This also needs a separate payload table.
   *
   *  @author Martin Wache
   *  @date 2010-05-18
   */

  // Folder payload mode
  namespace PayloadMode
  {
    enum Mode { INLINEPAYLOAD=1, SEPARATEPAYLOAD, VECTORPAYLOAD };
  }

}
#endif
#endif // RELATIONALCOOL_PAYLOADMODE_H
