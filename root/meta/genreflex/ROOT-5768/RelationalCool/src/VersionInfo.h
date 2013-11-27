// $Id: VersionInfo.h,v 1.47 2010-03-31 09:41:06 avalassi Exp $
#ifndef RELATIONALCOOL_VERSION_INFO_H
#define RELATIONALCOOL_VERSION_INFO_H

// Include files
#include "CoolKernel/VersionInfo.h"

// Local include files
#include "VersionNumber.h"

namespace cool {

  /** VersionInfo.h
   *
   * Code version number and schema version number for the current release.
   *
   * @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   * @date 2005-04-25
   */

  namespace VersionInfo
  {
#ifdef COOL_VERSIONINFO_RELEASE
    const VersionNumber release = COOL_VERSIONINFO_RELEASE;
#else
#error COOL_VERSIONINFO_RELEASE is not defined
#endif
    const VersionNumber schemaVersion = "2.0.0";
    const std::string schemaEvolutionPrefix = "SE_2_2_0_";
  }

} // namespace

#endif
