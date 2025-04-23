// $Id: MessageLevels.h,v 1.3 2009-12-16 17:41:24 avalassi Exp $
#ifndef COOLKERNEL_MESSAGELEVELS_H
#define COOLKERNEL_MESSAGELEVELS_H 1

// Workaround for Windows (win32_vc71_dbg)
// ERROR seems to be defined in WinUser.h (required by one of the STL headers?)
// See also RelationalCool/src/CoralApplication.cpp
// See also RelationalCool/utilities/coolPrivileges/RalPrivilegeManager.cpp
#ifdef WIN32
#ifdef ERROR
#undef ERROR
#pragma message ("WARN!NG: in CoolKernel/CoolKernel/MessageLevels.h")
#pragma message ("WARN!NG: 'ERROR' was defined and has been undefined")
#endif
#endif

namespace cool
{

  /** @file MessageLevels.h
   *
   *  Priority levels for messages printed by COOL applications.
   *  The higher the level, the higher the priority of the message.
   *
   *  Messages are handled by a message service whose message level threshold
   *  can be set by the user. The higher the message service threshold, the
   *  less verbose the messaging (as low priority messages are not printed).
   *
   *  The message levels are exactly the same as those used by GAUDI and SEAL.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-17
   */

  namespace MSG
  {

    enum Level {
      NIL = 0,
      VERBOSE,
      DEBUG,
      INFO,
      WARNING,
      ERROR,
      FATAL,
      ALWAYS,
      NUM_LEVELS
    };

  }

}
#endif // COOLKERNEL_MESSAGELEVELS_H
