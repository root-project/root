
//===--- TerminalConfigUnix.cpp - termios storage ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TerminalReader and TerminalDisplay need to reset the terminal configuration
// upon destruction, to leave the terminal as before. To avoid a possible
// misunderstanding of what "before" means, centralize their storage of the
// previous termios and have them share it.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_UNIXTERMINALSETTINGS_H
#define TEXTINPUT_UNIXTERMINALSETTINGS_H

struct termios;

namespace textinput {
class TerminalConfigUnix {
public:
  ~TerminalConfigUnix();

  static TerminalConfigUnix& Get();
  termios* TIOS() { return fConfTIOS; }

  void Attach();
  void Detach();
  bool IsAttached() { return fIsAttached; }

private:
  TerminalConfigUnix();

  bool fIsAttached; // whether fConfTIOS is active.
  termios* fOldTIOS; // tty configuration before grabbing
  termios* fConfTIOS; // tty configuration while active
};
   
}
#endif // TEXTINPUT_UNIXTERMINALSETTINGS_H
