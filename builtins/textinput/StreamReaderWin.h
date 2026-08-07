//===--- TerminalReaderWin.h - Input From Windows Console -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for reading from Window's cmd.exe console.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_STREAMREADERWIN_H
#define TEXTINPUT_STREAMREADERWIN_H

#include "textinput/StreamReader.h"
#include "textinput/UTF8.h"
#include <Windows.h>

namespace textinput {
  // Windows console and pipe input
  class StreamReaderWin: public StreamReader {
  public:
    StreamReaderWin();
    ~StreamReaderWin();

    void GrabInputFocus();
    void ReleaseInputFocus();

    bool HavePendingInput(bool wait);
    bool ReadInput(size_t& nRead, InputData& in);

    bool IsFromTTY() override { return fIsConsole; }

  private:
    void HandleError(const char* Where) const;
    void HandleKeyEvent(char32_t C, InputData& in);
    // Turn a UTF-16 code unit from the console into a code point. Returns
    // false while waiting for the second half of a surrogate pair, i.e. when
    // there is no character to report yet.
    bool DecodeUTF16(wchar_t U, char32_t& Out);
    // Read one character's worth of UTF-8 from a redirected (non-console)
    // input. Returns false on EOF.
    bool ReadPipeChar(char32_t& Out);

    bool fHaveInputFocus; // whether the console is configured
    bool fIsConsole; // whether the input is a console or file
    HANDLE fIn; // input handle
    DWORD fOldMode; // configuration before grabbing input device
    DWORD fMyMode; // configuration while active
    wchar_t fPendingSurrogate; // high surrogate awaiting its low half
  };
}

#endif // TEXTINPUT_STREAMREADERWIN_H
