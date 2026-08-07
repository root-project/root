//===--- TerminalReaderWin.cpp - Input From Windows Console -----*- C++ -*-===//
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

#ifdef _WIN32

#include "textinput/StreamReaderWin.h"

#include <io.h>
#include <cstdio>
#include <Windows.h>

// MSVC 7.1 is missing these definitions:
#ifndef ENABLE_QUICK_EDIT_MODE
# define ENABLE_QUICK_EDIT_MODE 0x0040
#endif
#ifndef ENABLE_EXTENDED_FLAGS
# define ENABLE_EXTENDED_FLAGS 0x0080
#endif
#ifndef ENABLE_LINE_INPUT
# define ENABLE_LINE_INPUT 0x0002
#endif
#ifndef ENABLE_PROCESSED_INPUT
# define ENABLE_PROCESSED_INPUT 0x0001
#endif
#ifndef ENABLE_ECHO_INPUT
# define ENABLE_ECHO_INPUT 0x0004
#endif
#ifndef ENABLE_INSERT_MODE
# define ENABLE_INSERT_MODE 0x0020
#endif
// End MSVC 7.1 quirks

// winnls.h only defines these for WINVER >= 0x0600.
#ifndef IS_HIGH_SURROGATE
# define IS_HIGH_SURROGATE(wch) (((wch) >= 0xD800) && ((wch) <= 0xDBFF))
#endif
#ifndef IS_LOW_SURROGATE
# define IS_LOW_SURROGATE(wch) (((wch) >= 0xDC00) && ((wch) <= 0xDFFF))
#endif

namespace textinput {
  StreamReaderWin::StreamReaderWin(): fHaveInputFocus(false), fIsConsole(true),
    fOldMode(0), fMyMode(0), fPendingSurrogate(0) {
    fIn = ::GetStdHandle(STD_INPUT_HANDLE);
    bool fIsConsole = ::GetConsoleMode(fIn, &fOldMode) != 0;
    if (fIsConsole) {
      // Allocate our own console handle, to prevent redirection from
      // stealing it.
      fIn = ::CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);
      ::GetConsoleMode(fIn, &fOldMode);
      fMyMode = fOldMode | ENABLE_QUICK_EDIT_MODE | ENABLE_EXTENDED_FLAGS;
      fMyMode &= ~(ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT
        | ENABLE_ECHO_INPUT | ENABLE_INSERT_MODE);
    }
  }

  StreamReaderWin::~StreamReaderWin() {
    if (fIsConsole) {
      // We allocated CONIN$:
      CloseHandle(fIn);
    }
  }

  void
  StreamReaderWin::GrabInputFocus() {
    if (fHaveInputFocus) return;
    if (fIsConsole && !SetConsoleMode(fIn, fMyMode)) {
      fIsConsole = false;
    }
    fHaveInputFocus = true;
  }

  void
  StreamReaderWin::ReleaseInputFocus() {
    if (!fHaveInputFocus) return;
    if (fIsConsole && !SetConsoleMode(fIn, fOldMode)) {
      fIsConsole = false;
    }
    fHaveInputFocus = false;
  }

  bool
  StreamReaderWin::HavePendingInput(bool wait) {
    DWORD ret = ::WaitForSingleObject(fIn,  wait ? INFINITE : 0);
    if (ret == WAIT_FAILED) {
      HandleError("waiting for console input");
      // We don't know. Better block rather than veto input:
      return true;
    }
    return ret == WAIT_OBJECT_0;
  }

  bool
  StreamReaderWin::ReadInput(size_t& nRead, InputData& in) {
    DWORD NRead = 0;
    in.SetModifier(InputData::kModNone);
    char32_t C = 0;
    if (fIsConsole) {
      INPUT_RECORD buf;
      // Read the wide variant: uChar.AsciiChar loses everything the console's
      // code page cannot represent, which is most of Unicode.
      if (!::ReadConsoleInputW(fIn, &buf, 1, &NRead)) {
        HandleError("reading console input");
        return false;
      }

      switch (buf.EventType) {
      case KEY_EVENT:
      {
        if (!buf.Event.KeyEvent.bKeyDown) return false;

        WORD Key = buf.Event.KeyEvent.wVirtualKeyCode;
        const wchar_t Unicode = buf.Event.KeyEvent.uChar.UnicodeChar;
        if (buf.Event.KeyEvent.dwControlKeyState
          & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) {
          if (buf.Event.KeyEvent.dwControlKeyState
             & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)) {
             // special "Alt Gr" case (equivalent to Ctrl+Alt)...
            in.SetModifier(InputData::kModNone);
          }
          else {
            in.SetModifier(InputData::kModCtrl);
          }
        }
        if ((Key >= 0x30 && Key <= 0x5A /*0-Z*/)
          || (Key >= VK_NUMPAD0 && Key <= VK_DIVIDE)
          || (Key >= VK_OEM_1 && Key <= VK_OEM_102)
          || Key == VK_SPACE) {
            // Half a surrogate pair is not yet a character; wait for the rest.
            if (!DecodeUTF16(Unicode, C)) return false;
            if (buf.Event.KeyEvent.dwControlKeyState
              & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) {
               // C is already 1..
            }
        } else {
          switch (Key) {
            case VK_BACK:   in.SetExtended(InputData::kEIBackSpace); break;
            case VK_TAB:    in.SetExtended(InputData::kEITab); break;
            case VK_RETURN: in.SetExtended(InputData::kEIEnter); break;
            case VK_ESCAPE: in.SetExtended(InputData::kEIEsc); break;
            case VK_PRIOR:  in.SetExtended(InputData::kEIPgUp); break;
            case VK_NEXT:   in.SetExtended(InputData::kEIPgDown); break;
            case VK_END:    in.SetExtended(InputData::kEIEnd); break;
            case VK_HOME:   in.SetExtended(InputData::kEIHome); break;
            case VK_LEFT:   in.SetExtended(InputData::kEILeft); break;
            case VK_UP:     in.SetExtended(InputData::kEIUp); break;
            case VK_RIGHT:  in.SetExtended(InputData::kEIRight); break;
            case VK_DOWN:   in.SetExtended(InputData::kEIDown); break;
            case VK_INSERT: in.SetExtended(InputData::kEIIns); break;
            case VK_DELETE: in.SetExtended(InputData::kEIDel); break;
            case VK_F1:     in.SetExtended(InputData::kEIF1); break;
            case VK_F2:     in.SetExtended(InputData::kEIF2); break;
            case VK_F3:     in.SetExtended(InputData::kEIF3); break;
            case VK_F4:     in.SetExtended(InputData::kEIF4); break;
            case VK_F5:     in.SetExtended(InputData::kEIF5); break;
            case VK_F6:     in.SetExtended(InputData::kEIF6); break;
            case VK_F7:     in.SetExtended(InputData::kEIF7); break;
            case VK_F8:     in.SetExtended(InputData::kEIF8); break;
            case VK_F9:     in.SetExtended(InputData::kEIF9); break;
            case VK_F10:    in.SetExtended(InputData::kEIF10); break;
            case VK_F11:    in.SetExtended(InputData::kEIF11); break;
            case VK_F12:    in.SetExtended(InputData::kEIF12); break;
            default:
              // No virtual key code of its own, but it still produced a
              // character: IME composition, a dead key resolving, or an AltGr
              // combination on a non-US layout. Those are exactly the keys
              // that type the non-ASCII characters we are here for.
              if (Unicode >= 0x20 || IS_HIGH_SURROGATE(Unicode)
                  || IS_LOW_SURROGATE(Unicode)) {
                if (!DecodeUTF16(Unicode, C)) return false;
                HandleKeyEvent(C, in);
                ++nRead;
                return true;
              }
              in.SetExtended(InputData::kEIUninitialized); return false;
          }
          return true;
        }
        break;
      }
      case WINDOW_BUFFER_SIZE_EVENT:
        in.SetExtended(InputData::kEIResizeEvent);
        ++nRead;
        return true;
        break;
      default:
        return false;
      }
    } else {
      if (!ReadPipeChar(C)) {
        in.SetExtended(InputData::kEIEOF);
        return true;
      }
    }
    HandleKeyEvent(C, in);
    ++nRead;
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Read one character from redirected input, which is a byte stream and is
  /// taken to be UTF-8 - the same encoding the rest of ROOT uses for text.
  ///
  /// \param[out] Out the character read
  /// \return false at end of input
  bool
  StreamReaderWin::ReadPipeChar(char32_t& Out) {
    UTF8Decoder Dec;
    bool Reprocess = false;
    UTF8Decoder::EResult Res = UTF8Decoder::kNeedMore;
    bool AnyByteRead = false;

    while (Res == UTF8Decoder::kNeedMore) {
      unsigned char Byte = 0;
      DWORD NRead = 0;
      // Testing for the End of a File
      // https://msdn.microsoft.com/en-us/library/windows/desktop/aa365690(v=vs.85).aspx
      if (!::ReadFile(fIn, &Byte, 1, &NRead, NULL)) {
        if (NRead != 0) {
          switch (::GetLastError()) {
            default:
              HandleError("reading file input");
              return false;
            case ERROR_HANDLE_EOF:
            case ERROR_BROKEN_PIPE:
              break;
          }
          NRead = 0;
        }
      }
      if (NRead == 0) {
        // End of input. If it arrived in the middle of a character, report
        // the truncated character rather than losing it silently.
        if (!AnyByteRead) return false;
        Out = kInvalidChar;
        return true;
      }
      AnyByteRead = true;
      Res = Dec.Push(Byte, Out, Reprocess);
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Combine the UTF-16 code units the console hands us into a code point.
  ///
  /// wchar_t is 16 bits on Windows, so anything above the basic multilingual
  /// plane - emoji, most notably - arrives as two events that have to be put
  /// back together.
  ///
  /// \param[in] U the code unit just read
  /// \param[out] Out the character, when one is complete
  /// \return false if this was a high surrogate and the low half is still to come
  bool
  StreamReaderWin::DecodeUTF16(wchar_t U, char32_t& Out) {
    if (fPendingSurrogate) {
      const wchar_t High = fPendingSurrogate;
      fPendingSurrogate = 0;
      if (IS_LOW_SURROGATE(U)) {
        Out = 0x10000 + ((static_cast<char32_t>(High) - 0xD800) << 10)
                      + (static_cast<char32_t>(U) - 0xDC00);
        return true;
      }
      // The high surrogate was never completed; drop it and carry on with U.
    }
    if (IS_HIGH_SURROGATE(U)) {
      fPendingSurrogate = U;
      return false;
    }
    if (IS_LOW_SURROGATE(U)) { // unpaired
      Out = kInvalidChar;
      return true;
    }
    Out = U;
    return true;
  }

  void
  StreamReaderWin::HandleError(const char* Where) const {
    DWORD Err = GetLastError();
    LPVOID MsgBuf = 0;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS, NULL, Err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &MsgBuf, 0, NULL);

    printf("Error %d in textinput::StreamReaderWin %s: %s\n", Err, Where, (const char *)MsgBuf);
    LocalFree(MsgBuf);
  }

  void
  StreamReaderWin::HandleKeyEvent(char32_t C, InputData& in) {
    if (C < 0x80 && isprint(static_cast<int>(C))) {
      in.SetRaw(C);
    } else if (C < 32) {
      in.SetRaw(C);
      in.SetModifier(InputData::kModCtrl);
    } else {
      // Everything else, including every character outside ASCII.
      in.SetRaw(C);
    }
  }
}
#endif // _WIN32
