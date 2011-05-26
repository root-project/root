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

#ifdef WIN32

#include "textinput/StreamReaderWin.h"

#include <io.h>
#include <stdio.h>
#include <Windows.h>

namespace textinput {
  StreamReaderWin::StreamReaderWin(): fHaveInputFocus(false), fIsConsole(true),
    fOldMode(0), fMyMode(0) {
    fIn = ::GetStdHandle(STD_INPUT_HANDLE);
    fIsConsole = ::GetConsoleMode(fIn, &fOldMode) != 0;
    fMyMode = fOldMode | ENABLE_QUICK_EDIT_MODE | ENABLE_EXTENDED_FLAGS; 
    fMyMode = fOldMode & ~(ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT
      | ENABLE_ECHO_INPUT | ENABLE_INSERT_MODE); 
  }

  StreamReaderWin::~StreamReaderWin() {}

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
  StreamReaderWin::HavePendingInput() {
    DWORD ret = ::WaitForSingleObject(fIn, 0);
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
    char C;
    if (fIsConsole) {
      INPUT_RECORD buf;
      if (!::ReadConsoleInput(fIn, &buf, 1, &NRead)) {
        HandleError("reading console input");
        return false;
      }

      switch (buf.EventType) {
      case KEY_EVENT:
      {
        if (!buf.Event.KeyEvent.bKeyDown) return false;

        WORD Key = buf.Event.KeyEvent.wVirtualKeyCode;
        if (buf.Event.KeyEvent.dwControlKeyState
          & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) {
            in.SetModifier(InputData::kModCtrl);
        }
        if ((Key >= 0x30 && Key <= 0x5A /*0-Z*/)
          || (Key >= VK_NUMPAD0 && Key <= VK_DIVIDE)
          || (Key >= VK_OEM_PLUS && Key <= VK_OEM_102)
          || Key == VK_SPACE) {
            C = buf.Event.KeyEvent.uChar.AsciiChar;
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
            default:        in.SetExtended(InputData::kEIUninitialized); return false;
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
      if (!::ReadFile(fIn, &C, 1, &NRead, NULL)) {
        HandleError("reading file input");
        return false;
      }
    }
    HandleKeyEvent(C, in);
    ++nRead;
    return true;
  } 

  void
  StreamReaderWin::HandleError(const char* Where) const {
    DWORD Err = GetLastError(); 
    LPVOID MsgBuf = 0;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS, NULL, Err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &MsgBuf, 0, NULL);

    printf("Error %d in textinput::StreamReaderWin %s: %s\n", Err, Where, MsgBuf); 
    LocalFree(MsgBuf);
  }

  void
  StreamReaderWin::HandleKeyEvent(unsigned char C, InputData& in) {
    if (isprint(C)) {
      in.SetRaw(C);
    } else if (C < 32) {
      in.SetRaw(C);
      in.SetModifier(InputData::kModCtrl);
    } else {
      // woohoo, what's that?!
      in.SetRaw(C);
    }
  }
}
#endif // WIN32
