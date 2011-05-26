#ifndef WIN32

//===--- TerminalDisplayUnix.cpp - Output To UNIX Terminal ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for writing to a UNIX terminal. It tries to
//  support all "common" terminal types.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/TerminalDisplayUnix.h"

#include "textinput/TerminalConfigUnix.h"
#include "textinput/Color.h"

#include <stdio.h>
// putenv not in cstdlib on Solaris
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <csignal>
#include <cstring>
#include <sstream>

using std::signal;
using std::strstr;

namespace {
  textinput::TerminalDisplayUnix*& gTerminalDisplayUnix() {
    static textinput::TerminalDisplayUnix* S = 0;
    return S;
  }

  void InitRGB256(unsigned char rgb256[][3]) {
    // initialize the array with the expected standard colors:
    // (from http://frexx.de/xterm-256-notes)
    unsigned char rgbidx = 0;
    // this is not what I see, though it's supposedly the default:
    //   rgb[0][0] =   0; rgb[0][1] =   0; rgb[0][1] =   0;
    // use this instead, just to be on the safe side:
    rgb256[0][0] =  46; rgb256[0][1] =  52; rgb256[0][1] =  64;
    rgb256[1][0] = 205; rgb256[1][1] =   0; rgb256[1][1] =   0;
    rgb256[2][0] =   0; rgb256[2][1] = 205; rgb256[2][1] =   0;
    rgb256[3][0] = 205; rgb256[3][1] = 205; rgb256[3][1] =   0;
    rgb256[4][0] =   0; rgb256[4][1] =   0; rgb256[4][1] = 238;
    rgb256[5][0] = 205; rgb256[5][1] =   0; rgb256[5][1] = 205;
    rgb256[6][0] =   0; rgb256[6][1] = 205; rgb256[6][1] = 205;
    rgb256[7][0] = 229; rgb256[7][1] = 229; rgb256[7][1] = 229;
    
    // this is not what I see, though it's supposedly the default:
    //   rgb256[ 8][0] = 127; rgb256[ 8][1] = 127; rgb256[ 8][1] = 127;
    // use this instead, just to be on the safe side:
    rgb256[ 8][0] =   0; rgb256[ 8][1] =   0; rgb256[ 8][1] =   0;
    rgb256[ 9][0] = 255; rgb256[ 9][1] =   0; rgb256[ 9][1] =   0;
    rgb256[10][0] =   0; rgb256[10][1] = 255; rgb256[10][1] =   0;
    rgb256[11][0] = 255; rgb256[11][1] = 255; rgb256[11][1] =   0;
    rgb256[12][0] =  92; rgb256[12][1] =  92; rgb256[12][1] = 255;
    rgb256[13][0] = 255; rgb256[13][1] =   0; rgb256[13][1] = 255;
    rgb256[14][0] =   0; rgb256[14][1] = 255; rgb256[14][1] = 255;
    rgb256[15][0] = 255; rgb256[15][1] = 255; rgb256[15][1] = 255;
    
    for (unsigned char red = 0; red < 6; ++red) {
      for (unsigned char green = 0; green < 6; ++green) {
        for (unsigned char blue = 0; blue < 6; ++blue) {
          rgbidx = 16 + (red * 36) + (green * 6) + blue;
          rgb256[rgbidx][0] = red ? (red * 40 + 55) : 0;
          rgb256[rgbidx][1] = green ? (green * 40 + 55) : 0;
          rgb256[rgbidx][2] = blue ? (blue * 40 + 55) : 0;
        }
      }
    }
    // colors 232-255 are a grayscale ramp, intentionally leaving out
    // black and white
    for (unsigned char gray = 0; gray < 24; ++gray) {
      unsigned char level = (gray * 10) + 8;
      rgb256[232 + gray][0] = level;
      rgb256[232 + gray][1] = level;
      rgb256[232 + gray][2] = level;
    }
  }
} // unnamed namespace

extern "C" void TerminalDisplayUnix__handleResizeSignal(int) {
  gTerminalDisplayUnix()->HandleResizeSignal();
}

namespace textinput {
  // If input is not a tty don't write in tty-mode either.
  TerminalDisplayUnix::TerminalDisplayUnix():
    fIsAttached(false), fNColors(16),
    fIsTTY(isatty(fileno(stdin)) && isatty(fileno(stdout))) {
    HandleResizeSignal();
    gTerminalDisplayUnix() = this;
    signal(SIGWINCH, TerminalDisplayUnix__handleResizeSignal);
#ifdef TCSANOW
    TerminalConfigUnix::Get().TIOS()->c_lflag &= ~(ECHO);
    TerminalConfigUnix::Get().TIOS()->c_lflag |= ECHOCTL|ECHOKE|ECHOE;
#endif
    const char* TERM = getenv("TERM");
    if (TERM &&  strstr(TERM, "256")) {
      fNColors = 256;
    }
  }

  TerminalDisplayUnix::~TerminalDisplayUnix() {
    Detach();
  }
  
  void
  TerminalDisplayUnix::HandleResizeSignal() {
#ifdef TIOCGWINSZ
    struct winsize sz;
    int ret = ioctl(fileno(stdout), TIOCGWINSZ, (char*)&sz);
    if (!ret && sz.ws_col) {
      SetWidth(sz.ws_col);

      // Export what we found.
      std::stringstream s;
      s << sz.ws_col;
      setenv("COLUMS", s.str().c_str(), 1 /*overwrite*/);
      s.clear();
      s << sz.ws_row;
      setenv("LINES", s.str().c_str(), 1 /*overwrite*/);
    }
#else
    // try $COLUMNS
    const char* COLUMNS = getenv("COLUMNS");
    if (COLUMNS) {
      long width = atol(COLUMNS);
      if (width > 4 && width < 1024*16) {
        SetWidth(width);
      }
    }
#endif
  }

  void
  TerminalDisplayUnix::SetColor(char CIdx, const Color& C) {
    if (!fIsTTY) return;
    
    if (CIdx == 0) {
      // Default color, reset.
      static const char text[] = {(char)0x1b, '[', '0', 'm', 0};
      WriteRawString(text, 4);
      return;
    }

    int ANSIIdx = 0;
    if (fNColors == 256) {
      ANSIIdx = GetClosestColorIdx256(C);
    } else {
      ANSIIdx = GetClosestColorIdx16(C);
    }
    char buf[6] = {'\x1b', '[', '0', '0', 'm', 0};
    int val = 30 + ANSIIdx;
    if (val > 37) {
      val = 90 - 30 - 8;
    }
    if (val > 97) {
      printf("ERROR in SetColor, ANSIIdx=%d, val=%d, RGB=(%d,%d,%d)\n", ANSIIdx,
             val,(int)C.fR,(int)C.fG,(int)C.fB);
    }
    buf[2] += val / 10;
    buf[3] += val % 10;
    WriteRawString(buf, 5);
  }
  
  void
  TerminalDisplayUnix::MoveFront() {
    static const char text[] = {(char)0x1b, '[', '1', 'G', 0};
    if (!fIsTTY) return;
    WriteRawString(text, sizeof(text));
  }
  
  void
  TerminalDisplayUnix::MoveInternal(char What, size_t n) {
    static const char cmd[] = "\x1b[";
    if (!fIsTTY) return;
    std::string text;
    for (size_t i = 0; i < n; ++i) {
      text += cmd;
      text += What;
    }
    WriteRawString(text.c_str(), text.length()); 
  }

  void
  TerminalDisplayUnix::MoveUp(size_t nLines /* = 1 */) {
    MoveInternal('A', nLines);
  }

  void
  TerminalDisplayUnix::MoveDown(size_t nLines /* = 1 */) {
    if (!fIsTTY) return;
    std::string moves(nLines, 0x0a);
    WriteRawString(moves.c_str(), nLines);
  }
  
  void
  TerminalDisplayUnix::MoveRight(size_t nCols /* = 1 */) {
    MoveInternal('C', nCols);
  }
  
  void
  TerminalDisplayUnix::MoveLeft(size_t nCols /* = 1 */) {
    MoveInternal('D', nCols);
  }
  
  void
  TerminalDisplayUnix::EraseToRight() {
    static const char text[] = {(char)0x1b, '[', 'K', 0};
    if (!fIsTTY) return;
    WriteRawString(text, sizeof(text));
  }

  void
  TerminalDisplayUnix::WriteRawString(const char *text, size_t len) {
    if (write(fileno(stdout), text, len) == -1) {
      // Silence Ubuntu's "unused result". We don't care if it fails.
    }
  }

  void
  TerminalDisplayUnix::ActOnEOL() {
    if (!fIsTTY) return;
    WriteRawString(" \b", 2);
    //MoveUp();
  }
  
  void
  TerminalDisplayUnix::Attach() {
    // set to noecho
    if (fIsAttached) return;
    fflush(stdout);
    TerminalConfigUnix::Get().Attach();
    fIsAttached = true;
    NotifyTextChange(Range::AllWithPrompt());
  }
  
  void
  TerminalDisplayUnix::Detach() {
    if (!fIsAttached) return;
    fflush(stdout);
    TerminalConfigUnix::Get().Detach();
    TerminalDisplay::Detach();
    fIsAttached = false;
  }

  int
  TerminalDisplayUnix::GetClosestColorIdx16(const Color& C) {
    int r = C.fR;
    int g = C.fG;
    int b = C.fB;
    int sum = r + g + b;
    r = r > sum / 4;
    g = g > sum / 4;
    b = b > sum / 4;
    
    // ANSI:
    return r + (g * 2) + (b * 4);
    // ! ANSI:
    // return (r * 4) + (g * 2) + b;
  }

  int
  TerminalDisplayUnix::GetClosestColorIdx256(const Color& C) {
    static unsigned char rgb256[256][3] = {{0}};
    if (rgb256[0][0] == 0) {
      InitRGB256(rgb256);
    }
    
    // Find the closest index.
    // A: the closest color match (square of geometric distance in RGB)
    // B: the closest brightness match
    // Treat them equally, which suppresses differences
    // in color due to squared distance.
    
    // start with black:
    int idx = 0;
    char r = C.fR;
    char g = C.fG;
    char b = C.fB;
    int graylvl = (r + g + b)/3;
    long mindelta = (r*r + g*g + b*b) + graylvl;
    if (mindelta) {
      for (unsigned int i = 1; i < 256; ++i) {
        long delta = (rgb256[i][0] + rgb256[i][1] + rgb256[i][2])/3 - graylvl;
        if (delta < 0) delta = -delta;
        delta += (r-rgb256[i][0])*(r-rgb256[i][0]) +
        (g-rgb256[i][1])*(g-rgb256[i][1]) +
        (b-rgb256[i][2])*(b-rgb256[i][2]);
        
        if (delta < mindelta) {
          mindelta = delta;
          idx = i;
          if (mindelta == 0) break;
        }
      }
    }
    return idx;
  }
        
}

#endif // #ifndef WIN32
