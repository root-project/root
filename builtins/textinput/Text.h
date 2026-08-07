//===--- Text.h - Colored Text ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for a string plus its characters' color
// indexes.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TEXT_H
#define TEXTINPUT_TEXT_H
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include "textinput/Range.h"
#include "textinput/UTF8.h"

namespace textinput {
  class Colorizer;

  // A colored string.
  //
  // Stored as UTF-32 so that one element is exactly one character: every index
  // in this class - and thus the cursor, the ranges handed to the display and
  // the color vector - counts characters, never bytes. The UTF-8 form, which
  // is what everything outside textinput speaks, is produced by GetText() and
  // cached until the text is modified.
  //
  // Note that one character is not one terminal column: see GetWidthOfChar().
  class Text {
  public:
    Text() {}
    Text(const char* S): fString(UTF8ToUTF32(S, std::char_traits<char>::length(S))),
      fColor(fString.length()) {}
    Text(const std::string& S, char C = 0): fString(UTF8ToUTF32(S)),
      fColor(fString.length(), C) {}
    Text(const std::u32string& S, char C = 0): fString(S),
      fColor(S.length(), C) {}

    // The text as UTF-8. Cached; invalidated by every mutation.
    const std::string& GetText() const {
      UpdateUTF8();
      return fUTF8;
    }
    const std::u32string& GetChars() const { return fString; }

    const std::vector<char>& GetColors() const { return fColor; }
    std::vector<char>& GetColors() { return fColor; }
    char GetColor(size_t i) const { return fColor[i]; }

    // Number of characters, not bytes and not columns.
    size_t length() const { return fString.length(); }
    bool empty() const { return fString.empty(); }

    // Number of terminal columns taken up by character i, and by the
    // characters in [from, to).
    // Not called GetCharWidth(): windows.h #defines that to GetCharWidthA.
    size_t GetWidthOfChar(size_t i) const { return CharWidth(fString[i]); }
    size_t GetWidth(size_t from, size_t to) const {
      if (to > length()) to = length();
      size_t W = 0;
      for (size_t i = from; i < to; ++i) W += CharWidth(fString[i]);
      return W;
    }

    // Byte offset of character i in GetText(); i may be length(), giving the
    // total byte count. Used to hand whole characters to the terminal.
    size_t GetByteOffset(size_t i) const {
      UpdateUTF8();
      return fByteOffset[i < fByteOffset.size() ? i : fByteOffset.size() - 1];
    }

    // Inverse of GetByteOffset(): index of the character containing the byte
    // at Offset in GetText(). An offset at or past the end gives length().
    size_t GetCharIndex(size_t Offset) const {
      UpdateUTF8();
      return std::upper_bound(fByteOffset.begin(), fByteOffset.end(), Offset)
             - fByteOffset.begin() - 1;
    }

    std::u32string substr(size_t pos, size_t len = std::u32string::npos) const {
      return fString.substr(pos, len);
    }

    void insert(size_t pos, char32_t C) {
      // Insert C at pos, set to default color.
      fString.insert(pos, 1, C); fColor.insert(fColor.begin() + pos, 0);
      fUTF8Valid = false;
    }
    void insert(size_t pos, const std::u32string& S) {
      // Insert S at pos, set to default color.
      fColor.insert(fColor.begin() + pos, S.length(), 0);
      fString.insert(pos, S);
      fUTF8Valid = false;
    }
    void erase(size_t pos, size_t len = 1) {
      // Erase len characters starting at pos.
      fString.erase(pos, len);
      fColor.erase(fColor.begin() + pos, fColor.begin() + pos + len);
      fUTF8Valid = false;
    }
    void clear() { fString.clear(); fColor.clear(); fUTF8Valid = false; }

    void
    SetColor(const Range &R, char C) {
      // Set colors of characters in range R to C.
      size_t len = R.fLength;
      if (len == (size_t) -1) {
        len = length() - R.fStart;
      }
      std::fill_n(fColor.begin() + R.fStart, len, C);
    }

    char32_t operator[](size_t i) const { return fString[i]; }
    // No non-const operator[]: handing out a reference would let a caller
    // change the text behind the back of the UTF-8 cache.
    void SetChar(size_t i, char32_t C) { fString[i] = C; fUTF8Valid = false; }

    Text& operator+=(char32_t C) { insert(length(), C); return *this; }
    Text& operator=(const std::string& S) {
      // Assign UTF-8 string S to this, initialize with default colors.
      fString = UTF8ToUTF32(S);
      fColor.assign(fString.length(), 0);
      fUTF8Valid = false;
      return *this;
    }
    Text& operator=(const std::u32string& S) {
      fString = S;
      fColor.assign(S.length(), 0);
      fUTF8Valid = false;
      return *this;
    }
  private:
    // Rebuild the UTF-8 form and the character -> byte offset table, if the
    // text has changed since they were last built.
    void UpdateUTF8() const {
      if (fUTF8Valid) return;
      fUTF8.clear();
      fUTF8.reserve(fString.length());
      fByteOffset.clear();
      fByteOffset.reserve(fString.length() + 1);
      for (char32_t C : fString) {
        fByteOffset.push_back(fUTF8.length());
        AppendUTF8(fUTF8, C);
      }
      fByteOffset.push_back(fUTF8.length()); // one past the end
      fUTF8Valid = true;
    }

    std::u32string fString; // actual text, one element per character
    std::vector<char> fColor; // color index of chars; Colorizer converts to RGB
    mutable std::string fUTF8; // cache of fString as UTF-8
    mutable std::vector<size_t> fByteOffset; // offset of each char in fUTF8
    mutable bool fUTF8Valid = false; // whether the two caches are up to date
  };
}
#endif // TEXTINPUT_TEXT_H
