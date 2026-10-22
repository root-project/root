//===--- UTF8.h - UTF-8 Conversion And Display Width ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the conversion between UTF-8 (used by everything outside
//  textinput: the interpreter, the history file, Getline's C interface) and
//  UTF-32 (used inside textinput, where one buffer element must be exactly one
//  character), plus the number of terminal columns a character occupies.
//
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_UTF8_H
#define TEXTINPUT_UTF8_H

#include <cstddef>
#include <string>

namespace textinput {

  // The character substituted for malformed input, U+FFFD REPLACEMENT
  // CHARACTER. Decoding never fails; it produces this instead, so that a stray
  // byte from a mistyped paste cannot desynchronize the line buffer.
  const char32_t kInvalidChar = 0xFFFD;

  // Number of bytes in the UTF-8 sequence introduced by Lead, or 0 if Lead is
  // not a valid lead byte (i.e. it is a continuation byte or is never valid).
  inline size_t UTF8SequenceLength(unsigned char Lead) {
    if (Lead < 0x80) return 1;
    if (Lead < 0xC2) return 0; // continuation byte, or overlong lead C0/C1
    if (Lead < 0xE0) return 2;
    if (Lead < 0xF0) return 3;
    if (Lead < 0xF5) return 4; // F5..FF encode beyond U+10FFFF
    return 0;
  }

  inline bool IsUTF8Continuation(unsigned char C) {
    return (C & 0xC0) == 0x80;
  }

  // Whether C is a valid Unicode scalar value, i.e. neither beyond the
  // Unicode range nor one half of a surrogate pair.
  inline bool IsValidCodePoint(char32_t C) {
    return C <= 0x10FFFF && (C < 0xD800 || C > 0xDFFF);
  }

  // Append C to Out as UTF-8. Invalid code points are replaced.
  void AppendUTF8(std::string& Out, char32_t C);

  // Decode a UTF-8 string. Malformed sequences become kInvalidChar.
  std::u32string UTF8ToUTF32(const char* S, size_t Len);
  inline std::u32string UTF8ToUTF32(const std::string& S) {
    return UTF8ToUTF32(S.data(), S.length());
  }

  // Encode as UTF-8.
  std::string UTF32ToUTF8(const char32_t* S, size_t Len);
  inline std::string UTF32ToUTF8(const std::u32string& S) {
    return UTF32ToUTF8(S.data(), S.length());
  }

  // The number of terminal columns taken up by C: 0 for combining marks and
  // other zero-width characters, 2 for East Asian Wide / Fullwidth characters
  // and emoji, 1 for everything else.
  size_t CharWidth(char32_t C);

  // Accumulates the bytes of one UTF-8 sequence as they arrive from a stream.
  // The readers cannot decode a whole buffer at once: they hand textinput one
  // character at a time and must not block waiting for a character that the
  // user has not typed yet.
  class UTF8Decoder {
  public:
    // What Push() did with the byte it was given.
    enum EResult {
      kNeedMore,  // byte consumed, sequence still incomplete
      kComplete,  // byte consumed, Out holds the decoded character
      kInvalid    // sequence is malformed; Out is kInvalidChar and, if
                  // Reprocess is set, the byte was not consumed
    };

    // Feed one byte. On kInvalid with Reprocess == true, the caller must feed
    // C again (to a now-reset decoder), because it starts a new sequence.
    EResult Push(unsigned char C, char32_t& Out, bool& Reprocess);

    void Reset() { fPending = 0; }
    bool IsPending() const { return fPending != 0; }

  private:
    char32_t fValue = 0;   // code point assembled so far
    size_t fPending = 0;   // continuation bytes still expected
    size_t fLength = 0;    // total length of the sequence being decoded
  };
}

#endif // TEXTINPUT_UTF8_H
