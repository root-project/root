//===--- TerminalDisplay.h - Output To Terminal -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the abstract interface for writing to a terminal.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TERMINALDISPLAY_H
#define TEXTINPUT_TERMINALDISPLAY_H

#include <cstddef>                      // for size_t
#include <string>                       // for string
#include <vector>                       // for vector
#include "textinput/Display.h"
#include "textinput/Editor.h"
#include "textinput/Range.h"            // for Range
#include "textinput/Text.h"             // for Text
#include "textinput/TextInputContext.h"

namespace textinput {
  class Color;

  // Base class for output to a terminal.
  class TerminalDisplay: public Display {
  public:
    ~TerminalDisplay();
    static TerminalDisplay* Create();

    void NotifyTextChange(Range r) override;
    void NotifyCursorChange() override;
    void NotifyResetInput() override;
    void NotifyError() override;
    void Detach() override;
    void DisplayInfo(const std::vector<std::string>& Options) override;
    bool IsTTY() const { return fIsTTY; }

  protected:
    TerminalDisplay(bool isTTY):
      fIsTTY(isTTY), fWidth(80), fPrevColor(-1) {}
    void SetIsTTY(bool isTTY) { fIsTTY = isTTY; }
    Pos GetCursor() const {
      // Collect the different prompts and the text cursor to calculate
      // the cursor position in the terminal.
      size_t idx = GetContext()->GetCursor();
      idx += GetContext()->GetPrompt().length();
      idx += GetContext()->GetEditor()->GetEditorPrompt().length();
      return IndexToPos(idx);
    }

    // Lay out the first idx characters of what is displayed - the prompt, the
    // editor prompt and the input line, concatenated - and return where the
    // next character goes.
    //
    // This cannot be index arithmetic: a character is not a column. Combining
    // marks take no column of their own and CJK characters and emoji take two,
    // so the mapping has to walk the text and add up the widths.
    Pos IndexToPos(size_t idx) const;

    // Width, in columns, of character idx of that same concatenation.
    size_t WidthOfDisplayChar(size_t idx) const;

    // Grow r at the front so that it starts on a character that owns a cell,
    // not on a combining mark that shares the previous one.
    void ExtendRangeForCombiningChars(Range& r) const;

    // Place a character of width w and move on, wrapping at the right margin.
    // Every column computation goes through here, so that the cursor position
    // we compute and the position the terminal actually reaches agree.
    void AdvancePos(Pos& p, size_t w) const {
      if (p.fCol + w > fWidth) { // does not fit, starts on the next line
        p.fCol = 0;
        ++p.fLine;
      }
      p.fCol += w;
      if (p.fCol >= fWidth) { // filled the line exactly
        p.fCol = 0;
        ++p.fLine;
      }
    }

    size_t GetWidth() const { return fWidth; }
    void SetWidth(size_t width) { fWidth = width; }

    virtual void Move(Pos p);
    virtual void MoveUp(size_t nLines = 1) = 0;
    virtual void MoveDown(size_t nLines = 1) = 0;
    virtual void MoveLeft(size_t nCols = 1) = 0;
    virtual void MoveRight(size_t nCols = 1) = 0;
    virtual void MoveFront() = 0;
    size_t WriteWrapped(Range::EPromptUpdate PromptUpdate, bool masked,
                        size_t offset, size_t len = (size_t)-1);
    size_t WriteWrappedTextPart(const Text &text, size_t TextOffset,
                                size_t Requested);
    // Write n spaces, to fill the columns a double-width character could not
    // fit into before it wraps to the next line.
    void WriteBlanks(size_t n);
    virtual void SetColor(char CIdx, const Color& C) = 0;
    virtual void WriteRawString(const char* text, size_t len) = 0;
    virtual void ActOnEOL() {}

    virtual void EraseToRight() = 0;

  protected:
    bool fIsTTY; // whether this is a terminal or redirected
    size_t fWidth; // Width of the terminal in character columns
    Pos fWriteEnd; // Position just past the end of the output written.
    Pos fWritePos; // Current position of writing (temporarily != cursor)
    char fPrevColor; // currently configured color
  };
}
#endif // TEXTINPUT_TERMINALDISPLAY_H
