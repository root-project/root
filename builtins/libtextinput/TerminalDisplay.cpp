//===--- TerminalDisplay.cpp - Output To Terminal ---------------*- C++ -*-===//
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

#include "textinput/TerminalDisplay.h"

#ifdef _WIN32
#include "textinput/TerminalDisplayWin.h"
#else
#include "textinput/TerminalDisplayUnix.h"
#endif

#include "textinput/TextInput.h"
#include "textinput/Color.h"
#include "textinput/Text.h"
#include "textinput/Editor.h"

namespace textinput {
  TerminalDisplay::~TerminalDisplay() {}

  TerminalDisplay*
  TerminalDisplay::Create() {
#ifdef _WIN32
    return new TerminalDisplayWin();
#else
    return new TerminalDisplayUnix();
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the text has been changed in range r.
  /// Rewrite the display in range r and move back to the cursor.
  ///
  /// \param[in] r Range to write out the text for.
  void
  TerminalDisplay::NotifyTextChange(Range r) {
    if (!IsTTY()) return;
    Attach();
    ExtendRangeForCombiningChars(r);
    WriteWrapped(r.fPromptUpdate, GetContext()->GetTextInput()->IsInputMasked(),
      r.fStart, r.fLength);
    Move(GetCursor());
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Move the start of r back to the character whose terminal cell the change
  /// really affects.
  ///
  /// A zero-width character - a combining accent, say - is not drawn into a
  /// cell of its own but into the cell of the character it follows. So
  /// redrawing from the mark would leave the base character behind, and
  /// redrawing from just past a mark that was deleted would leave the mark
  /// itself on the screen. Both are fixed by starting one character earlier
  /// and then skipping back over any further marks.
  ///
  /// \param[in,out] r range to redraw, in characters of the input line
  void
  TerminalDisplay::ExtendRangeForCombiningChars(Range& r) const {
    if (r.fStart == 0) return;
    const Text& Line = GetContext()->GetLine();
    if (r.fStart > Line.length()) return;

    size_t Start = r.fStart;
    do {
      --Start;
    } while (Start > 0 && Line.GetWidthOfChar(Start) == 0);

    if (r.fLength != Range::End()) {
      r.fLength += r.fStart - Start;
    }
    r.fStart = Start;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the cursor has been changed. Move to the cursor.
  void
  TerminalDisplay::NotifyCursorChange() {
    Attach();
    Move(GetCursor());
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the input has been taken.
  /// Move to the next line, reset written length and position.
  void
  TerminalDisplay::NotifyResetInput() {
    Attach();
    if (IsTTY()) {
      WriteRawString("\n", 1);
    }
    fWriteEnd = Pos();
    fWritePos = Pos();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that there has been an error.
  /// Write out the BEL character.
  void
  TerminalDisplay::NotifyError() {
    Attach();
    WriteRawString("\x07", 1);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Display an informational message at the prompt.
  /// Acts like a pop-up. Used e.g. for tab-completion.
  ///
  /// \param[in] Options options to write out
  void
  TerminalDisplay::DisplayInfo(const std::vector<std::string>& Options) {
    char infoColIdx = 0;
    if (GetContext()->GetColorizer()) {
       infoColIdx = GetContext()->GetColorizer()->GetInfoColor();
    }
    WriteRawString("\n", 1);
    for (size_t i = 0, n = Options.size(); i < n; ++i) {
      Text t(Options[i], infoColIdx);
      // Each option starts on a line of its own.
      fWritePos.fCol = 0;
      WriteWrappedTextPart(t, 0, (size_t) -1);
      WriteRawString("\n", 1);
    }
    // Reset position
    Detach();
    Attach();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Detach from the abstract display by resetting the position
  /// and written text length. If Colorizer is present, reset the color too.
  void
  TerminalDisplay::Detach() {
    fWritePos = Pos();
    fWriteEnd = Pos();
    if (GetContext()->GetColorizer()) {
      Color DefaultColor;
      GetContext()->GetColorizer()->GetColor(0, DefaultColor);
      SetColor(0, DefaultColor);
      // We can't tell whether the application will activate a different color:
      fPrevColor = -1;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Width in terminal columns of character idx of the concatenation of the
  /// prompt, the editor prompt and the input line.
  size_t
  TerminalDisplay::WidthOfDisplayChar(size_t idx) const {
    const Text& Prompt = GetContext()->GetPrompt();
    if (idx < Prompt.length()) return Prompt.GetWidthOfChar(idx);
    idx -= Prompt.length();

    const Text& EditPrompt = GetContext()->GetEditor()->GetEditorPrompt();
    if (idx < EditPrompt.length()) return EditPrompt.GetWidthOfChar(idx);
    idx -= EditPrompt.length();

    const Text& Line = GetContext()->GetLine();
    if (idx < Line.length()) {
      // Masked input is echoed as '*', whatever was actually typed.
      if (GetContext()->GetTextInput()->IsInputMasked()) return 1;
      return Line.GetWidthOfChar(idx);
    }
    return 1; // past the end of the text: the cursor itself
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Where the idx'th character of the displayed text ends up on the terminal.
  Display::Pos
  TerminalDisplay::IndexToPos(size_t idx) const {
    Pos P;
    for (size_t i = 0; i < idx; ++i) {
      AdvancePos(P, WidthOfDisplayChar(i));
    }
    return P;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Write n spaces.
  void
  TerminalDisplay::WriteBlanks(size_t n) {
    if (!n || !IsTTY()) return;
    const std::string Blanks(n, ' ');
    WriteRawString(Blanks.c_str(), Blanks.length());
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Write out wrapped text to the display. Used in WriteWrapped and DisplayInfo
  ///
  /// Writing starts at fWritePos, which the caller must have moved to the right
  /// place; the position is advanced by the width of what is written, so that
  /// it stays in step with where the terminal's cursor actually is.
  ///
  /// \param[in] text text to write out
  /// \param[in] TextOffset where to begin writing out text from
  /// \param[in] NumRequested number of text characters requested for output
  size_t
  TerminalDisplay::WriteWrappedTextPart(const Text &text, size_t TextOffset,
                                        size_t NumRequested) {
    size_t Start = TextOffset;
    size_t NumRemaining = NumRequested; // optimistic

    size_t NumAvailable = text.length() - Start;
    if (NumRequested == (size_t) -1) { // requested max available
      NumRequested = NumAvailable;
    }

    // If we have some text available for output
    if (NumAvailable > 0) {
      // If we don't have enough to output NumRemaining, output only what's available
      if (NumAvailable < NumRemaining) {
        NumRemaining = NumAvailable;
      }

      while (NumRemaining > 0) {
        // How many columns can this line still hold?
        size_t numToEOL = GetWidth() - fWritePos.fCol;

        // How many characters fit into them? Not the same number, as soon as
        // the text contains anything but plain single-width characters.
        size_t numThisLine = 0;
        size_t colsThisLine = 0;
        while (numThisLine < NumRemaining) {
          size_t w = text.GetWidthOfChar(Start + numThisLine);
          if (colsThisLine + w > numToEOL) break;
          colsThisLine += w;
          ++numThisLine;
        }

        if (numThisLine == 0) {
          if (fWritePos.fCol == 0) {
            // The character is wider than the whole terminal. Nothing is
            // gained by wrapping again, and looping would never end, so write
            // it and let the terminal cope.
            numThisLine = 1;
            colsThisLine = numToEOL;
          } else {
            // A double-width character does not fit into what is left of this
            // line. Blank those columns out and put the character at the start
            // of the next line - splitting it across the margin would leave
            // the terminal and us disagreeing about where the cursor is.
            WriteBlanks(numToEOL);
            ActOnEOL();
            fWritePos.fCol = 0;
            ++fWritePos.fLine;
            continue;
          }
        }

        // If there is a Colorizer, we only write same-colored chunks.
        // How long is the current chunk? Adjust numThisLine.
        if (GetContext()->GetColorizer()) {
          const std::vector<char>& Colors = text.GetColors();
          char ThisColor = Colors[Start];
          size_t numSameColor = 1;
          while (numSameColor < numThisLine
                 && ThisColor == Colors[Start + numSameColor])
            ++numSameColor;
          if (numSameColor < numThisLine) {
            numThisLine = numSameColor;
            colsThisLine = text.GetWidth(Start, Start + numThisLine);
          }

          if (ThisColor != fPrevColor) {
            Color C;
            GetContext()->GetColorizer()->GetColor(ThisColor, C);
            SetColor(ThisColor, C);
            fPrevColor = ThisColor;
          }
        }

        // Write out the characters and update the write position. The terminal
        // wants bytes, so translate the character range into a byte range.
        const size_t ByteStart = text.GetByteOffset(Start);
        const size_t ByteEnd = text.GetByteOffset(Start + numThisLine);
        WriteRawString(text.GetText().c_str() + ByteStart, ByteEnd - ByteStart);
        fWritePos.fCol += colsThisLine;
        if (fWritePos.fCol >= GetWidth()) { // If we hit EOL, wrap around
          ActOnEOL();
          fWritePos.fCol = 0;
          ++fWritePos.fLine;
        }

        Start += numThisLine;
        NumRemaining -= numThisLine;
      }
    }

    // If we have processed the characters we have requested
    if (NumRequested == NumAvailable) {
      const size_t NumWroteLines = fWritePos.fLine;
      const size_t NumPrevLines = fWriteEnd.fLine;
      if (fWritePos < fWriteEnd) {
        // If we wrote less than previously,
        // erase the rest of the current line
        EraseToRight();
      }
      if (NumWroteLines < NumPrevLines) {
        // If we wrote less lines than previously,
        // erase the surplus previous lines
        Pos prevWC = GetCursor();
        MoveFront();
        fWritePos.fCol = 0;
        for (size_t l = NumWroteLines + 1; l <= NumPrevLines; ++l) {
          MoveDown();
          ++fWritePos.fLine;
          EraseToRight();
        }
        Move(prevWC);
      }
    }
    return NumRemaining;
  }

  size_t
  TerminalDisplay::WriteWrapped(Range::EPromptUpdate PromptUpdate, bool masked,
                                size_t Offset, size_t Requested /* = -1*/) {
    Attach();

    const Text& Prompt = GetContext()->GetPrompt();
    size_t PromptLen = GetContext()->GetPrompt().length();
    const Text& EditPrompt = GetContext()->GetEditor()->GetEditorPrompt();
    size_t EditorPromptLen = EditPrompt.length();

    if (!IsTTY()) {
       PromptLen = 0;
       EditorPromptLen = 0;
       PromptUpdate = Range::kNoPromptUpdate;
    }

    // If updating prompt, write the main prompt first (e.g. [cling]$)
    if (PromptUpdate & Range::kUpdatePrompt) {
      // Writing from front means we write the prompt, too
      Move(Pos());
      WriteWrappedTextPart(Prompt, 0, PromptLen);
    }
    // If updating any prompt
    if (PromptUpdate != Range::kNoPromptUpdate) {
      // Any prompt update means we'll have to re-write the editor prompt
      Move(IndexToPos(PromptLen));
      if (EditorPromptLen) {
        WriteWrappedTextPart(EditPrompt, 0, EditorPromptLen);
      }
      // Any prompt update means we'll have to re-write the text
      Offset = 0;
      Requested = (size_t) -1;
    }
    Move(IndexToPos(PromptLen + EditorPromptLen + Offset));

    size_t avail = 0;
    if (masked) {
      Text mask(std::u32string(GetContext()->GetLine().length(), U'*'), 0);
      avail = WriteWrappedTextPart(mask, Offset, Requested);
    } else {
      avail = WriteWrappedTextPart(GetContext()->GetLine(), Offset, Requested);
    }
    fWriteEnd = IndexToPos(PromptLen + EditorPromptLen
                           + GetContext()->GetLine().length());
    return avail;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Move the cursor to the required position.
  ///
  /// \param[in] p position to move to
  void
  TerminalDisplay::Move(Pos p) {
    Attach();
    if (fWritePos == p) return;
    if (fWritePos.fLine > p.fLine) {
      MoveUp(fWritePos.fLine - p.fLine);
      fWritePos.fLine -= fWritePos.fLine - p.fLine;
    } else if (fWritePos.fLine < p.fLine) {
      MoveDown(p.fLine - fWritePos.fLine);
      fWritePos.fLine += p.fLine - fWritePos.fLine;
    }

    if (p.fCol == 0) {
      MoveFront();
      fWritePos.fCol = 0;
    } else if (fWritePos.fCol > p.fCol) {
      MoveLeft(fWritePos.fCol - p.fCol);
      fWritePos.fCol -= fWritePos.fCol - p.fCol;
    } else if (p.fCol > fWritePos.fCol) {
      MoveRight(p.fCol - fWritePos.fCol);
      fWritePos.fCol += p.fCol - fWritePos.fCol;
    }
  }
}
