## 2D Graphics Libraries

### TText

- The character position was not correct with the Cocoa backend.
  (see https://sft.its.cern.ch/jira/browse/ROOT-6561)

### TTexDump

- The hollow fill style was not rendered correctly.
  (see https://sft.its.cern.ch/jira/browse/ROOT-6841)
- Better line width matching with screen and pdf output.
- Text color was ignored. It was always black.

### TPad

- In `TPad::ShowGuidelines` the number of guide lines is limited to 15. Above
  that they become useless.

### TLatex

- With the Cocoa backend the PDF and PS output produced miss-aligned exponents
  because the `GetTextExtend` method behaved differently in batch mode and "screen"
  mode. This is now fixed. See http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=18883
