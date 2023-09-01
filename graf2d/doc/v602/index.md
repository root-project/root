## 2D Graphics Libraries

### TLegend

-  Due to the way the vertical text centring is done (bounding based) the
   spacing between lines may appeared irregular in some cases. This was
   previously fixed. But in case the text size of legend's items is smaller
   than the line spacing some misalignment appeared.

### TLatex

-  The interface to TMathText did not work when the size was set in pixel
   (precision 3).

### TTeXDump

- The marker definition is now inside the picture definition. Being outside
  produced some side effect on the picture positioning. (cf Jira Report 6470).

### Typographically correct minus sign

- Negative values as well as negative exponents were typeset with a hyphen
  instead of a real minus sign in axis labels and statistics numbers. Now is the
  TLatex `#minus` sign is used instead, which improve the appearance of the plots
  and make them even better for publications.