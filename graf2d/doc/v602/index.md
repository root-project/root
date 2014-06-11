## 2D Graphics Libraries

### TLegend

-  Due to the way the vertical text centring is done (bounding based) the
   spacing between lines may appeared irregular in some cases. This was
   previously fixed. But in case the text size of legend's items is smaller
   than the line spacing some misalignment appeared.

### TLatex

-  The interface to TMathText did not work when the size was set in pixel 
   (precision 3). 
