## Tree Libraries

### TTreePlayer and TSelectorDraw

-   The option `colz` in a command like `nt->Draw("b:a:c>>h", "", "colz");`
    erased the histogram `h`. (Jira report ROOT-4508).
-   Make sure the number of bins for 2D histograms generated when drawing
    3 variables with option COL is the same as drawing 2 variables.

