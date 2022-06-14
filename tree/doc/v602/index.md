## Tree Libraries

### TTreePlayer and TSelectorDraw

-   The option `colz` in a command like `nt->Draw("b:a:c>>h", "", "colz");`
    erased the histogram `h`. (Jira report ROOT-4508).
-   Make sure the number of bins for 2D histograms generated when drawing
    3 variables with option COL is the same as drawing 2 variables.
-   In case of a 2D scatter plot drawing (with or without option COL) the automatically
    computed lower limits of the histogram's axis might be 0. In that case it is better to set them
    to the minimum of the data set (if it is >0) to avoid data cut when plotting in log scale.

