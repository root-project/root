
## Histogram Libraries

### TGraph2DPainter

-   In some case and extra point was drawn in the center od the plot when a
    `TGRaph2d`was drawn with `P`, `P0`, or `PCOL` options.

### THistPainter

- It was possible to interactively zoom outside the histograms' limits. Protections
  have been added.
- When an histogram was drawn with the option `E0` and log scale along the Y axis,
  some additional markers were drawn at the bottom line of the plot. This was
  reported <a href="http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=18778">here</a>.

### THStack

- New drawing option `NOSTACKB`. histograms are drawn next to each other as
  bar charts.

    Example:

    ``` {.cpp}
    TCanvas* nostackb() {
       TCanvas *cst0 = new TCanvas("cst0","cst0",600,400);
       THStack *hs = new THStack("hs","Stacked 1D histograms: option #font[82]{\"nostackb\"}");

       TH1F *h1 = new TH1F("h1","h1",10,-4,4);
       h1->FillRandom("gaus",20000);
       h1->SetFillColor(kRed);
       hs->Add(h1);

       TH1F *h2 = new TH1F("h2","h2",10,-4,4);
       h2->FillRandom("gaus",15000);
       h2->SetFillColor(kBlue);
       hs->Add(h2);

       TH1F *h3 = new TH1F("h3","h3",10,-4,4);
       h3->FillRandom("gaus",10000);
       h3->SetFillColor(kGreen);
       hs->Add(h3);

       hs->Draw("nostackb");
       return cst0;
    }

    ```
    ![NOSTACKB plot example](nostackb.png "NOSTACKB plot example")
