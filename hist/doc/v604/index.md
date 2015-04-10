
## Histogram Libraries

### TFormula

-  New version of the TFormula class based on Cling. Formula expressions are now used to create  functions which are passed to Cling to be Just In Time compiled.
The expression is therefore compiled using Clang/LLVVM which will give execution time as compiled code and in addition correctness of the result obtained.
-  This class is not 100% backward compatible with the old TFormula class, which is still available in ROOT as =ROOT::v5::TFormula=.
    Some of the TFormula member funtions available in version 5, such as =Analyze= and =AnalyzeFunction= are not available in the new TFormula class. 
    On the other hand formula expressions which were valid in version 5 are still valid in TFormula version 6
-  TFormula is not anymore a base class for TF1. 	

### TF1

- Change of its inheritance structure. =TF1= has not anymore =TFormula= as a base class, so this code 


    ``` {.cpp}
    TF1 * f1 = new TF1("f1","f1","sin(x)",0,10);
	TFormula * formula = (TFormula *) f1;
	```

**it is not valid anymore.**
The equivalent correct code is now 
  ``` {.cpp}
    TF1 * f1 = new TF1("f1","f1","sin(x)",0,10);
	TFormula * formula = f1->GetFormula(); 
	```


### TGraph2DPainter

-   In some case and extra point was drawn in the center od the plot when a
    `TGRaph2d`was drawn with `P`, `P0`, or `PCOL` options.

### THistPainter

- It was possible to interactively zoom outside the histograms' limits. Protections
  have been added.
- When an histogram was drawn with the option `E0` and log scale along the Y axis,
  some additional markers were drawn at the bottom line of the plot. This was
  reported <a href="http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=18778">here</a>.
- Implement the option `0` combined with the option `COL` as requested
  <a href="https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=19046">here</a>.
  When the minimum of the histogram is set to a greater value than the real minimum,
  the bins having a value between the real minimum and the new minimum are not drawn
  unless the option <tt>0</tt> is set.

    Example:

    ``` {.cpp}
    {
       TCanvas *c1 = new TCanvas("c1","c1",600,600);
       c1->Divide(1,2);
       TH2F *hcol21 = new TH2F("hcol21","Option COLZ",40,-4,4,40,-20,20);
       TH2F *hcol22 = new TH2F("hcol22","Option COLZ0",40,-4,4,40,-20,20);
       Float_t px, py;
       for (Int_t i = 0; i < 25000; i++) {
          gRandom->Rannor(px,py);
          hcol21->Fill(px,5*py);
          hcol22->Fill(px,5*py);
       }
       hcol21->SetBit(TH1::kNoStats);
       hcol22->SetBit(TH1::kNoStats);
       gStyle->SetPalette(1);
       c1->cd(1); hcol21->Draw("COLZ");
       c1->cd(2); hcol22->Draw("COLZ0");
       hcol22->SetMaximum(100);
       hcol22->SetMinimum(40);
       return c1;
}
    ```
    ![COLZ0 plot example](colzo.png "COLZ0 plot example")
- The parameter `gStyle->SetHistTopMargin()` was ignored when plotting a 2D histogram
  using the option `E`. This can be seen plotting the histogram with `"LEGO E"`.
- GetObjectInfo did not display the right value form "Sum" for histograms plotted
  with option SAME on top of an histogram displayed with a subrange. This was
  reported in ROOT-7124.

### TGraph2D

- Change `GetHistogram()` in order to be able to access the returned histogram
  before the 2D graph has been filled with points. That was not possible previously.
  This problem was reported
 <a href="https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=19186">here</a>.
- When a `TGraph2D` was entirely in the plane `Z=0` the 3D could not be defined.
  This problem was reported
 <a href="https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=19517>here</a>.

### THStack

- Implement `GetNhists()` to return the number of histograms in the stack.
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
