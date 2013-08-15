## Histogram Libraries

### THistPainter

-   With option COL `TProfile2D` histograms are handled differently
    because, for this type of 2D histograms, it is possible to know if
    an empty bin has been filled or not. So even if all the bins'
    contents are positive some empty bins might be painted. And vice
    versa, if some bins have a negative content some empty bins might be
    not painted.
-   With option COLZ the axis attributes were not imported correctly on
    the palette axis.
-   Make sure the 2D drawing options COL, ARR, CONT and BOX are ignored
    when used to plot a 1D histogram. 1D histograms plotted with one of
    these options is now drawn with the default 1D plotting option. This
    is useful when the default option in the TBrowser is, for instance,
    COL. Before this change the 1D histogram appeared as blank.
-   New plotting option `"LEGO3"`. Like the option `"LEGO1"`, the
    option `"LEGO3"` draws a lego plot using the hidden surface removal
    technique but doesn't draw the border lines of each individual
    lego-bar. This is very useful for histograms having many bins. With
    such histograms the option `"LEGO1"` gives a black image because of
    the border lines. This option also works with stacked legos.
-   New plotting option `"LEGO4"`. Draw a lego plot with hidden surface
    removal, like LEGO1, but without the shadow effect on each lego-bar.
-   Line attributes can be used in lego plots to change the edges'
    style. It works when drawing a `TH2` in LEGO or SURF mode whatever
    the coordinate system used (car, pol, cyl, sph, and psr). It also
    handles `THStack` (lego only).
-   Implement in THistPainter::GetObjectInfo the case of TProfile and
    TProfile2D to print the tooltip information on each bin. Disable the
    printing of the bin information for TH3, since it is not currently
    possible to retrieve the 3d bin number from the pixel coordinate.
-   Fit parameters with very long name destroyed the stats display. This
    is now fixed. \
    Example:

    ``` {.cpp}
    {
       gStyle->SetOptFit(111);
       TH1F *hist = new TH1F("hist","hist",100,-5,5);
       TF1 *fit = new TF1("fit","gaus",-5,5);
       fit->SetParName(2,"Parameter with very very very very long name");
       hist->FillRandom("gaus",5000);
       hist->Draw();
       hist->Fit(fit);
    }
    ```
-   The statistics display has a new option: "I"=2 (the default one
    remains "i"=1). The value displayed for integral is
    `TH1::Integral("width")` instead of `TH1::Integral()`.
    Example:

    ``` {.cpp}
    {
       TH1D * histo1D = new TH1D ("histo1D","",2,0.,4.) ; 
       histo1D->SetBinContent( 1,1.) ;
       histo1D->SetBinContent( 2,2.) ;
       TCanvas * canvas = new TCanvas () ;
       canvas->Divide(2,1) ;
       canvas->cd(1) ; gStyle->SetOptStat("nemruoi") ; histo1D->DrawClone() ;
       canvas->cd(2) ; gStyle->SetOptStat("nemruoI") ; histo1D->DrawClone() ;
    }
    ```
-   `TH1` was drawn improperly in "Logx" mode if "X" axis starts at 
    negative values. The following macro illustrades this problem.
    ``` {.cpp}
    {
       TCanvas *c1 = new TCanvas("c1", "c1",0,0,1200,700);
       int n = 100;
       Float_t d = 0.5;
       TH1F *h1 = new TH1F("h1", "x_min = - d", n, -d, 100-d);
       h1->Fill(1, 1); h1->Fill(3, 3); h1->Fill(5, 5); h1->Fill(7, 7);
 
       TH1F *h2 = new TH1F("h2", "x_min = +d", n, d, 100+d);
       h2->Fill(1, 1); h2->Fill(3, 3); h2->Fill(5, 5); h2->Fill(7, 7);
 
       c1->Divide(1, 2);
       c1->cd(1); gPad->SetLogx(); h1->Draw(); // upper picture
       c1->cd(2); gPad->SetLogx(); h2->Draw(); // lower picture
       h1->GetXaxis()->SetMoreLogLabels();
       h2->GetXaxis()->SetMoreLogLabels();
       c1_1->SetGridx();
       c1_2->SetGridx();
    }
    ```
-   In `PaintStat2` the temporary string used to paint the fit parameters 
    was too small and in some cases the errors where truncated. The size 
    of the string is now the same as in `PaintStat`.


### TGraphPainter

-   Fix the problem described [here](http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=8591).
    When drawn with option SAME the histogram 1st and last bins might
    be wrong. The following macro shows the problem:

    ``` {.cpp}
    {
       TCanvas *c = new TCanvas("c","c",900,900);
       c->Divide (1,2);
           
       TH1D * histo1 = new TH1D ("histo1","histo1",100,0.,100.) ;
       histo1->SetBinContent(51,80.) ;
           
       TH1D * histo2 = new TH1D ("histo2","histo2",100,49.9,51.1) ;  /// not ok
       histo2->SetMinimum(0.) ; histo2->SetMaximum(100.) ;
           
       c->cd(1); gPad->DrawFrame(49.9, 0., 51.1, 100);
       histo1->Draw("same");
           
       Double_t xAxis[4] = {3., 5., 7., 9.};
       TH1D *histo2 = new TH1D("histo","",3, xAxis);
       histo2->SetBinContent(1,2.);
       histo2->SetBinContent(2,4.);
       histo2->SetBinContent(3,3.);
           
       c->cd(2); gPad->DrawFrame(4.,0., 10.,5.);
       histo2->Draw("same");
    }
    ```
-   In `TGraph2DPainter::PaintLevels` the colour levels used to paint
    the triangles did not match the minimum and maximum set by the 
    user on the `TGraph2D`. This problem was reported 
    [here](http://root.cern.ch/phpBB3/viewtopic.php?f=3&t=16937&p=72314#p72314)

### TPaletteAxis

-   The histogram Z axis title is now painted along the palette axis.

### TH2, TH3

-   Update Projection methods of both TH2 and TH3 to not return a null
    pointer when an histogram with the same name already existed and it
    was not compatible. Now just set the new correct binning on the
    previously existing histogram.

### TH1

-   The following code should produce a plot. It did not.

    ``` {.cpp}
       TH1F* h=new TH1F("hist", "histogram", 10, 0, 3); 
       h->FillRandom("gaus"); 
       h->Draw("same"); 
    ```

### TGraph

-   `TGraph::Draw()` needed at least the option `AL` to draw the graph
     axis even when there was no active canvas or when the active canvas
     did not have any axis defined. This was counter-intuitive. Now if
     `TGraph::Draw()` is invoked without parameter and if there is no
     axis defined in the current canvas, the option `ALP` is automatically
     set.

### TGraph2D

-   When `GetX(YZ)axis` were called on a `TGraph2D`, the frame limit and
    plotting options were changed.
-   Modify the `Clear` function in order to be able to reuse a
    `TGraph2D` after a `Clear` is performed.
-   In `GetHistogram()` the lower and higher axis limits are always
    different.
-   Protection added to avoid a Seg Fault on `.q` when `SetHistogram()` 
    is called on a `TGraph2D`.

### TF1

-   Implement the possibility to save a `TF1` as C code indenpant from
    ROOT. It is enough to save the function as a ".cc" file. \
    Example:

    ``` {.cpp}
       root [0] TF1 *f1 = new TF1("f1","x*x",-10,10)
       root [1] f1->SaveAs("f1.cc");
       Info in <TF1::SaveAs>: cc file: f1.cc has been generated
          root [2] .x f1.cc(9.)
          (double)8.10019368181367980e+01
    ```

