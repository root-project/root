# Histograms #

Histograms play a fundamental role in any type of physics analysis, not
only to visualise measurements but being a powerful form of data
reduction. ROOT offers many classes that represent histograms, all
inheriting from the `TH1` class. We will focus in this chapter on uni-
and bi- dimensional histograms whose bin-contents are represented by
floating point numbers [^3], the `TH1F` and `TH2F` classes respectively.

## Your First Histogram ##

Let's suppose you want to measure the counts of a Geiger detector put in
proximity of a radioactive source in a given time interval. This would
give you an idea of the activity of your source. The count distribution
in this case is a Poisson distribution. Let's see how operatively you
can fill and draw a histogram with the following example macro.

``` {.cpp .numberLines}
 // Create, Fill and draw an Histogram which reproduces the
 // counts of a scaler linked to a Geiger counter.

 void macro5(){
     TH1F* cnt_r_h=new TH1F("count_rate",
                 "Count Rate;N_{Counts};# occurencies",
                 100, // Number of Bins
                 -0.5, // Lower X Boundary
                 15.5); // Upper X Boundary

     const float mean_count=3.6;
     TRandom3 rndgen;
     // simulate the measurements
     for (int imeas=0;imeas<400;imeas++)
         cnt_r_h->Fill(rndgen.Poisson(mean_count));

     TCanvas* c= new TCanvas();
     cnt_r_h->Draw();

     TCanvas* c_norm= new TCanvas();
     cnt_r_h->DrawNormalized();

     // Print summary
     cout << "Moments of Distribution:\n"
          << " - Mean = " << cnt_r_h->GetMean() << " +- "
                          << cnt_r_h->GetMeanError() << "\n"
          << " - RMS = " << cnt_r_h->GetRMS() << " +- "
                         << cnt_r_h->GetRMSError() << "\n"
          << " - Skewness = " << cnt_r_h->GetSkewness() << "\n"
          << " - Kurtosis = " << cnt_r_h->GetKurtosis() << "\n";
 }
```

Which gives you the following plot (Figure [5.1](#f51)):

[f51]: figures/poisson.png "f51"
<a name="f51"></a>

![The result of a counting (pseudo) experiment. Only bins corresponding
to integer values are filled given the discrete nature of the poissonian
distribution. \label{f51}][f51]

Using histograms is rather simple. The main differences with respect to
graphs that emerge from the example are:

-   line *5*: The histograms have a name and a title right from the
    start, no predefined number of entries but a number of bins and a
    lower-upper range.

-   line *15*: An entry is stored in the histogram through the
    `TH1F::Fill` method.

-   line *19* and *22*: The histogram can be drawn also normalised, ROOT
    automatically takes cares of the necessary rescaling.

-   line *25* to *31*: This small snippet shows how easy it is to access
    the moments and associated errors of a histogram.

## Add and Divide Histograms ##

Quite a large number of operations can be carried out with histograms.
The most useful are addition and division. In the following macro we
will learn how to manage these procedures within ROOT.

``` {.cpp .numberLines}
 // Divide and add 1D Histograms

 void format_h(TH1F* h, int linecolor){
     h->SetLineWidth(3);
     h->SetLineColor(linecolor);
 }

 void macro6(){

     TH1F* sig_h=new TH1F("sig_h","Signal Histo",50,0,10);
     TH1F* gaus_h1=new TH1F("gaus_h1","Gauss Histo 1",30,0,10);
     TH1F* gaus_h2=new TH1F("gaus_h2","Gauss Histo 2",30,0,10);
     TH1F* bkg_h=new TH1F("exp_h","Exponential Histo",50,0,10);

     // simulate the measurements
     TRandom3 rndgen;
     for (int imeas=0;imeas<4000;imeas++){
         exp_h->Fill(rndgen.Exp(4));
         if (imeas%4==0) gaus_h1->Fill(rndgen.Gaus(5,2));
         if (imeas%4==0) gaus_h2->Fill(rndgen.Gaus(5,2));
         if (imeas%10==0)sig_h->Fill(rndgen.Gaus(5,.5));}

     // Format Histograms
     TH1F* histos[4]={sig_h,bkg_h,gaus_h1,gaus_h2};
     for (int i=0;i<4;++i){
         histos[i]->Sumw2(); // *Very* Important
         format_h(histos[i],i+1);
         }

     // Sum
     TH1F* sum_h= new TH1F(*bkg_h);
     sum_h->Add(sig_h,1.);
     sum_h->SetTitle("Exponential + Gaussian");
     format_h(sum_h,kBlue);

     TCanvas* c_sum= new TCanvas();
     sum_h->Draw("hist");
     bkg_h->Draw("SameHist");
     sig_h->Draw("SameHist");

     // Divide
     TH1F* dividend=new TH1F(*gaus_h1);
     dividend->Divide(gaus_h2);

     // Graphical Maquillage
     dividend->SetTitle(";X axis;Gaus Histo 1 / Gaus Histo 2");
     format_h(dividend,kOrange);
     gaus_h1->SetTitle(";;Gaus Histo 1 and Gaus Histo 2");
     gStyle->SetOptStat(0);

     TCanvas* c_divide= new TCanvas();
     c_divide->Divide(1,2,0,0);
     c_divide->cd(1);
     c_divide->GetPad(1)->SetRightMargin(.01);
     gaus_h1->DrawNormalized("Hist");
     gaus_h2->DrawNormalized("HistSame");

     c_divide->cd(2);
     dividend->GetYaxis()->SetRangeUser(0,2.49);
     c_divide->GetPad(2)->SetGridy();
     c_divide->GetPad(2)->SetRightMargin(.01);
     dividend->Draw();
 }
```

The plots that you will obtain are shown in Figures [5.2](#f52) and [5.3](#f53).

[f52]: figures/histo_sum.png "f52"
<a name="f52"></a>

![The sum of two histograms.\label{f52}][f52]

[f53]: figures/histo_ratio.png "f53"
<a name="f53"></a>

![The ratio of two histograms.\label{f53}][f53]

Some lines now need a bit of clarification:

-   line *3*: Cint, as we know, is also able to interpret more than one
    function per file. In this case the function simply sets up some
    parameters to conveniently set the line of histograms.

-   line *20* to *22*: Some contracted C++ syntax for conditional
    statements is used to fill the histograms with different numbers of
    entries inside the loop.

-   line *27*: This is a crucial step for the sum and ratio of
    histograms to handle errors properly. The method `TH1::Sumw2` makes
    sure that the squares of weights are stored inside the histogram
    (equivalent to the number of entries per bin if weights of 1 are
    used). This information is needed to correctly calculate the errors
    of each bin entry when the methods `TH1::Add` and `TH1::Divide` are
    invoked.

-   line *33*: The sum of two histograms. A weight can be assigned to
    the added histogram, for example to comfortably switch to
    subtraction.

-   line *44*: The division of two histograms is rather straightforward.

-   line *53* to *63*: When you draw two quantities and their ratios, it
    is much better if all the information is condensed in one single
    plot. These lines provide a skeleton to perform this operation.

## Two-dimensional Histograms ##

Two-dimensional histograms are a very useful tool, for example to
inspect correlations between variables. You can exploit the
bi-dimensional histogram classes provided by ROOT in a very simple way.
Let's see how in the following macro:

``` {.cpp}
// Draw a Bidimensional Histogram in many ways
// together with its profiles and projections

void macro7(){
    gStyle->SetPalette(53);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TH2F bidi_h("bidi_h","2D Histo;Guassian Vals;Exp. Vals",
                30,-5,5,  // X axis
                30,0,10); // Y axis

    TRandom3 rgen;
    for (int i=0;i<500000;i++)
        bidi_h.Fill(rgen.Gaus(0,2),10-rgen.Exp(4),.1);

    TCanvas* c=new TCanvas("Canvas","Canvas",800,800);
    c->Divide(2,2);
    c->cd(1);bidi_h.DrawClone("Cont1");
    c->cd(2);bidi_h.DrawClone("Colz");
    c->cd(3);bidi_h.DrawClone("lego2");
    c->cd(4);bidi_h.DrawClone("surf3");

    // Profiles and Projections
    TCanvas* c2=new TCanvas("Canvas2","Canvas2",800,800);
    c2->Divide(2,2);
    c2->cd(1);bidi_h.ProjectionX()->DrawClone();
    c2->cd(2);bidi_h.ProjectionY()->DrawClone();
    c2->cd(3);bidi_h.ProfileX()->DrawClone();
    c2->cd(4);bidi_h.ProfileY()->DrawClone();
}
```

Two kinds of plots are provided within the code, the first one
containing three-dimensional representations (Figure [5.4](#f54)) and the second one
projections and profiles (Figure [5.5](#f55)) of the bi-dimensional histogram.

[f54]: figures/th2f.png "f54"
<a name="f54"></a>

![Different ways of representing bi-dimensional
histograms.\label{f54}][f54]

[f55]: figures/proj_and_prof.png "f55"
<a name="f55"></a>

![The projections and profiles of bi-dimensional
histograms.\label{f55}][f55]

When a projection is performed along the x (y) direction, for every bin
along the x (y) axis, all bin contents along the y (x) axis are summed
up (upper the plots of Figure [5.5](#f55)). When a profile is performed along the x (y)
direction, for every bin along the x (y) axis, the average of all the
bin contents along the y (x) is calculated together with their RMS and
displayed as a symbol with error bar (lower two plots of Figure [5.5](#f55)).

Correlations between the variables are quantified by the methods
`Double_t GetCovariance()` and `Double_t GetCorrelationFactor()`.

[^3]: To optimise the memory usage you might go for one byte (TH1C), short (TH1S), integer (TH1I) or double-precision (TH1D) bin-content.
