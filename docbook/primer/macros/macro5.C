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
