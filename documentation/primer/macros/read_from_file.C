void read_from_file(){

    // Let's open the TFile
    TFile* in_file= new TFile("my_rootfile.root");

    // Get the Histogram out
    TH1F* h = (TH1F*) in_file.GetObjectChecked("my_histogram",
                                               "TH1F");

    // Draw it
    h->Draw();
}
