//example of ROOT script creating/reading a simple ntuple
//The ntuple has 50 columns of ints, 50 columns of floats and
//one column with a string.
//The script may be executed with a different number of rows.
//It shows the time to write & read in compress and no compress mode.
//The script can be executed using interpreted (CINT) or compiled code
//Example of a ROOT session
// root > .x julius.C  (script executed via CINT)
//or
// root > .x julius.C+ (script executed via the compiler)
//
//By default, the script runs with ndecades = 4 (100,1000,10000,100000 rows)
//  tree write no compress
//  tree read  no compress
//  tree write compress
//  tree read  compress
// one may call the script with ndecades=5 to process the case with one million rowsone can specify the number of rows in the ntuple via, eg
// root > .x julius.C(5)
//By default the reader reads only 50% of the rows. This may be changed
//by changing the default value of fraction in the function jread.

#include "TFile.h"
#include "TStopwatch.h"
#include "TTree.h"
#include "TRandom.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TLegend.h"

Double_t jwrite(int nrows,int compress) {
        const int nints = 50;
        const int nfloats = 50;
        int integers[nints];
        float floats[nfloats];
        char astring[12];

        TStopwatch timer;
        timer.Start();

        //build the Tree and its branches
        TFile f("julius.root","recreate","test julius",compress);
        TTree *T = new TTree("T","julius test");
        int i,j;
        T->Branch("string",astring,"string/C");
        for (i=0;i<nints;i++) {
                T->Branch(Form("int%d",i),&integers[i],Form("int%d/I",i));
        }
        for (i=0;i<nfloats;i++) {
                T->Branch(Form("float%d",i),&floats[i],Form("float%d/F",i));
        }

        //fill the tree
        for (i=0;i<nrows;i++) {
                sprintf(astring,"s%d",i);
                for (j=0;j<nints;j++) integers[j] = 100*i+j;
                for (j=0;j<nfloats;j++) floats[j] = 100*i+j;
                T->Fill();
        }

        //save/print the Tree
        T->AutoSave();
        //T->Print();
        timer.Stop();
        Double_t cpu = timer.CpuTime();
        printf("Write: compress=  %d, nrows=%d  : RT=%7.3f s, Cpu=%7.3f s, filesize = %ld bytes\n",compress,nrows,timer.RealTime(),cpu,(long) f.GetEND());
        return cpu;
}
Double_t jread(double fraction =0.5) {
        const int nints = 50;
        const int nfloats = 50;
        int integers[nints];
        float floats[nfloats];
        char astring[12];

        TStopwatch timer;
        timer.Start();

        //read the Tree and its branches
        TFile f("julius.root");
        TTree *T = (TTree*)f.Get("T");
        int i,j;
        int nentries = (int)T->GetEntries();
        //set the branch addresses
        T->SetBranchAddress("string",astring);
        for (j=0;j<nints;j++)   T->SetBranchAddress(Form("int%d",j),&integers[j]);
        for (j=0;j<nfloats;j++) T->SetBranchAddress(Form("float%d",j),&floats[j]);

        //read the Tree (only fraction of all entries)
        TRandom r;
        for (i=0;i<nentries;i++) {
                if (r.Rndm() < fraction) T->GetEntry(i);
        }
        timer.Stop();
        Double_t cpu = timer.CpuTime();
    printf("Read:  fraction=%g, nrows=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fraction,nentries,timer.RealTime(),cpu);
        return cpu;
}
void julius(int ndecades=4) {
        Int_t nrows = 100;
        TGraph *gw0 = new TGraph(ndecades); gw0->SetMarkerStyle(24); gw0->SetMarkerColor(4);
        TGraph *gw1 = new TGraph(ndecades); gw1->SetMarkerStyle(20); gw1->SetMarkerColor(4);
        TGraph *gr0 = new TGraph(ndecades); gr0->SetMarkerStyle(25); gr0->SetMarkerColor(2);
        TGraph *gr1 = new TGraph(ndecades); gr1->SetMarkerStyle(21); gr1->SetMarkerColor(2);
        for (Int_t i=0;i<ndecades;i++) {
                gw0->SetPoint(i,nrows,jwrite(nrows,0));
                gr0->SetPoint(i,nrows,jread());
                gw1->SetPoint(i,nrows,jwrite(nrows,1));
                gr1->SetPoint(i,nrows,jread());
                nrows *= 10;
        }
        TMultiGraph *mg = new TMultiGraph();
        mg->Add(gw0); mg->Add(gr0); mg->Add(gw1); mg->Add(gr1);
        TCanvas *c1 = new TCanvas("c1","Julius benchmark",10,10,700,600);
        c1->SetGrid();
        c1->SetLogx();
        c1->SetLogy();
        mg->Draw("alp");
        c1->Update();
        TLegend *legend = new TLegend(.13,.7,.5,.88);
        legend->AddEntry(gw1,"write compress=1","lp");
        legend->AddEntry(gw0,"write compress=0","lp");
        legend->AddEntry(gr1,"read compress=1","lp");
        legend->AddEntry(gr0,"read compress=0","lp");
        legend->Draw();
        TAxis *xaxis = mg->GetXaxis();
        TAxis *yaxis = mg->GetYaxis();
        yaxis->SetNoExponent(kTRUE);
        xaxis->SetTitle("Number of rows");
        yaxis->SetTitle("time in seconds");
        c1->Modified();
}
