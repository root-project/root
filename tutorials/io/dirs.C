/// \file
/// \ingroup tutorial_io
/// \notebook -nodraw
/// This macro illustrates how to create a hierarchy of directories
/// in a Root file.
/// Ten directories called plane0, plane1, ..., plane9 are created.
/// Each plane directory contains 200 histograms.
/// Note that the macro deletes the TFile object at the end!
/// Connect the file again in read mode:
/// ~~~{.bash}
///   Root [0] TFile top("top.root");
/// ~~~
/// The hierarchy can be browsed by the Root browser as shown below
/// ~~~{.bash}
///   Root  TBrowser b;
/// ~~~
/// Click on the left pane on one of the plane directories.
/// This shows the list of all histograms in this directory.
/// Double click on one histogram to draw it (left mouse button).
/// Select different options with the right mouse button.
/// Instead of using the browser, you can also do:
/// ~~~{.bash}
///   Root > tof->cd();
///   Root > plane3->cd();
///   Root > h3_90N->Draw();
/// ~~~
/// \macro_code
///
/// \author Rene Brun

void dirs() {
   // create a new Root file
   TFile *top = new TFile("top.root","recreate");

   // create a subdirectory "tof" in this file
   TDirectory *cdtof = top->mkdir("tof");
   cdtof->cd();    // make the "tof" directory the current directory

   // create a new subdirectory for each plane
   const Int_t nplanes = 10;
   const Int_t ncounters = 100;
   char dirname[50];
   char hname[20];
   char htitle[80];
   Int_t i,j,k;
   TDirectory *cdplane[nplanes];
   TH1F *hn[nplanes][ncounters];
   TH1F *hs[nplanes][ncounters];
   for (i=0;i<nplanes;i++) {
      sprintf(dirname,"plane%d",i);
      cdplane[i] = cdtof->mkdir(dirname);
      cdplane[i]->cd();
      // create counter histograms
      for (j=0;j<ncounters;j++) {
         sprintf(hname,"h%d_%dN",i,j);
         sprintf(htitle,"hist for counter:%d in plane:%d North",j,i);
         hn[i][j] = new TH1F(hname,htitle,100,0,100);
         sprintf(hname,"h%d_%dS",i,j);
         sprintf(htitle,"hist for counter:%d in plane:%d South",j,i);
         hs[i][j] = new TH1F(hname,htitle,100,0,100);
      }
      cdtof->cd();    // change current directory to top
   }

   // Fill histograms
   TRandom r;
   for (i=0;i<nplanes;i++) {
      cdplane[i]->cd();
      for (j=0;j<ncounters;j++) {
         for (k=0;k<100;k++) {
            hn[i][j]->Fill(100*r.Rndm(),i+j);
            hs[i][j]->Fill(100*r.Rndm(),i+j+k);
         }
      }
   }

   // save histogram hierarchy in the file
   top->Write();
   delete top;
}

