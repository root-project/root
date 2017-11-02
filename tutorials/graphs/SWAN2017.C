/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// This macro illustrates the use of the time axis on a TGraph
/// with data read from a text file containing the SWAN usage
/// statistics during July 2017.
///
/// \macro_image
/// \macro_code
///
/// \authors Danilo Piparo, Olivier Couet

void SWAN2017()
{
   // Open the data file
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/graphs/");
   dir.ReplaceAll("/./","/");
   FILE *f = fopen(Form("%sSWAN2017.dat",dir.Data()),"r");

   // Create the time graph
   auto *g = new TGraph();
   g->SetTitle("SWAN Users during July 2017;Time;Number of Users");

   // Read the data and fill the graph with time along the X axis and number
   // of users along the Y axis
   char line[80];
   float v;
   char dt[20];
   int i = 0;
   while (fgets(line,80,f)) {
      sscanf(&line[20]  ,"%f", &v);
      strncpy(dt, line, 18);
      dt[19] = '\0';
      g->SetPoint(i, TDatime(dt).Convert(), v);
      i++;
   }
   fclose(f);

   // Draw the graph
   auto *c = new TCanvas("c", "c", 950, 500);
   c->SetLeftMargin(0.07);
   c->SetRightMargin(0.04);
   c->SetGrid();
   g->SetLineWidth(3);
   g->SetLineColor(kBlue);
   g->Draw("al");
   g->GetYaxis()->CenterTitle();

   // Make the X axis labelled with time.
   g->GetXaxis()->SetTimeDisplay(1);
   g->GetXaxis()->CenterTitle();
   g->GetXaxis()->SetTimeFormat("%a %d");
   g->GetXaxis()->SetTimeOffset(0);
   g->GetXaxis()->SetNdivisions(-219);
   g->GetXaxis()->SetLimits(TDatime(2017, 7, 3, 0,0,0).Convert(),
                            TDatime(2017, 7, 22, 0,0,0).Convert());
   g->GetXaxis()->SetLabelSize(0.025);
   g->GetXaxis()->CenterLabels();
}
