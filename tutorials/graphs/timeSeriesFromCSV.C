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

void timeSeriesFromCSV()
{
   // Open the data file. This csv contains the usage statistics of a CERN IT
   // service, SWAN, during two weeks. We would like to plot this data with
   // ROOT to draw some conclusions from it.
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/graphs/");
   dir.ReplaceAll("/./", "/");
   FILE *f = fopen(Form("%sSWAN2017.dat", dir.Data()), "r");

   // Create the time graph
   auto g = new TGraph();
   g->SetTitle("SWAN Users during July 2017;Time;Number of Sessions");

   // Read the data and fill the graph with time along the X axis and number
   // of users along the Y axis
   char line[80];
   float v;
   char dt[20];
   int i = 0;
   while (fgets(line, 80, f)) {
      sscanf(&line[20], "%f", &v);
      strncpy(dt, line, 18);
      dt[19] = '\0';
      g->SetPoint(i, TDatime(dt).Convert(), v);
      i++;
   }
   fclose(f);

   // Draw the graph
   auto c = new TCanvas("c", "c", 950, 500);
   c->SetLeftMargin(0.07);
   c->SetRightMargin(0.04);
   c->SetGrid();
   g->SetLineWidth(3);
   g->SetLineColor(kBlue);
   g->Draw("al");
   g->GetYaxis()->CenterTitle();

   // Make the X axis labelled with time.
   auto xaxis = g->GetXaxis();
   xaxis->SetTimeDisplay(1);
   xaxis->CenterTitle();
   xaxis->SetTimeFormat("%a %d");
   xaxis->SetTimeOffset(0);
   xaxis->SetNdivisions(-219);
   xaxis->SetLimits(TDatime(2017, 7, 3, 0, 0, 0).Convert(), TDatime(2017, 7, 22, 0, 0, 0).Convert());
   xaxis->SetLabelSize(0.025);
   xaxis->CenterLabels();
}
