/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// This macro illustrates the use of the time axis on a TGraph
/// with data read from a text file containing the SWAN usage
/// statistics during July 2017.
/// We exploit the TDataFrame for reading from the file
///
/// \macro_image
/// \macro_code
///
/// \authors Danilo Piparo, Olivier Couet

void timeSeriesFromCSV_TDF()
{
   // Open the data file. This csv contains the usage statistics of a CERN IT
   // service, SWAN, during two weeks. We would like to plot this data with
   // ROOT to draw some conclusions from it.
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/graphs/");
   dir.ReplaceAll("/./", "/");

   // Read the data from the file using TDataFrame. We do not have headers and
   // we would like the delimiter to be a space
   auto tdf = ROOT::RDF::MakeCsvDataFrame(Form("%sSWAN2017.dat", dir.Data()), false, ' ');

   // We now prepare the graph input
   auto d = tdf.Define("TimeStamp", "auto s = string(Col0) + ' ' +  Col1; return (float) TDatime(s.c_str()).Convert();")
               .Define("Value", "(float)Col2");
   auto timeStamps = d.Take<float>("TimeStamp");
   auto values = d.Take<float>("Value");

   // Create the time graph
   auto g = new TGraph(values->size(), timeStamps->data(), values->data());
   g->SetTitle("SWAN Users during July 2017;Time;Number of Sessions");

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
