/// Minimal example showing how a progress bar interacts with `Range`.
///
/// The progress bar must be attached to the node *after* the range is applied
/// so the total number of entries displayed corresponds to the ranged dataset
/// and not the original input.
///
/// \macro_code
void df108_ProgressRange()
{
   ROOT::RDataFrame df(100);
   auto ranged = df.Range(0, 10);
   ROOT::RDF::Experimental::AddProgressBar(ranged);
   auto h = ranged.Define("x", []() { return gRandom->Rndm(); }).Histo1D("x");
   std::cout << h->GetEntries() << std::endl;
}