// Reads a DST file and then rewrites it with a possibly different event version

void RewriteDst(TString eventlibrary, TString input, TString output)
{
  gSystem->Load(eventlibrary);
  
  // Open the input file and prepare for event reading
  TChain input("s", "s");
  input.AddFile(filename);
  TBranch *branch = input.GetBranch("s");
  NuEvent *nu = new NuEvent();
  branch->SetAddress(&nu);
  
  // Output tree
  NuEvent *fNuEvent = new NuEvent();
  TTree outtree("s", "s");
  outtree.Branch("s", "NuEvent", &fNuEvent, 32000, 2);
  
  // Loop over all input NuEvents
  for (Int_t i=0;i<input.GetEntries();++i) {

    if (i >= 16) {
      break;
    }
    
    cout << "Event: " << i << endl;

    // Get the event
    branch->GetEntry(i);
   
    // Fill the new event 
    *fNuEvent = *nu;
    outtree.Fill();
  }
  
  // This is where we should open the file, and write the tree
  TFile tf(output, "RECREATE");
  outtree.Write();
  tf.Close();
  cout << ". Done." << endl;
}

