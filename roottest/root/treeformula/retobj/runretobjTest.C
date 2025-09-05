TTree *tree = nullptr;
TTreeFormula *tf = nullptr;
TObject *o = nullptr;


void check(const char*arg)
{
  tf = new TTreeFormula("test",arg,tree);
  tree->GetEntry(0);
  o = (TObject*)tf->EvalObject();
  o = (TObject*)tf->EvalObject();
  if (o) o->IsA()->Print();
  if (tf->EvalClass())
     tf->EvalClass()->Print();
}


void runretobjTest()
{
   TFile *Event = TFile::Open("Event.new.split0.root");
   tree = (TTree*)Event->Get("T");

   check("event.fH");
   check("event.GetHistogram()");
   check("event.fH.GetXaxis()");
   check("event.GetHistogram().GetXaxis()");
   check("event.fH.GetXaxis().IsA()");
   //check("event.GetHeader()");
   #ifndef ClingWorkAroundCallfuncAndReturnByValue
   check("event.GetTrackCopy()");
   check("event.GetTrackCopy(2)");
   #endif // ClingWorkAroundCallfuncAndReturnByValue

   TFile::Open("mcpool.root");
   TTree* Events = nullptr;
   gFile->GetObject("Events",Events);
   Events->Draw("HepMCProduct_PythiaInput__HepMC.obj.evt_.m_signal_process_vertex.@m_particles_out.size()");
}

/*
 Note that there is still a problem (probably when
 first calling tf->EvalObject() and THEN loading the library
 and recalling tf->EvalObject()
*/
