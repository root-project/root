{
gSystem->Load("libTreePlayer");
gROOT->ProcessLine(".L helper.C");

gSystem->Load("libEvent");
TFile *Event = TFile::Open("Event.new.split0.root"); 
tree = (TTree*)Event->Get("T");
TTreeFormula *tf;
TObject * o;

check("event.fH");
check("event.GetHistogram()");
check("event.fH.GetXaxis()");
check("event.GetHistogram().GetXaxis()");
check("event.fH.GetXaxis().IsA()");
//check("event.GetHeader()");
check("event.GetTrackCopy()");
check("event.GetTrackCopy(2)");

new TFile("mcpool.root");
Events->Draw("HepMCProduct_PythiaInput__HepMC.obj.evt_.m_signal_process_vertex.@m_particles_out.size()");
}

/*
 Note that there is still a problem (probably when 
 first calling tf->EvalObject() and THEN loading the library
 and recalling tf->EvalObject()
*/
