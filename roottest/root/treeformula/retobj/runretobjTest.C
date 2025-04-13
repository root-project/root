#ifdef ClingWorkAroundUnnamedDetection2
void runretobjTest()
#endif
{
gSystem->Load("libTreePlayer");
#ifndef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(".L helper.C");
#endif

gSystem->Load("libEvent");
TFile *Event = TFile::Open("Event.new.split0.root");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("TTree *tree = (TTree*)gFile->Get(\"T\"); TTreeFormula *tf;TObject * o;");
   gROOT->ProcessLine(".L helper.C");
#else
   auto tree = (TTree*)Event->Get("T");
   TTreeFormula *tf;
   TObject * o;
#endif

#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
                   "check(\"event.fH\");"
                   "check(\"event.GetHistogram()\");"
                   "check(\"event.fH.GetXaxis()\");"
                   "check(\"event.GetHistogram().GetXaxis()\");"
                   "check(\"event.fH.GetXaxis().IsA()\");"
//                   "//check(\"event.GetHeader()\");"
#ifndef ClingWorkAroundCallfuncAndReturnByValue
                   "check(\"event.GetTrackCopy()\");"
                   "check(\"event.GetTrackCopy(2)\");"
#endif
                   );
#else
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
#endif // ClingWorkAroundMissingDynamicScope

new TFile("mcpool.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree* Events; gFile->GetObject("Events",Events);
#endif
Events->Draw("HepMCProduct_PythiaInput__HepMC.obj.evt_.m_signal_process_vertex.@m_particles_out.size()");
}

/*
 Note that there is still a problem (probably when 
 first calling tf->EvalObject() and THEN loading the library
 and recalling tf->EvalObject()
*/
