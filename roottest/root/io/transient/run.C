{
gSystem->Load("libEG");
gSystem->Load("libPhysics");
//root [2] .L ../lib/libmicro.so
gROOT->ProcessLine(".L TBigDSWriteParticle.cxx+");
gROOT->ProcessLine(".L TBigDSWriteEvent.cxx+");

TFile *tfile= new TFile("/tmp/dsttest.root","RECREATE");
TTree *tft = new TTree("tft","");
tfile->SetCompressionLevel(2);
TBigDSWriteEvent *myevent=new TBigDSWriteEvent();
TBigDSWriteParticle *mypart;
tft->Branch("TBigDSWriteEvent","TBigDSWriteEvent",&myevent,5000000,1);

// Error in <TClass::New>: cannot create object of class TIter

TBufferFile b(TBuffer::kWrite);
b.WriteObjectAny(myevent,TBigDSWriteEvent::Class());

}
