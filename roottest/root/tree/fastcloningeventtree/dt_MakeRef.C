#include "dt_RunDrawTest.C"

#include "TClassTable.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TSystem.h"
#include "TClonesArray.h"

#include "Event.h"

#include <iostream>

TH1* RefClone(TH1* orig)
{
  TH1 *cloned = (TH1*)orig->Clone();
  TString name = orig->GetName();
  name.Prepend("ref");
  cloned->SetName(name);
  cloned->Reset();
  return cloned;
};

TH1* RefClone(TDirectory* from, const char* name) {
  TH1 * orig = (TH1*)from->Get(name);
  if (!orig) {
    cerr << "Missing " << name << " from " << from->GetName() << endl;
    return 0;
  }
  return RefClone(orig);
}

void MakeHisto(TTree *tree, TDirectory* To) {

   cout << "Generating histograms from TTree::Draw" << endl;
   TDirectory* where = GenerateDrawHist(tree);
   To->cd();

   Event *event = new Event();
   tree->SetBranchAddress("event",&event);

   //We make clones of the generated histograms
   //We set new names and reset the clones.
   //We want to have identical histogram limits
   TH1 *refNtrack = RefClone(where,"hNtrack");
   TH1 *refGetNtrack = RefClone(where,"hGetNtrack");
   TH1 *refNseg   = RefClone(where,"hNseg");
   TH1 *refTemp   = RefClone(where,"hTemp");
   TH1 *refHmean  = RefClone(where,"hHmean");
   TH1 *refHAxisMax = RefClone(where,"hHAxisMax");
   TH1 *refHAxisGetMax = RefClone(where,"hHAxisGetMax");
   TH1 *refHGetAxisGetMax  = RefClone(where,"hHGetAxisGetMax");
   TH1 *refHGetAxisMax  = RefClone(where,"hHGetAxisMax");
   TH1 *refGetHGetAxisMax  = RefClone(where,"hGetHGetAxisMax");
   TH1 *refGetRefHGetAxisMax  = RefClone(where,"hGetRefHGetAxisMax");

   TH1 *refPx     = RefClone(where,"hPx");
   TH1 *refPy     = RefClone(where,"hPy");
   TH1 *refPz     = RefClone(where,"hPz");
   TH1 *refRandom = RefClone(where,"hRandom");
   TH1 *refMass2  = RefClone(where,"hMass2");
   TH1 *refBx     = RefClone(where,"hBx");
   TH1 *refBy     = RefClone(where,"hBy");
   TH1 *refXfirst = RefClone(where,"hXfirst");
   TH1 *refYfirst = RefClone(where,"hYfirst");
   TH1 *refZfirst = RefClone(where,"hZfirst");
   TH1 *refXlast  = RefClone(where,"hXlast");
   TH1 *refYlast  = RefClone(where,"hYlast");
   TH1 *refZlast  = RefClone(where,"hZlast");
   TH1 *refCharge = RefClone(where,"hCharge");
   TH1 *refNpoint = RefClone(where,"hNpoint");
   TH1 *refValid  = RefClone(where,"hValid");
   TH1 *refPointValue  = RefClone(where,"hPointValue");
   TH1 *refAlias  = RefClone(where,"hAlias");

   TH1 *refFullMatrix   = RefClone(where,"hFullMatrix");
   TH1 *refColMatrix    = RefClone(where,"hColMatrix");
   TH1 *refRowMatrix    = RefClone(where,"hRowMatrix");
   TH1 *refCellMatrix   = RefClone(where,"hCellMatrix");
   TH1 *refFullOper     = RefClone(where,"hFullOper");
   TH1 *refCellOper     = RefClone(where,"hCellOper");
   TH1 *refColOper      = RefClone(where,"hColOper");
   TH1 *refRowOper      = RefClone(where,"hRowOper");
   TH1 *refMatchRowOper = RefClone(where,"hMatchRowOper");
   TH1 *refMatchColOper = RefClone(where,"hMatchColOper");
   TH1 *refRowMatOper   = RefClone(where,"hRowMatOper");
   TH1 *refMatchDiffOper= RefClone(where,"hMatchDiffOper");
   TH1 *refFullOper2    = RefClone(where,"hFullOper2");

   TH1 *refClosestDistance  = RefClone(where,"hClosestDistance");
   TH1 *refClosestDistance2 = RefClone(where,"hClosestDistance2");
   TH1 *refClosestDistance9 = RefClone(where,"hClosestDistance9");

   TH1 *refClosestDistanceIndex = RefClone(where, "hClosestDistanceIndex");
   TH2 *refPxInd = (TH2F*)RefClone(where,"hPxInd");

   TH1 *refSqrtNtrack = RefClone(where,"hSqrtNtrack");
   TH1 *refShiftValid = RefClone(where,"hShiftValid");
   TH1 *refAndValid = RefClone(where,"hAndValid");

   TH1 *refString = RefClone(where,"hString");
   TH1 *refAliasStr = RefClone(where,"hAliasStr");

   TH1 *refPxBx = RefClone(where,"hPxBx");
   TH1 *refPxBxWeight =  RefClone(where,"hPxBxWeight");

   TH1 *refTriggerBits = RefClone(where,"hTriggerBits");
   TH1 *refTriggerBitsFunc = RefClone(where,"hTriggerBitsFunc");
   TH1 *refFiltTriggerBits = RefClone(where,"hFiltTriggerBits");

   TH1 *refTrackTrigger = RefClone(where,"hTrackTrigger");
   TH1 *refFiltTrackTrigger = RefClone(where,"hFiltTrackTrigger");

   TH1 *refBreit = RefClone(where,"hBreit");

   TH1 *refAlt = RefClone(where,"hAlt");

   TH1 *refRun = RefClone(where,"hRun");
   // TH1 *refVRun2 = RefClone(where,"hVRun2");
   // TH1 *refVRunIndex = RefClone(where,"hVRunIndex");
   TH1 *refRunFunc = RefClone(where,"hRunFunc");
   // TH1 *refVPx = RefClone(where,"hVPx");
   // TH1 *refVCharge = RefClone(where,"hVCharge");
   //TH1F *ref = RefClone(where,"");

   // Loop with user code on all events and fill the ref histograms
   // The code below should produce identical results to the tree->Draw above

   std::cout << "Recalculating the histograms with custom loop." << std::endl;

   TClonesArray *tracks = event->GetTracks();
   Int_t nev = (Int_t)tree->GetEntries();
   Int_t i, ntracks, evmod,i0,i1, Nvertex;
   Track *t;
   EventHeader *head;
   Int_t nbin = 0;
   for (Int_t ev=0;ev<nev;ev++) {
      nbin += tree->GetEntry(ev);
      head = event->GetHeader();
      evmod = head->GetEvtNum()%10;
      refNtrack->Fill(event->GetNtrack());
      refGetNtrack->Fill(event->GetNtrack());
      refNseg->Fill(event->GetNseg());
      refTemp->Fill(event->GetTemperature());
      refHmean->Fill(event->GetHistogram()->GetMean());
      refHAxisMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refHAxisGetMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refHGetAxisGetMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refHGetAxisMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refGetHGetAxisMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refGetRefHGetAxisMax->Fill(event->GetHistogram()->GetXaxis()->GetXmax());
      refSqrtNtrack->Fill(sqrt(event->GetNtrack()));

      if (!strcmp("type1",event->GetType()))
        refString->Fill(event->GetHeader()->GetEvtNum());
      if (strstr(event->GetType(),"1")) {
        refString->Fill(event->GetHeader()->GetEvtNum());
      }
      refAliasStr->Fill(strstr(event->GetType(),"1")!=0);

      Nvertex = event->GetNvertex();
      for(i0=0;i0<Nvertex;i0++) {
         refClosestDistance->Fill(event->GetClosestDistance(i0));
      }
      if (Nvertex>2) refClosestDistance2->Fill(event->GetClosestDistance(2));
      if (Nvertex>9) refClosestDistance9->Fill(event->GetClosestDistance(9));
      refClosestDistanceIndex->Fill(event->GetClosestDistance(Nvertex/2));

      /*
      {
         std::vector<EventHeader>::const_iterator eiter = event->fVEvtHdr.begin();
         int i=0;
         while(eiter != event->fVEvtHdr.end()) {
            Int_t run = (*eiter).GetRun();
            refVRun->Fill( run  );
            refVRunFunc->Fill( run );
            if (i==2) refVRun2->Fill( run );
            if (i==(Nvertex/2)) refVRunIndex->Fill( run );
            ++i;
            ++eiter;
         }
         if (i<=(Nvertex/2)) refVRunIndex->Fill( 0 );
      }
      {
         std::vector<Track*>::const_iterator titer = event->fVTracks.begin();
         int i=0;
         while(titer != event->fVTracks.end()) {
            float px = (*titer)->GetPx();
            float charge = (*titer)->GetCharge();
            if (px<0) refVCharge->Fill(charge);
            refVPx->Fill(px);
            ++titer;
         }
      }
      */

      for(i0=0;i0<4;i0++) {
         for(i1=0;i1<4;i1++) {
            refFullMatrix->Fill(event->GetMatrix(i0,i1));

            int i2 = i0*4+i1;
            refAlt->Fill( event->GetMatrix(i0,i1) - ( (i2<Nvertex) ? event->GetClosestDistance(i2) : 0 ) );

         }
         refColMatrix->Fill(event->GetMatrix(i0,0));
         refRowMatrix->Fill(event->GetMatrix(1,i0)); // done here because the matrix is square!

      }
      refCellMatrix->Fill(event->GetMatrix(2,2));

      TBits bits = event->GetTriggerBits();
      Int_t nbits = bits.GetNbits();
      Int_t ncx = refTriggerBits->GetXaxis()->GetNbins();
      Int_t nextbit = -1;
      while(1) {
         nextbit = bits.FirstSetBit(nextbit+1);
         if (nextbit >= nbits) break;
         if (nextbit > ncx) refTriggerBits->Fill(ncx+1);
         else               refTriggerBits->Fill(nextbit);
         if (nextbit > ncx) refTriggerBitsFunc->Fill(ncx+1);
         else               refTriggerBitsFunc->Fill(nextbit);
      }
      if (bits.TestBitNumber(28)) refFiltTriggerBits->Fill(nbits);

      ntracks = event->GetNtrack();
      if ( 5 < ntracks ) {
         t = (Track*)tracks->UncheckedAt(5);
         for(i0=0;i0<4;i0++) {
            for(i1=0;i1<4;i1++) {
            }
            refColOper->Fill( event->GetMatrix(i0,1) - t->GetVertex(1) );
            refRowOper->Fill( event->GetMatrix(2,i0) - t->GetVertex(2) );
         }
         for(i0=0;i0<3;i0++) {
            refMatchRowOper->Fill( event->GetMatrix(2,i0) - t->GetVertex(i0) );
            refMatchDiffOper->Fill( event->GetMatrix(i0,2) - t->GetVertex(i0) );
         }
         refCellOper->Fill( event->GetMatrix(2,1) - t->GetVertex(1) );
      }
      for (i=0;i<ntracks;i++) {
         t = (Track*)tracks->UncheckedAt(i);
         if (evmod == 0) refPx->Fill(t->GetPx());
         if (evmod == 0) refPy->Fill(t->GetPy());
         if (evmod == 0) refPz->Fill(t->GetPz());
         if (evmod == 1) refRandom->Fill(t->GetRandom(),3);
         if (evmod == 1) refMass2->Fill(t->GetMass2());
         if (evmod == 1) refBx->Fill(t->GetBx());
         if (evmod == 1) refBy->Fill(t->GetBy());
         if (evmod == 2) refXfirst->Fill(t->GetXfirst());
         if (evmod == 2) refYfirst->Fill(t->GetYfirst());
         if (evmod == 2) refZfirst->Fill(t->GetZfirst());
         if (evmod == 3) refXlast->Fill(t->GetXlast());
         if (evmod == 3) refYlast->Fill(t->GetYlast());
         if (evmod == 3) refZlast->Fill(t->GetZlast());
         if (t->GetPx() < 0) {
            refCharge->Fill(t->GetCharge());
            refNpoint->Fill(t->GetNpoint());
            refValid->Fill(t->GetValid());
         }
         Int_t valid = t->GetValid();
         refShiftValid->Fill(valid << 4);
         refShiftValid->Fill( (valid << 4) >> 2 );
         if (event->GetNvertex()>10 && event->GetNseg()<=6000) {
            refAndValid->Fill( t->GetValid() & 0x1 );
         }

         Track * t2 = (Track*)tracks->At(t->GetNpoint()/6);
         if (t2 && t2->GetPy()>0) {
            refPxInd->Fill(t2->GetPy(),t->GetPx());
         }
         float Bx,By;
         Bx = t->GetBx();
         By = t->GetBy();
         if ((Bx>.15) || (By<=-.15)) refPxBx->Fill(t->GetPx());
         double weight = Bx*Bx*(Bx>.15) + By*By*(By<=-.15);
         if (weight) refPxBxWeight->Fill(t->GetPx(),weight);

         if (i<4) {
            for(i1=0;i1<3;i1++) { // 3 is the min of the 2nd dim of Matrix and Vertex
               refFullOper ->Fill( event->GetMatrix(i,i1) - t->GetVertex(i1) );
               refFullOper2->Fill( event->GetMatrix(i,i1) - t->GetVertex(i1) );
               refRowMatOper->Fill( event->GetMatrix(i,2) - t->GetVertex(i1) );
            }
            refMatchColOper->Fill( event->GetMatrix(i,2) - t->GetVertex(1) );
         }
         for(i1=0; i1<t->GetN(); i1++) {
            refPointValue->Fill( t->GetPointValue(i1) );
         }
         TBits bits = t->GetTriggerBits();
         Int_t nbits = bits.GetNbits();
         Int_t ncx = refTrackTrigger->GetXaxis()->GetNbins();
         Int_t nextbit = -1;
         while(1) {
            nextbit = bits.FirstSetBit(nextbit+1);
            if (nextbit >= nbits) break;
            if (nextbit > ncx) refTrackTrigger->Fill(ncx+1);
            else               refTrackTrigger->Fill(nextbit);
         }
         if (bits.TestBitNumber(5)) refFiltTrackTrigger->Fill(t->GetPx());
         refBreit->Fill(TMath::BreitWigner(t->GetPx(),3,2));

         refAlias->Fill(head->GetEvtNum()*6+t->GetPx()*t->GetPy());
      }
   }

   delete event;
   // Event::Reset();

}

void dt_MakeRef(const char* from, Int_t verboseLevel = 2)
{
   SetVerboseLevel(verboseLevel);

   gHasLibrary = kTRUE;

   //if (!TClassTable::GetDict("Event")) {
   //   gSystem->Load("libTestIoEvent");
   //   gHasLibrary = kTRUE;
   //}

   gROOT->GetList()->Delete();

   TFile *hfile = new TFile(from);
   TTree *tree = (TTree*)hfile->Get("T");

   TFile* f = TFile::Open("dt_reference.root","recreate");
   MakeHisto(tree,f);
   f->Write();
   delete f;

   delete hfile;

   gROOT->cd();
   cout << "Checking histograms" << endl;
   Compare(gDirectory);

    //   gROOT->GetList()->Delete();

}
