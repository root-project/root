#include "dt_RunDrawTest.C"

#include "TClassTable.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TSystem.h"

#ifndef __CINT__
#include "Event.h"
#endif

R__LOAD_LIBRARY(libEvent)

template <class Hist>
Hist *RefClone(Hist* orig) {
   Hist *cloned = (Hist*)orig->Clone();
   TString name = orig->GetName();
   name.Prepend("ref");
   cloned->SetName(name);
   cloned->Reset();
   return cloned;
};

template <class Hist>
Hist* RefClone(TDirectory* from, const char* name) {
  Hist * orig = dynamic_cast<Hist*>(from->Get(name));
  if (!orig) {
    TObject *obj = from->Get(name);
    if (obj) {
       cerr << "Type of " << name << " is incorrect. It is "
            << obj->IsA()->GetName() << " instead of " << Hist::Class()->GetName() << endl;
    } else {
      cerr << "Missing " << name << " from " << from->GetName() << endl;
    }
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
   TH1F *refNtrack = RefClone<TH1F>(where,"hNtrack");
   TH1F *refGetNtrack = RefClone<TH1F>(where,"hGetNtrack");
   TH1F *refNseg   = RefClone<TH1F>(where,"hNseg");
   TH1F *refTemp   = RefClone<TH1F>(where,"hTemp");
   TH1F *refHmean  = RefClone<TH1F>(where,"hHmean");
   TH1F *refHAxisMax = RefClone<TH1F>(where,"hHAxisMax");
   TH1F *refHAxisGetMax = RefClone<TH1F>(where,"hHAxisGetMax");
   TH1F *refHGetAxisGetMax  = RefClone<TH1F>(where,"hHGetAxisGetMax");
   TH1F *refHGetAxisMax  = RefClone<TH1F>(where,"hHGetAxisMax");
   TH1F *refGetHGetAxisMax  = RefClone<TH1F>(where,"hGetHGetAxisMax");
   TH1F *refGetRefHGetAxisMax  = RefClone<TH1F>(where,"hGetRefHGetAxisMax");

   TH1F *refPx     = RefClone<TH1F>(where,"hPx");
   TH1F *refPy     = RefClone<TH1F>(where,"hPy");
   TH1F *refPz     = RefClone<TH1F>(where,"hPz");
   TH1F *refRandom = RefClone<TH1F>(where,"hRandom");
   TH1F *refMass2  = RefClone<TH1F>(where,"hMass2");
   TH1F *refBx     = RefClone<TH1F>(where,"hBx");
   TH1F *refBy     = RefClone<TH1F>(where,"hBy");
   TH1F *refXfirst = RefClone<TH1F>(where,"hXfirst");
   TH1F *refYfirst = RefClone<TH1F>(where,"hYfirst");
   TH1F *refZfirst = RefClone<TH1F>(where,"hZfirst");
   TH1F *refXlast  = RefClone<TH1F>(where,"hXlast");
   TH1F *refYlast  = RefClone<TH1F>(where,"hYlast");
   TH1F *refZlast  = RefClone<TH1F>(where,"hZlast");
   TH1F *refCharge = RefClone<TH1F>(where,"hCharge");
   TH1F *refNpoint = RefClone<TH1F>(where,"hNpoint");
   TH1F *refValid  = RefClone<TH1F>(where,"hValid");
   TH1F *refPointValue  = RefClone<TH1F>(where,"hPointValue");
   TH1F *refAlias  = RefClone<TH1F>(where,"hAlias");
   TH1F *refAliasSymbol  = RefClone<TH1F>(where,"hAliasSymbol");
   TH1F *refAliasSymbolFunc  = RefClone<TH1F>(where,"hAliasSymbolFunc");
   TH1F *refBool   = RefClone<TH1F>(where,"hBool");

   TH1F *refFullMatrix   = RefClone<TH1F>(where,"hFullMatrix");
   TH1F *refColMatrix    = RefClone<TH1F>(where,"hColMatrix");
   TH1F *refRowMatrix    = RefClone<TH1F>(where,"hRowMatrix");
   TH1F *refCellMatrix   = RefClone<TH1F>(where,"hCellMatrix");
   TH1F *refFullOper     = RefClone<TH1F>(where,"hFullOper");
   TH1F *refCellOper     = RefClone<TH1F>(where,"hCellOper");
   TH1F *refColOper      = RefClone<TH1F>(where,"hColOper");
   TH1F *refRowOper      = RefClone<TH1F>(where,"hRowOper");
   TH1F *refMatchRowOper = RefClone<TH1F>(where,"hMatchRowOper");
   TH1F *refMatchColOper = RefClone<TH1F>(where,"hMatchColOper");
   TH1F *refRowMatOper   = RefClone<TH1F>(where,"hRowMatOper");
   TH1F *refMatchDiffOper= RefClone<TH1F>(where,"hMatchDiffOper");
   TH1F *refFullOper2    = RefClone<TH1F>(where,"hFullOper2");

   TH1F *refClosestDistance  = RefClone<TH1F>(where,"hClosestDistance");
   TH1F *refClosestDistance2 = RefClone<TH1F>(where,"hClosestDistance2");
   TH1F *refClosestDistance9 = RefClone<TH1F>(where,"hClosestDistance9");

   TH1F *refClosestDistanceIndex = RefClone<TH1F>(where, "hClosestDistanceIndex");
   TH2F *refPxInd = RefClone<TH2F>(where,"hPxInd");

   TH1F *refSqrtNtrack = RefClone<TH1F>(where,"hSqrtNtrack");
   TH1F *refShiftValid = RefClone<TH1F>(where,"hShiftValid");
   TH1F *refAndValid = RefClone<TH1F>(where,"hAndValid");

   TH1F *refString = RefClone<TH1F>(where,"hString");
   TH1F *refStringSpace = RefClone<TH1F>(where,"hStringSpace");
   TH1F *refAliasStr = RefClone<TH1F>(where,"hAliasStr");

   TH1F *refPxBx = RefClone<TH1F>(where,"hPxBx");
   TH1F *refPxBxWeight =  RefClone<TH1F>(where,"hPxBxWeight");

   TH1F *refTriggerBits = RefClone<TH1F>(where,"hTriggerBits");
   TH1F *refTriggerBitsFunc = RefClone<TH1F>(where,"hTriggerBitsFunc");
   TH1F *refFiltTriggerBits = RefClone<TH1F>(where,"hFiltTriggerBits");

   TH1F *refTrackTrigger = RefClone<TH1F>(where,"hTrackTrigger");
   TH1F *refFiltTrackTrigger = RefClone<TH1F>(where,"hFiltTrackTrigger");

   TH1F *refBreit = RefClone<TH1F>(where,"hBreit");

   TH1F *refAlt = RefClone<TH1F>(where,"hAlt");

   TH1F *refSize  = RefClone<TH1F>(where,"hSize");
   TH1F *refSize2 = RefClone<TH1F>(where,"hSize2");

   TH1F *refSumPx = RefClone<TH1F>(where,"hSumPx");
   TH2F *refMaxPx = RefClone<TH2F>(where,"hMaxPx");
   TH2F *refMinPx = RefClone<TH2F>(where,"hMinPx");

   // Loop with user code on all events and fill the ref histograms
   // The code below should produce identical results to the tree->Draw above

   cout << "Recalculating the histograms with custom loop." << endl;

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
      if (strcmp("Event Histogram",event->GetHistogram()->GetTitle())==0) {
         refStringSpace->Fill(1);
      }
      refAliasStr->Fill(strstr(event->GetType(),"1")!=0);
      refBool->Fill(event->IsValid());

      Nvertex = event->GetNvertex();
      for(i0=0;i0<Nvertex;i0++) {
         refClosestDistance->Fill(event->GetClosestDistance(i0));
      }
      if (Nvertex>2) refClosestDistance2->Fill(event->GetClosestDistance(2));
      if (Nvertex>9) refClosestDistance9->Fill(event->GetClosestDistance(9));
      refClosestDistanceIndex->Fill(event->GetClosestDistance(Nvertex/2));

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
      refSize->Fill(ntracks);
      refSize->Fill(ntracks);
      refSize2->Fill(ntracks);
      refSize2->Fill(ntracks);
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
      Double_t sumPx = 0;
      Double_t maxPx = 0.0;
      Double_t maxPy = 0.0;
      Double_t minPx = 5.0;
      Double_t minPy = 5.0;
      for (i=0;i<ntracks;i++) {
         t = (Track*)tracks->UncheckedAt(i);
         sumPx += t->GetPx();
         if (t->GetPy() > 1.0) {
            if (t->GetPx() > maxPx) maxPx = t->GetPx();
            if (t->GetPx() < minPx) minPx = t->GetPx();
         }
         if (t->GetPy() > maxPy) maxPy = t->GetPy();
         if (t->GetPy() < minPy) minPy = t->GetPy();
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
         refAliasSymbol->Fill(t->GetPx()+t->GetPy());
         refAliasSymbolFunc->Fill(t->GetPx()+t->GetPy());
      }
      refSumPx->Fill(sumPx);
      if (maxPx > 0) {
         refMaxPx->Fill(maxPy,maxPx);
      }
      if (minPx < 5.0) {
         refMinPx->Fill(minPy,minPx);
      }
   }

   delete event;
   Event::Reset();

}

void dt_MakeRef(const char* from, Int_t verboseLevel = 2) {
   SetVerboseLevel(verboseLevel);

   if (!TClassTable::GetDict("Event")) {
      gSystem->Load("libEvent");
      gHasLibrary = kTRUE;
   } else {
      gHasLibrary = kTRUE;
   }

   gROOT->GetList()->Delete();

   TFile *hfile = new TFile(from);
   TTree *tree = (TTree*)hfile->Get("T");

   TFile* f= new TFile("dt_reference.root","recreate");
   MakeHisto(tree,f);
   f->Write();
   delete f;

   delete hfile;

   gROOT->cd();
   cout << "Checking histograms" << endl;
   Compare(gDirectory);

    //   gROOT->GetList()->Delete();

}
