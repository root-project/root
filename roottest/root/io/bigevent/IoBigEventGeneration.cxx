// @(#)root/test:$Id$
// Author: Rene Brun   19/08/96

////////////////////////////////////////////////////////////////////////
//
//                       Event and Track classes
//                       =======================
//
//  The Event class is a naive/simple example of an event structure.
//     public:
//        char           fType[20];
//        Int_t          fNtrack;
//        Int_t          fNseg;
//        Int_t          fNvertex;
//        UInt_t         fFlag;
//        Float_t        fTemperature;
//        EventHeader    fEvtHdr;
//        TClonesArray  *fTracks;
//        TH1F          *fH;
//        Float_t        fMatrix[4][4];
//        Float_t       *fClosestDistance; //[fNvertex] indexed array! 
//
//   The EventHeader class has 3 data members (integers):
//     public:
//        Int_t          fEvtNum;
//        Int_t          fRun;
//        Int_t          fDate;
//
//
//   The Event data member fTracks is a pointer to a TClonesArray.
//   It is an array of a variable number of tracks per event.
//   Each element of the array is an object of class Track with the members:
//     private:
//        Float_t      fPx;           //X component of the momentum
//        Float_t      fPy;           //Y component of the momentum
//        Float_t      fPz;           //Z component of the momentum
//        Float_t      fRandom;       //A random track quantity
//        Float_t      fMass2;        //The mass square of this particle
//        Float_t      fBx;           //X intercept at the vertex
//        Float_t      fBy;           //Y intercept at the vertex
//        Float_t      fMeanCharge;   //Mean charge deposition of all hits of this track
//        Float_t      fXfirst;       //X coordinate of the first point
//        Float_t      fXlast;        //X coordinate of the last point
//        Float_t      fYfirst;       //Y coordinate of the first point
//        Float_t      fYlast;        //Y coordinate of the last point
//        Float_t      fZfirst;       //Z coordinate of the first point
//        Float_t      fZlast;        //Z coordinate of the last point
//        Float_t      fCharge;       //Charge of this track
//        Float_t      fVertex[3];    //Track vertex position
//        Int_t        fNpoint;       //Number of points for this track
//        Short_t      fValid;        //Validity criterion
//
//   An example of a batch program to use the Event/Track classes is given
//   in this directory: MainEvent.
//   Look also in the same directory at the following macros:
//     - eventa.C  an example how to read the tree
//     - eventb.C  how to read events conditionally
//
//   During the processing of the event (optionally) also a large number
//   of histograms can be filled. The creation and handling of the
//   histograms is taken care of by the HistogramManager class.
//
////////////////////////////////////////////////////////////////////////

#include "TLine.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TAttLine.h"

#include "Event.h"


ClassImp(EventHeader)
ClassImp(Event)
ClassImp(Track)
ClassImp(UShortVector)
ClassImp(BigTrack)
ClassImp(HistogramManager)

TClonesArray *Event::fgTracks = 0;
TH1F *Event::fgHist = 0;
Double_t EventHeader::fgNever = 55;

//______________________________________________________________________________
Event::Event()
{
   // default constructor of an Event object.
   // When the constructor is invoked for the first time, the class static
   // variable fgTracks is 0 and the TClonesArray fgTracks is created.

   //if (!fgTracks) fgTracks = new TClonesArray("Track", 1000);
   if (!fgTracks) fgTracks = new TClonesArray("BigTrack", 1000);
   fTracks = fgTracks;
   fNtrack = 0;
   fH      = 0;
   fNvertex= 0;
   Int_t i0,i1;
   for (i0 = 0; i0 < 4; i0++) {
      for (i1 = 0; i1 < 4; i1++) {
         fMatrix[i0][i1] = 0.0;
      }
   }
   for (i0 = 0; i0 <10; i0++) fMeasures[i0] = 0;
   fClosestDistance = 0;
   
   Int_t i,j;
   for (i=0;i<12;i++) fVectorint.push_back(i);
   
   fString = "this is a C++ string";
   fStringp = new string("hello string");
   fTstringp = 0;
   fVaxis[0] = new TAxis();
   fVaxis[1] = new TAxis();
   fVaxis[2] = new TAxis();
      
   fPaxis = 0;
   fQaxis = 0;
   fMapTAxisp = 0;
   fSetTAxisp = 0;
   fMultiSetTAxisp = 0;
   fListString = new list<string>;
   fListString->push_back("First string");
   fListString->push_back("Second string");
   fListString->push_back("Third string");

   fListStringp.push_back(new string("1st string"));
   fListStringp.push_back(new string("2nd string"));
   fListStringp.push_back(new string("3rd string"));
   fListStringp.push_back(new string("4th string"));
   
   fVectAxis.clear();
   static vector<TAxis*> vecta;
   for (i = 0; i < 4; i++) {
      vecta.clear();
      for (j = 0; j < i + 2; j++) {
         vecta.push_back(new TAxis());
      }
      fVectAxis.push_back(vecta);
   }
   fMapString.clear();
   fDequePair.clear();
   
   fVectorshort.clear();
   for (i = 0; i < 120; i++)
      fVectorshort.push_back(i);

   fVectorTobject = new vector<TObject>;
   TObject a1,a2,a3;
   fVectorTobject->push_back(a1);
   fVectorTobject->push_back(a2);
   fVectorTobject->push_back(a3);

   for (i = 0; i < 6; i++)
      fVectorTnamed[i] = new vector<TNamed>;

   TLine line;
   fVectorTLine.clear();
   for (i = 0; i < 40; i++) {
      line.SetX1(gRandom->Rndm());
      line.SetX2(gRandom->Rndm());
      line.SetY1(gRandom->Rndm());
      line.SetY2(gRandom->Rndm());
      line.SetLineColor(i); 
      fVectorTLine.push_back(line);
   }

   TAttLine attline;
   fDeque.clear();
   for (i = 0; i < 4; i++) {
      attline.SetLineColor(i);
      fDeque.push_back(attline);
   }

   fArrayF.Set(24);
   fArrayI = new TArrayI(24);
   for (i = 0; i < 24; i++) {
      fArrayI->fArray[i] = 24 - i;
      fArrayF.fArray[i] = 48 - 2 * i;
   }

   fRefH = 0;
   fEventName = 0;
   fTracksInVertex = 0;
}

//______________________________________________________________________________
Event::Event(Int_t /*enumber*/)
{
   // Create an Event object.
   // When the constructor is invoked for the first time, the class static
   // variable fgTracks is 0 and the TClonesArray fgTracks is created.

   if (!fgTracks)
      fgTracks = new TClonesArray("Track", 1000);
   fTracks = fgTracks;
   fNtrack = 0;
   fH      = 0;
   Int_t i0, i1;
   for (i0 = 0; i0 < 4; i0++) {
      for (i1 = 0; i1 < 4; i1++) {
         fMatrix[i0][i1] = 0.0;
      }
   }
   for (i0 = 0; i0 < 10; i0++)
      fMeasures[i0] = 0;
   fClosestDistance = 0;
   
   Int_t i;
   for (i = 0; i < 12; i++)
      fVectorint.push_back(i);

   fString = "this is a C++ string";
   fTstringp = 0;
   fVaxis[0] = new TAxis();
   fVaxis[1] = new TAxis();
   fVaxis[2] = new TAxis();
      
   fPaxis = 0;
   fQaxis = 0;
   fMapTAxisp = 0;
   fSetTAxisp = 0;
   fMultiSetTAxisp = 0;

   fVectorshort.clear();
   for (i = 0; i < 120; i++)
      fVectorshort.push_back(i);

   fVectorTobject = new vector<TObject>;
   TObject a1, a2, a3;
   fVectorTobject->push_back(a1);
   fVectorTobject->push_back(a2);
   fVectorTobject->push_back(a3);

   for (i = 0; i < 6; i++)
      fVectorTnamed[i] = new vector<TNamed>;

   TAttLine attline;
   fDeque.clear();
   for (i = 0; i < 40; i++) {
      attline.SetLineColor(i);
      fDeque.push_back(attline);
   }
   fEventName = 0;
}

//______________________________________________________________________________
Event::~Event()
{
   Clear();
   if (fH == fgHist)
      fgHist = 0;
   delete fH;
   fH = 0;
   delete [] fClosestDistance;
   delete fVaxis[0];
   delete fVaxis[1];
   delete fVaxis[2];
   if (fEventName)
      delete[] fEventName;
   if (fTracksInVertex)
      delete[] fTracksInVertex;
}

//______________________________________________________________________________
void Event::AddTrack(Float_t random)
{
   // Add a new track to the list of tracks for this event.
   // To avoid calling the very time consuming operator new for each track,
   // the standard but not well know C++ operator "new with placement"
   // is called. If tracks[i] is 0, a new Track object will be created
   // otherwise the previous Track[i] will be overwritten.

   TClonesArray &tracks = *fTracks;
   // new (tracks[fNtrack++]) Track(random);
   new (tracks[fNtrack]) BigTrack(random, fNtrack % 100);
   ++fNtrack;
}

//______________________________________________________________________________
void Event::Clear(Option_t *option)
{
   fUshort.clear();
   fTracks->Clear(option);
}

//______________________________________________________________________________
void Event::Reset(Option_t * /*option*/)
{
   // Static function to reset all static objects for this event
   // fgTracks->Delete(option);
   delete fgTracks;
   fgTracks = 0;
   fgHist = 0;
}

//______________________________________________________________________________
void Event::SetHeader(Int_t i, Int_t run, Int_t date, Float_t random)
{
   if (i % 10 < 3)
      fBoolA = kTRUE;
   else
      fBoolA = kFALSE;
   Int_t nch = 15;
   if (i > 100)
      nch += 3;
   if (i > 10000)
      nch += 3;
   if (fEventName)
      delete[] fEventName;
   fEventName = new char[nch];
   snprintf(fEventName, nch, "Event%d_Run%d", i, 200);
   fNtrack = 0;
   fEvtHdr.Set(i, run, date);
   if (!fgHist)
      fgHist = new TH1F("hstat", "Event Histogram", 100, 0, 1);
   fH = fgHist;
   fH->Fill(random);
   fRefH = fH;
   //fill Lachaud strings
   fLachaud.clear();
   nch = Int_t(10 * random);
   char lachaud[64];
   string lac;
   for (Int_t j=0;j<nch;j++) {
      snprintf(lachaud, 64, "run%d event%d j%d", run, i, j);
      lac = lachaud;
      fLachaud.push_back(lac);
   }
}

//______________________________________________________________________________
void Event::SetMeasure(UChar_t which, Int_t what) {
   if (which < 10)
      fMeasures[which] = what;
}

//______________________________________________________________________________
void Event::SetRandomVertex() {
   // This delete is to test the relocation of variable length array
   delete [] fClosestDistance;
   if (!fNvertex) {
      fClosestDistance = 0;
      return;
   }
   fClosestDistance = new Float_t[fNvertex];
   fTracksInVertex  = new char[fNvertex];
   fTstringp = new TString[fNvertex];
   fPaxis = new TAxis[fNvertex];
   fQaxis = new TAxis*[fNvertex];
   fMapTNamedp.clear();
   fMultiMapTNamedp.clear();
   fSetTAxis.clear();
   //delete fMapTAxisp;
   fMapTAxisp = new map<TAxis *, int>;
   fSetTAxisp = new set<TAxis*>;
   fMultiSetTAxisp = new multiset<TAxis*>;
   for (Int_t k = 0; k < fNvertex; k++ ) {
      fTracksInVertex[k] = k;
      fClosestDistance[k] = gRandom->Gaus(1, 1);
      fTstringp[k] = "fTstringp";
      fQaxis[k] = new TAxis();
      fMapTNamedp.insert(make_pair(new TNamed("ii", "jj"), k));
      fMapTAxisp->insert(make_pair(new TAxis(), k));
      fSetTAxisp->insert(new TAxis());
      fMultiMapTNamedp.insert(make_pair(new TNamed("ii", "jj"), k));
      fMultiSetTAxisp->insert(new TAxis());
   }
   Double_t x = gRandom->Gaus(0, 1);
   Double_t y = gRandom->Gaus(0, 1);
   Double_t z = gRandom->Gaus(0, 100);
   Double_t t = TMath::Sqrt(x * x + y * y + z * z);
   fLorentz.SetXYZT(x, y, z, t);
}

//______________________________________________________________________________
void Event::ShowLachaud() {
   
   vector<string>::iterator R__k;
   printf("Lachaud vector has %zd entries\n", fLachaud.size());
   for (R__k = fLachaud.begin(); R__k != fLachaud.end(); ++R__k) {
      printf(" %s\n", (*R__k).c_str());
   }
}

//______________________________________________________________________________
UShortVector::UShortVector(Int_t n)
{
   // Create a big track object.
   fNshorts = n;
}

//______________________________________________________________________________
BigTrack::BigTrack(Float_t random, Int_t special) : Track(random)
{
   // Create a big track object.
   fSpecial = special;
   Double_t x = gRandom->Gaus(0, 1);
   Double_t y = gRandom->Gaus(0, 1);
   Double_t z = gRandom->Gaus(0, 100);
   Double_t t = TMath::Sqrt(x * x + y * y + z * z);
   fKine.SetXYZT(x, y, z, t);
}

//______________________________________________________________________________
Track::Track(Float_t random) : TObject()
{
   // Create a track object.
   // Note that in this example, data members do not have any physical meaning.

   Float_t a, b, px, py;
   gRandom->Rannor(px, py);
   fPx = px;
   fPy = py;
   fPz = TMath::Sqrt(px * px + py * py);
   fRandom = 1000 * random;
   if (fRandom < 10)
      fMass2 = 0.08;
   else if (fRandom < 100)
      fMass2 = 0.8;
   else if (fRandom < 500)
      fMass2 = 4.5;
   else if (fRandom < 900)
      fMass2 = 8.9;
   else
      fMass2 = 9.8;
   gRandom->Rannor(a, b);
   fBx = 0.1 * a;
   fBy = 0.1 * b;
   fMeanCharge = 0.01 * gRandom->Rndm(1);
   gRandom->Rannor(a, b);
   fXfirst = a * 10;
   fXlast = b * 10;
   gRandom->Rannor(a, b);
   fYfirst = a * 12;
   fYlast = b * 16;
   gRandom->Rannor(a, b);
   fZfirst = 50 + 5 * a;
   fZlast = 200 + 10 * b;
   fCharge = Float_t(Int_t(3 * gRandom->Rndm(1)) - 1);
   fVertex[0] = gRandom->Gaus(0, 0.1);
   fVertex[1] = gRandom->Gaus(0, 0.2);
   fVertex[2] = gRandom->Gaus(0, 10);
   for (Int_t i0(0), in(0); i0 < 3; i0++) {
      for (Int_t i1 = 0; i1 < 4; i1++) {
         fCovar[i0][i1] = in;
         for (Int_t i2 = 0; i2 < 2; i2++) {
            fCovara[i0][i1][i2] = in;
            in++;
         }
      }
   }

   fNpoint = Int_t(60 + 10 * gRandom->Rndm(1));
   fValid = Int_t(0.6 + gRandom->Rndm(1));
   if (fNpoint) {
      fPoints = new Short_t[fNpoint];
      for (Int_t j = 0; j < fNpoint; j++)
         fPoints[j] = j;
   } else {
      fPoints = 0;
   }
   Int_t nints = (Int_t)gRandom->Rndm() * 4;
   fInts.Set(nints);
   for (Int_t i = 0; i < nints; i++)
      fInts.fArray[i] = i;
   static Int_t trackNumber = 0;
   trackNumber++;
   Int_t nch = 15;
   if (trackNumber > 100)
      nch += 3;
   if (trackNumber > 10000)
      nch += 3;
   // if (fTrackName) delete[] fTrackName;
   fTrackName = new char[nch];
   snprintf(fTrackName, nch, "Track%d", trackNumber);
   //Int_t i;
   //fHits.clear();
   //for (i=0;i<12;i++) fHits.push_back(i);

}

//______________________________________________________________________________
Track::~Track()
{
   delete[] fPoints;
   fPoints = 0;
   if (fTrackName)
      delete[] fTrackName;
   fTrackName = 0;
}

//______________________________________________________________________________
HistogramManager::HistogramManager(TDirectory *dir)
{
   // Create histogram manager object. Histograms will be created
   // in the "dir" directory.

   // Save current directory and cd to "dir".
   TDirectory *saved = gDirectory;
   dir->cd();

   fNtrack = new TH1F("hNtrack", "Ntrack", 100, 575, 625);
   fNseg = new TH1F("hNseg", "Nseg", 100, 5800, 6200);
   fTemperature = new TH1F("hTemperature", "Temperature", 100, 19.5, 20.5);
   fPx = new TH1F("hPx", "Px", 100, -4, 4);
   fPy = new TH1F("hPy", "Py", 100, -4, 4);
   fPz = new TH1F("hPz", "Pz", 100, 0, 5);
   fRandom = new TH1F("hRandom", "Random", 100, 0, 1000);
   fMass2 = new TH1F("hMass2", "Mass2", 100, 0, 12);
   fBx = new TH1F("hBx", "Bx", 100, -0.5, 0.5);
   fBy = new TH1F("hBy", "By", 100, -0.5, 0.5);
   fMeanCharge = new TH1F("hMeanCharge", "MeanCharge", 100, 0, 0.01);
   fXfirst = new TH1F("hXfirst", "Xfirst", 100, -40, 40);
   fXlast = new TH1F("hXlast", "Xlast", 100, -40, 40);
   fYfirst = new TH1F("hYfirst", "Yfirst", 100, -40, 40);
   fYlast = new TH1F("hYlast", "Ylast", 100, -40, 40);
   fZfirst = new TH1F("hZfirst", "Zfirst", 100, 0, 80);
   fZlast = new TH1F("hZlast", "Zlast", 100, 0, 250);
   fCharge = new TH1F("hCharge", "Charge", 100, -1.5, 1.5);
   fNpoint = new TH1F("hNpoint", "Npoint", 100, 50, 80);
   fValid = new TH1F("hValid", "Valid", 100, 0, 1.2);

   // cd back to original directory
   saved->cd();
}

//______________________________________________________________________________
HistogramManager::~HistogramManager()
{
   // Clean up all histograms.

   // Nothing to do. Histograms will be deleted when the directory
   // in which tey are stored is closed.
}

//______________________________________________________________________________
void HistogramManager::Hfill(Event *event)
{
   // Fill histograms.

   fNtrack->Fill(event->GetNtrack());
   fNseg->Fill(event->GetNseg());
   fTemperature->Fill(event->GetTemperature());

   for (Int_t itrack = 0; itrack < event->GetNtrack(); itrack++) {
      Track *track = (Track*)event->GetTracks()->UncheckedAt(itrack);
      fPx->Fill(track->GetPx());
      fPy->Fill(track->GetPy());
      fPz->Fill(track->GetPz());
      fRandom->Fill(track->GetRandom());
      fMass2->Fill(track->GetMass2());
      fBx->Fill(track->GetBx());
      fBy->Fill(track->GetBy());
      fMeanCharge->Fill(track->GetMeanCharge());
      fXfirst->Fill(track->GetXfirst());
      fXlast->Fill(track->GetXlast());
      fYfirst->Fill(track->GetYfirst());
      fYlast->Fill(track->GetYlast());
      fZfirst->Fill(track->GetZfirst());
      fZlast->Fill(track->GetZlast());
      fCharge->Fill(track->GetCharge());
      fNpoint->Fill(track->GetNpoint());
      fValid->Fill(track->GetValid());
   }
}
