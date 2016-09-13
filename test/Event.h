#ifndef ROOT_Event
#define ROOT_Event

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Event                                                                //
//                                                                      //
// Description of the event and track parameters                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TRef.h"
#include "TH1.h"
#include "TBits.h"
#include "TMath.h"


class TDirectory;

class Track : public TObject {

private:
   Float_t      fPx;           //X component of the momentum
   Float_t      fPy;           //Y component of the momentum
   Float_t      fPz;           //Z component of the momentum
   Float_t      fRandom;       //A random track quantity
   Float16_t    fMass2;        //[0,0,8] The mass square of this particle
   Float16_t    fBx;           //[0,0,10] X intercept at the vertex
   Float16_t    fBy;           //[0,0,10] Y intercept at the vertex
   Float_t      fMeanCharge;   //Mean charge deposition of all hits of this track
   Float16_t    fXfirst;       //X coordinate of the first point
   Float16_t    fXlast;        //X coordinate of the last point
   Float16_t    fYfirst;       //Y coordinate of the first point
   Float16_t    fYlast;        //Y coordinate of the last point
   Float16_t    fZfirst;       //Z coordinate of the first point
   Float16_t    fZlast;        //Z coordinate of the last point
   Double32_t   fCharge;       //[-1,1,2] Charge of this track
   Double32_t   fVertex[3];    //[-30,30,16] Track vertex position
   Int_t        fNpoint;       //Number of points for this track
   Short_t      fValid;        //Validity criterion
   Int_t        fNsp;          //Number of points for this track with a special value
   Double32_t*  fPointValue;   //[fNsp][0,3] a special quantity for some point.
   TBits        fTriggerBits;  //Bits triggered by this track.

public:
   Track() : fTriggerBits(64) { fNsp = 0; fPointValue = 0; }
   Track(const Track& orig);
   Track(Float_t random);
   virtual ~Track() {Clear(); delete [] fPointValue; fPointValue = nullptr; }
   Track &operator=(const Track &orig);

   void          Set(Float_t random);
   void          Clear(Option_t *option="");
   Float_t       GetPx() const { return fPx; }
   Float_t       GetPy() const { return fPy; }
   Float_t       GetPz() const { return fPz; }
   Float_t       GetPt() const { return TMath::Sqrt(fPx*fPx + fPy*fPy); }
   Float_t       GetRandom() const { return fRandom; }
   Float_t       GetBx() const { return fBx; }
   Float_t       GetBy() const { return fBy; }
   Float_t       GetMass2() const { return fMass2; }
   Float_t       GetMeanCharge() const { return fMeanCharge; }
   Float_t       GetXfirst() const { return fXfirst; }
   Float_t       GetXlast()  const { return fXlast; }
   Float_t       GetYfirst() const { return fYfirst; }
   Float_t       GetYlast()  const { return fYlast; }
   Float_t       GetZfirst() const { return fZfirst; }
   Float_t       GetZlast()  const { return fZlast; }
   Double32_t    GetCharge() const { return fCharge; }
   Double32_t    GetVertex(Int_t i=0) {return (i<3)?fVertex[i]:0;}
   Int_t         GetNpoint() const { return fNpoint; }
   TBits&        GetTriggerBits() { return fTriggerBits; }
   Short_t       GetValid()  const { return fValid; }
   virtual void  SetValid(Int_t valid=1) { fValid = valid; }
   Int_t         GetN() const { return fNsp; }
   Double32_t    GetPointValue(Int_t i=0) const { return (i<fNsp)?fPointValue[i]:0; }

   ClassDef(Track,2)  //A track segment
};

class EventHeader {

private:
   Int_t   fEvtNum;
   Int_t   fRun;
   Int_t   fDate;

public:
   EventHeader() : fEvtNum(0), fRun(0), fDate(0) { }
   virtual ~EventHeader() { }
   void   Set(Int_t i, Int_t r, Int_t d) { fEvtNum = i; fRun = r; fDate = d; }
   Int_t  GetEvtNum() const { return fEvtNum; }
   Int_t  GetRun() const { return fRun; }
   Int_t  GetDate() const { return fDate; }

   ClassDef(EventHeader,1)  //Event Header
};


class Event : public TObject {

private:
   char           fType[20];          //event type
   char          *fEventName;         //run+event number in character format
   Int_t          fNtrack;            //Number of tracks
   Int_t          fNseg;              //Number of track segments
   Int_t          fNvertex;
   UInt_t         fFlag;
   Double32_t     fTemperature;
   Int_t          fMeasures[10];
   Double32_t     fMatrix[4][4];
   Double32_t    *fClosestDistance;   //[fNvertex][0,0,6]
   EventHeader    fEvtHdr;
   TClonesArray  *fTracks;            //->array with all tracks
   TRefArray     *fHighPt;            //array of High Pt tracks only
   TRefArray     *fMuons;             //array of Muon tracks only
   TRef           fLastTrack;         //reference pointer to last track
   TRef           fWebHistogram;      //EXEC:GetWebHistogram reference to an histogram in a TWebFile
   TH1F          *fH;                 //->
   TBits          fTriggerBits;       //Bits triggered by this event.
   Bool_t         fIsValid;           //

   static TClonesArray *fgTracks;
   static TH1F         *fgHist;

   Event(const Event&) = delete;
   Event &operator=(const Event&) = delete;

public:
   Event();
   virtual ~Event();
   void          Build(Int_t ev, Int_t arg5=600, Float_t ptmin=1);
   void          Clear(Option_t *option ="");
   Bool_t        IsValid() const { return fIsValid; }
   static void   Reset(Option_t *option ="");
   void          ResetHistogramPointer() {fH=0;}
   void          SetNseg(Int_t n) { fNseg = n; }
   void          SetNtrack(Int_t n) { fNtrack = n; }
   void          SetNvertex(Int_t n) { fNvertex = n; SetRandomVertex(); }
   void          SetFlag(UInt_t f) { fFlag = f; }
   void          SetTemperature(Double32_t t) { fTemperature = t; }
   void          SetType(char *type) {strcpy(fType,type);}
   void          SetHeader(Int_t i, Int_t run, Int_t date, Float_t random);
   Track        *AddTrack(Float_t random, Float_t ptmin=1);
   void          SetMeasure(UChar_t which, Int_t what);
   void          SetMatrix(UChar_t x, UChar_t y, Double32_t what) { if (x<3&&y<3) fMatrix[x][y]=what;}
   void          SetRandomVertex();

   Float_t       GetClosestDistance(Int_t i) {return fClosestDistance[i];}
   char         *GetType() {return fType;}
   Int_t         GetNtrack() const { return fNtrack; }
   Int_t         GetNseg() const { return fNseg; }
   Int_t         GetNvertex() const { return fNvertex; }
   UInt_t        GetFlag() const { return fFlag; }
   Double32_t    GetTemperature() const { return fTemperature; }
   EventHeader  *GetHeader() { return &fEvtHdr; }
   TClonesArray *GetTracks() const {return fTracks;}
   TRefArray    *GetHighPt() const {return fHighPt;}
   TRefArray    *GetMuons()  const {return fMuons;}
   Track        *GetLastTrack() const {return (Track*)fLastTrack.GetObject();}
   TH1F         *GetHistogram() const {return fH;}
   TH1          *GetWebHistogram()  const {return (TH1*)fWebHistogram.GetObject();}
   Int_t         GetMeasure(UChar_t which) { return (which<10)?fMeasures[which]:0; }
   Double32_t    GetMatrix(UChar_t x, UChar_t y) { return (x<4&&y<4)?fMatrix[x][y]:0; }
   TBits&        GetTriggerBits() { return fTriggerBits; }

   ClassDef(Event,1)  //Event structure
};


class HistogramManager {

private:
   TH1F  *fNtrack;
   TH1F  *fNseg;
   TH1F  *fTemperature;
   TH1F  *fPx;
   TH1F  *fPy;
   TH1F  *fPz;
   TH1F  *fRandom;
   TH1F  *fMass2;
   TH1F  *fBx;
   TH1F  *fBy;
   TH1F  *fMeanCharge;
   TH1F  *fXfirst;
   TH1F  *fXlast;
   TH1F  *fYfirst;
   TH1F  *fYlast;
   TH1F  *fZfirst;
   TH1F  *fZlast;
   TH1F  *fCharge;
   TH1F  *fNpoint;
   TH1F  *fValid;

public:
   HistogramManager(TDirectory *dir);
   virtual ~HistogramManager();

   void Hfill(Event *event);

   ClassDef(HistogramManager,1)  //Manages all histograms
};

#endif
