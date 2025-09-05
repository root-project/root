#ifndef ROOT_Event
#define ROOT_Event

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Event                                                                //
//                                                                      //
// Description of the event and track parameters                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClonesArray.h"
#include "TH1.h"
#include "TAxis.h"
#include "TMath.h"
#include "TLine.h"
#include "TArrayF.h"
#include "TArrayI.h"
#include "TDatime.h"
#include "TRef.h"
#include "TLorentzVector.h"
#include <list>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <string>

#include <iostream>

//This is for portability of this test.  Do not try this at home!
using namespace std;

class TDirectory;

class UShortVector: public vector<UShort_t> {

private:
   Int_t fNshorts;   //length of STL std::vector (duplicated info)

public:
   UShortVector() {}
   UShortVector(Int_t n);
   virtual ~UShortVector() {}

   ClassDef(UShortVector,1)  //Encapsulated STL vector of UShorts
};

class EventHeader {

private:
   static Double_t fgNever;
public:
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
   //friend Bool_t  operator==(const EventHeader& h1, const EventHeader& h2);
   friend Bool_t  operator<=(const EventHeader& h1, const EventHeader& h2);

   ClassDef(EventHeader,1)  //Event Header
};
inline Bool_t     operator<=(const EventHeader& /* s1 */, const EventHeader& /* s2 */)
{ return 0; }

template <class T> struct template1 {};
template <class T> struct template2 {};

class Event : public TObject {
//class Event  {

private:
   enum {kSize=10};
   char                      fType[20];        //array of 20 chars
   char                     *fEventName;       //run+event number in character format
   Bool_t                    fBoolA;           //boolean flag
   Int_t                     fNtrack;          //number of tracks
   Int_t                     fNseg;            //number of segments
   Int_t                     fNvertex;         //number of vertices
   Int_t                     fMeasures[kSize]; //an array where dimension is an enum
   UInt_t                    fFlag;            //bit pattern event flag
   Float_t                   fMatrix[4][4];    //a two-dim array
   Float_t                  *fClosestDistance; //[fNvertex] pointer to an array of floats of length fNvertex
   Float_t                   fTemperature;     //event temperature
   char                     *fTracksInVertex;  //[fNvertex]
   vector<Long64_t>               fVectorLong64;       //STL vector on ints
   vector<Int_t>             fVectorint2;
   vector<int>               fVectorint;       //STL vector on ints
   vector<short>             fVectorshort;     //STL vector of shorts
   vector<double>            fVectorD[4];      //array of STL vectors of doubles
   vector<TLine>             fVectorTLine;     //|| array of STL vectors of TLine objects
   vector<TObject>          *fVectorTobject;   //|| pointer to an STL vector
   vector<TNamed>           *fVectorTnamed[6]; //|| array of pointers to STL vectors
   vector<string>            fLachaud;         //an STL vector of strings
   deque<TAttLine>           fDeque;           //STL deque
   list<const TObject*>      fVectorTobjectp;  //STL list of pointers to objects
   list<string>             *fListString;      //STL list of strings
   list<string *>            fListStringp;     //STL list of pointers to strings
   map<TNamed*,int>          fMapTNamedp;      //STL map

   template2< template1< int > > fTemplateMember;

   map<TString,TList*>       fMapList;         // STL map
   map<TString,TString*>     fMapTString;      //!STL map
   map<EventHeader,TString*> fMapHeaderP;      //!STL map
   map<EventHeader,TString>  fMapHeader;       //!STL map
   map<EventHeader,string>   fMapHeaderst;     //!STL map
   map<TAxis*,int>          *fMapTAxisp;       //pointer to STL map
   set<TAxis*>               fSetTAxis;        //STL set
   set<TAxis*>              *fSetTAxisp;       //pointer to STL set
   multimap<TNamed*,int>     fMultiMapTNamedp; //STL multimap
   multiset<TAxis*>         *fMultiSetTAxisp;  //pointer to STL multiset
   vector<vector<TAxis *> >  fVectAxis;       //!STL vector of vectors of TAxis*
   map<string,vector<int> >  fMapString;      //!STL map of string/vector
   deque<pair<float,float> > fDequePair;      //!STL deque of pair
   string                    fString;          //C++ standard string
   string                   *fStringp;         //pointer to standard C++ string
   TString                  *fTstringp;        //[fNvertex] array of TString
   TString                   fNames[12];       //array of TString
   TAxis                     fXaxis;           //example of class derived from TObject
   TAxis                     fYaxis[3];        //array of objects
   TAxis                    *fVaxis[3];        //pointer to an array of TAxis
   TAxis                    *fPaxis;           //[fNvertex] pointer to an array of TAxis of length fNvertex
   TAxis                   **fQaxis;           //[fNvertex] pointer to an array of pointers to TAxis objects
   TDatime                   fDatime;          //date and time
   EventHeader               fEvtHdr;          //example of class not derived from TObject
   TObjArray                 fObjArray;        //An object array of TObject*
   TClonesArray             *fTracks;          //-> array of tracks
   TH1F                     *fH;               //-> pointer to an histogram
   TLorentzVector            fLorentz;         //to test lorentzvector at top level
   TArrayF                   fArrayF;          //an array of floats
   TArrayI                  *fArrayI;          //a pointer to an array of integers
   UShortVector              fUshort;          //a TObject with an STL vector as base class
   TRef                      fRefH;            //Reference link to fH

   static TClonesArray      *fgTracks;
   static TH1F              *fgHist;

public:
                 Event();
                 Event(Int_t i);
   virtual      ~Event();
   void          Clear(Option_t *option ="") override;
   TDatime      &GetDatime() {return fDatime;}
   static void   Reset(Option_t *option ="");
   void          ResetHistogramPointer() {fH=0;}
   void          SetNseg(Int_t n) { fNseg = n; }
   void          SetNtrack(Int_t n) { fNtrack = n; }
   void          SetNvertex(Int_t n) { fNvertex = n; SetRandomVertex(); }
   void          SetFlag(UInt_t f) { fFlag = f; }
   void          SetTemperature(Float_t t) { fTemperature = t; }
   void          SetType(char *type) {strcpy(fType,type);}
   void          SetHeader(Int_t i, Int_t run, Int_t date, Float_t random);
   void          AddTrack(Float_t random);
   void          SetMeasure(UChar_t which, Int_t what);
   void          SetMatrix(UChar_t x, UChar_t y, Float_t what) { if (x<4&&y<4) fMatrix[x][y]=what;}
   void          SetRandomVertex();
   void          ShowLachaud();

   char         *GetType() {return fType;}
   Int_t         GetNtrack() const { return fNtrack; }
   Int_t         GetNseg() const { return fNseg; }
   Int_t         GetNvertex() const { return fNvertex; }
   UInt_t        GetFlag() const { return fFlag; }
   Float_t       GetTemperature() const { return fTemperature; }
   EventHeader  *GetHeader() { return &fEvtHdr; }
   TClonesArray *GetTracks() const { return fTracks; }
   TH1F         *GetHistogram() const { return fH; }
   Int_t         GetMeasure(UChar_t which) { return (which<10)?fMeasures[which]:0; }
   Float_t       GetMatrix(UChar_t x, UChar_t y) { return (x<4&&y<4)?fMatrix[x][y]:0; }

   UShortVector* GetUshort() { return &fUshort; }

   ClassDefOverride(Event,1)  //Event structure
};


class Track : public TObject {

private:
   Float_t      fPx;           //X component of the momentum
   Float_t      fPy;           //Y component of the momentum
   Float_t      fPz;           //Z component of the momentum
   Float_t      fRandom;       //A random track quantity
   Float_t      fMass2;        //The mass square of this particle
   Float_t      fBx;           //X intercept at the vertex
   Float_t      fBy;           //Y intercept at the vertex
   Float_t      fMeanCharge;   //Mean charge deposition of all hits of this track
   Float_t      fXfirst;       //X coordinate of the first point
   Float_t      fXlast;        //X coordinate of the last point
   Float_t      fYfirst;       //Y coordinate of the first point
   Float_t      fYlast;        //Y coordinate of the last point
   Float_t      fZfirst;       //Z coordinate of the first point
   Float_t      fZlast;        //Z coordinate of the last point
   vector<int>  fHits;         //list of hits
   Float_t      fCharge;       //Charge of this track
   Float_t      fVertex[3];    //Track vertex position
   Float_t      fCovar[3][4];  //Covariance matrix
   Float_t      fCovara[3][4][2];  //Covariance matrix
   Int_t        fNpoint;       //Number of points for this track
   Short_t      fValid;        //Validity criterion
   Short_t     *fPoints;       //[fNpoint] List of points
   char        *fTrackName;    //Track name
   TArrayI      fInts;         //some integers

public:
   Track() {fPoints=0; fTrackName = 0;}
   Track(Float_t random);
   virtual ~Track();
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
   Float_t       GetCharge() const { return fCharge; }
   Float_t       GetVertex(Int_t i=0) {return fVertex[i];}
   Float_t       GetCovar(Int_t i,Int_t j) {return fCovar[i][j];}
   Float_t       GetCovara(Int_t i,Int_t j,Int_t k) {return fCovara[i][j][k];}
   Int_t         GetNpoint() const { return fNpoint; }
   Short_t       GetValid()  const { return fValid; }
   virtual void  SetValid(Int_t valid=1) { fValid = valid; }

   ClassDefOverride(Track,1)  //A track segment
};

class BigTrack : public Track {

private:
   Int_t          fSpecial;    //The BigTrack validity flag
   TLorentzVector fKine;       //more kinematics

public:
   BigTrack() {fSpecial = 1234; }
   BigTrack(Float_t random, Int_t special);
   virtual ~BigTrack() { }

   ClassDefOverride(BigTrack,1)  //A Big track
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
