//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Mon Feb 23 09:53:52 2004 by ROOT version4.00/02)
//   from TTree T/An example of a ROOT tree
//   found on file: Event.new.split9.root
//////////////////////////////////////////////////////////


#ifndef gensel_h
#define gensel_h

// System Headers needed by the proxy
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelectorDraw.h>
#include <TPad.h>
#include <TProxy.h>
#include <TProxyDirector.h>
#include <TProxyTemplate.h>
using namespace ROOT;


// forward declarations needed by this particular proxy
class Event;
class TObject;
class EventHeader;
class TClonesArray;
class TRefArray;
class TSeqCollection;
class TCollection;
class TString;
class TProcessID;
class TNamed;
class TRef;
class TH1F;
class TH1;
class TAttLine;
class TAttFill;
class TAttMarker;
class TAxis;
class TAttAxis;
class TArrayD;
class THashList;
class TArray;
class TList;
class TArrayF;
class TBits;


// Header needed by this particular proxy
#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TSeqCollection.h"
#include "TCollection.h"
#include "TString.h"
#include "TProcessID.h"
#include "TNamed.h"
#include "TRef.h"
#include "TH1.h"
#include "TH1.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TAxis.h"
#include "TAttAxis.h"
#include "TArrayD.h"
#include "THashList.h"
#include "TArray.h"
#include "TList.h"
#include "TArrayF.h"
#include "TBits.h"
#include "script.h"


class gensel : public TSelector {
   public :
   TTree          *fChain;    //!pointer to the analyzed TTree or TChain
   TSelectorDraw  *fHelper;   //!helper class to create the default histogram
   TList          *fInput;    //!input list of the helper
   TH1            *htemp;     //!pointer to the histogram
   TProxyDirector  fDirector; //!Manages the proxys

   // Wrapper class for each unwounded class
   struct TPx_TObject
   {
      TPx_TObject(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix    (top,mid),
         obj         (director, top, mid),
         fUniqueID   (director, "fUniqueID"),
         fBits       (director, "fBits")
      {};
      TPx_TObject(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix    (""),
         obj         (director, parent, membername),
         fUniqueID   (director, "fUniqueID"),
         fBits       (director, "fBits")
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TObject* operator->() { return obj.ptr(); }
      TObjProxy<TObject > obj;

      TUIntProxy   fUniqueID;
      TUIntProxy   fBits;
   };
   struct TPx_EventHeader
   {
      TPx_EventHeader(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix   (top,mid),
         obj        (director, top, mid),
         fEvtNum    (director, ffPrefix, "fEvtNum"),
         fRun       (director, ffPrefix, "fRun"),
         fDate      (director, ffPrefix, "fDate")
      {};
      TPx_EventHeader(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix   (""),
         obj        (director, parent, membername),
         fEvtNum    (director, ffPrefix, "fEvtNum"),
         fRun       (director, ffPrefix, "fRun"),
         fDate      (director, ffPrefix, "fDate")
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      TProxy obj;

      TIntProxy   fEvtNum;
      TIntProxy   fRun;
      TIntProxy   fDate;
   };
   struct TClaPx_TBits
   {
      TClaPx_TBits(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix             (top,mid),
         obj                  (director, top, "", mid),
         fUniqueID            (director, ffPrefix, "fUniqueID"),
         fBits                (director, ffPrefix, "fBits"),
         fNbits               (director, ffPrefix, "fNbits"),
         fNbytes              (director, ffPrefix, "fNbytes"),
         fAllBits             (director, ffPrefix, "fAllBits")
      {};
      TClaPx_TBits(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix             (""),
         obj                  (director, parent, membername),
         fUniqueID            (director, ffPrefix, "fUniqueID"),
         fBits                (director, ffPrefix, "fBits"),
         fNbits               (director, ffPrefix, "fNbits"),
         fNbytes              (director, ffPrefix, "fNbytes"),
         fAllBits             (director, ffPrefix, "fAllBits")
      {};
      TProxyHelper          ffPrefix;
      InjectProxyInterface();
      const TBits* operator[](int i) { return obj.at(i); }
      TClaObjProxy<TBits > obj;

      TClaUIntProxy         fUniqueID;
      TClaUIntProxy         fBits;
      TClaUIntProxy         fNbits;
      TClaUIntProxy         fNbytes;
      TClaArrayUCharProxy   fAllBits;
   };
   struct TClaPx_Track
   {
      TClaPx_Track(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         fUniqueID            (director, ffPrefix, "fUniqueID"),
         fBits                (director, ffPrefix, "fBits"),
         fPx                  (director, ffPrefix, "fPx"),
         fPy                  (director, ffPrefix, "fPy"),
         fPz                  (director, ffPrefix, "fPz"),
         fRandom              (director, ffPrefix, "fRandom"),
         fMass2               (director, ffPrefix, "fMass2"),
         fBx                  (director, ffPrefix, "fBx"),
         fBy                  (director, ffPrefix, "fBy"),
         fMeanCharge          (director, ffPrefix, "fMeanCharge"),
         fXfirst              (director, ffPrefix, "fXfirst"),
         fXlast               (director, ffPrefix, "fXlast"),
         fYfirst              (director, ffPrefix, "fYfirst"),
         fYlast               (director, ffPrefix, "fYlast"),
         fZfirst              (director, ffPrefix, "fZfirst"),
         fZlast               (director, ffPrefix, "fZlast"),
         fCharge              (director, ffPrefix, "fCharge"),
         fVertex              (director, ffPrefix, "fVertex[3]"),
         fNpoint              (director, ffPrefix, "fNpoint"),
         fValid               (director, ffPrefix, "fValid"),
         fNsp                 (director, ffPrefix, "fNsp"),
         fPointValue          (director, ffPrefix, "fPointValue"),
         fTriggerBits         (director, ffPrefix, "fTriggerBits")
      {};
      TClaPx_Track(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix             (""),
         obj                  (director, parent, membername),
         fUniqueID            (director, ffPrefix, "fUniqueID"),
         fBits                (director, ffPrefix, "fBits"),
         fPx                  (director, ffPrefix, "fPx"),
         fPy                  (director, ffPrefix, "fPy"),
         fPz                  (director, ffPrefix, "fPz"),
         fRandom              (director, ffPrefix, "fRandom"),
         fMass2               (director, ffPrefix, "fMass2"),
         fBx                  (director, ffPrefix, "fBx"),
         fBy                  (director, ffPrefix, "fBy"),
         fMeanCharge          (director, ffPrefix, "fMeanCharge"),
         fXfirst              (director, ffPrefix, "fXfirst"),
         fXlast               (director, ffPrefix, "fXlast"),
         fYfirst              (director, ffPrefix, "fYfirst"),
         fYlast               (director, ffPrefix, "fYlast"),
         fZfirst              (director, ffPrefix, "fZfirst"),
         fZlast               (director, ffPrefix, "fZlast"),
         fCharge              (director, ffPrefix, "fCharge"),
         fVertex              (director, ffPrefix, "fVertex[3]"),
         fNpoint              (director, ffPrefix, "fNpoint"),
         fValid               (director, ffPrefix, "fValid"),
         fNsp                 (director, ffPrefix, "fNsp"),
         fPointValue          (director, ffPrefix, "fPointValue"),
         fTriggerBits         (director, ffPrefix, "fTriggerBits")
      {};
      TProxyHelper          ffPrefix;
      InjectProxyInterface();
      const TClonesArray* operator->() { return obj.ptr(); }
      TClaProxy obj;

      TClaUIntProxy         fUniqueID;
      TClaUIntProxy         fBits;
      TClaFloatProxy        fPx;
      TClaFloatProxy        fPy;
      TClaFloatProxy        fPz;
      TClaFloatProxy        fRandom;
      TClaFloatProxy        fMass2;
      TClaFloatProxy        fBx;
      TClaFloatProxy        fBy;
      TClaFloatProxy        fMeanCharge;
      TClaFloatProxy        fXfirst;
      TClaFloatProxy        fXlast;
      TClaFloatProxy        fYfirst;
      TClaFloatProxy        fYlast;
      TClaFloatProxy        fZfirst;
      TClaFloatProxy        fZlast;
      TClaFloatProxy        fCharge;
      TClaArrayFloatProxy   fVertex;
      TClaIntProxy          fNpoint;
      TClaShortProxy        fValid;
      TClaIntProxy          fNsp;
      TClaArrayFloatProxy   fPointValue;
      TClaPx_TBits          fTriggerBits;
   };
   struct TPx_TCollection
      : public TObjProxy<TObject >
   {
      TPx_TCollection(TProxyDirector* director,const char *top,const char *mid=0) :
         TObjProxy<TObject >  (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         fName                (director, obj.proxy(), "fName"),
         fSize                (director, obj.proxy(), "fSize")
      {};
      TPx_TCollection(TProxyDirector* director, TProxy *parent, const char *membername) :
         TObjProxy<TObject >  (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         fName                (director, obj.proxy(), "fName"),
         fSize                (director, obj.proxy(), "fSize")
      {};
      TProxyHelper          ffPrefix;
      InjectProxyInterface();
      const TCollection* operator->() { return obj.ptr(); }
      TObjProxy<TCollection > obj;

      TObjProxy<TString >   fName;
      TIntProxy             fSize;
   };
   struct TPx_TSeqCollection
      : public TPx_TCollection
   {
      TPx_TSeqCollection(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TCollection(director, top, mid),
         ffPrefix(top,mid),
         obj(director, top, mid)
      {};
      TPx_TSeqCollection(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TCollection(director, parent, membername),
         ffPrefix(""),
         obj(director, parent, membername)
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TSeqCollection* operator->() { return obj.ptr(); }
      TObjProxy<TSeqCollection > obj;

   };
   struct TPx_TNamed
      : public TObjProxy<TObject >
   {
      TPx_TNamed(TProxyDirector* director,const char *top,const char *mid=0) :
         TObjProxy<TObject >  (director, top, mid),
         ffPrefix             (top,mid),
         obj                  (director, top, mid),
         fName                (director, obj.proxy(), "fName"),
         fTitle               (director, obj.proxy(), "fTitle")
      {};
      TPx_TNamed(TProxyDirector* director, TProxy *parent, const char *membername) :
         TObjProxy<TObject >  (director, parent, membername),
         ffPrefix             (""),
         obj                  (director, parent, membername),
         fName                (director, obj.proxy(), "fName"),
         fTitle               (director, obj.proxy(), "fTitle")
      {};
      TProxyHelper          ffPrefix;
      InjectProxyInterface();
      const TNamed* operator->() { return obj.ptr(); }
      TObjProxy<TNamed > obj;

      TObjProxy<TString >   fName;
      TObjProxy<TString >   fTitle;
   };
   struct TPx_TProcessID
      : public TPx_TNamed
   {
      TPx_TProcessID(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TNamed(director, top, mid),
         ffPrefix(top,mid),
         obj(director, top, mid)
      {};
      TPx_TProcessID(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TNamed(director, parent, membername),
         ffPrefix(""),
         obj(director, parent, membername)
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TProcessID* operator->() { return obj.ptr(); }
      TObjProxy<TProcessID > obj;

   };
   struct TPx_TRefArray
      : public TPx_TSeqCollection
   {
      TPx_TRefArray(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TSeqCollection(director, top, mid),
         ffPrefix         (top,mid),
         obj              (director, top, mid),
         fPID             (director, obj.proxy(), "fPID"),
         fUIDs            (director, obj.proxy(), "fUIDs"),
         fLowerBound      (director, obj.proxy(), "fLowerBound"),
         fLast            (director, obj.proxy(), "fLast")
      {};
      TPx_TRefArray(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TSeqCollection(director, parent, membername),
         ffPrefix         (""),
         obj              (director, parent, membername),
         fPID             (director, obj.proxy(), "fPID"),
         fUIDs            (director, obj.proxy(), "fUIDs"),
         fLowerBound      (director, obj.proxy(), "fLowerBound"),
         fLast            (director, obj.proxy(), "fLast")
      {};
      TProxyHelper      ffPrefix;
      InjectProxyInterface();
      const TRefArray* operator->() { return obj.ptr(); }
      TObjProxy<TRefArray > obj;

      TPx_TProcessID    fPID;
      TArrayUIntProxy   fUIDs;
      TIntProxy         fLowerBound;
      TIntProxy         fLast;
   };
   struct TPx_TObject_1
   {
      TPx_TObject_1(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix    (top,mid),
         obj         (director, top, mid),
         fUniqueID   (director, obj.proxy(), "fUniqueID"),
         fBits       (director, obj.proxy(), "fBits")
      {};
      TPx_TObject_1(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix    (""),
         obj         (director, parent, membername),
         fUniqueID   (director, obj.proxy(), "fUniqueID"),
         fBits       (director, obj.proxy(), "fBits")
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TObject* operator->() { return obj.ptr(); }
      TObjProxy<TObject > obj;

      TUIntProxy   fUniqueID;
      TUIntProxy   fBits;
   };
   struct TPx_TRef
      : public TPx_TObject_1
   {
      TPx_TRef(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject_1(director, top, mid),
         ffPrefix(top,mid),
         obj(director, top, mid)
      {};
      TPx_TRef(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TObject_1(director, parent, membername),
         ffPrefix(""),
         obj(director, parent, membername)
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TRef* operator->() { return obj.ptr(); }
      TObjProxy<TRef > obj;

   };
   struct TPx_TAttLine
   {
      TPx_TAttLine(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         fLineColor   (director, obj.proxy(), "fLineColor"),
         fLineStyle   (director, obj.proxy(), "fLineStyle"),
         fLineWidth   (director, obj.proxy(), "fLineWidth")
      {};
      TPx_TAttLine(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix     (""),
         obj          (director, parent, membername),
         fLineColor   (director, obj.proxy(), "fLineColor"),
         fLineStyle   (director, obj.proxy(), "fLineStyle"),
         fLineWidth   (director, obj.proxy(), "fLineWidth")
      {};
      TProxyHelper  ffPrefix;
      InjectProxyInterface();
      const TAttLine* operator->() { return obj.ptr(); }
      TObjProxy<TAttLine > obj;

      TShortProxy   fLineColor;
      TShortProxy   fLineStyle;
      TShortProxy   fLineWidth;
   };
   struct TPx_TAttFill
   {
      TPx_TAttFill(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         fFillColor   (director, obj.proxy(), "fFillColor"),
         fFillStyle   (director, obj.proxy(), "fFillStyle")
      {};
      TPx_TAttFill(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix     (""),
         obj          (director, parent, membername),
         fFillColor   (director, obj.proxy(), "fFillColor"),
         fFillStyle   (director, obj.proxy(), "fFillStyle")
      {};
      TProxyHelper  ffPrefix;
      InjectProxyInterface();
      const TAttFill* operator->() { return obj.ptr(); }
      TObjProxy<TAttFill > obj;

      TShortProxy   fFillColor;
      TShortProxy   fFillStyle;
   };
   struct TPx_TAttMarker
   {
      TPx_TAttMarker(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix     (top,mid),
         obj          (director, top, mid),
         fMarkerColor (director, obj.proxy(), "fMarkerColor"),
         fMarkerStyle (director, obj.proxy(), "fMarkerStyle"),
         fMarkerSize  (director, obj.proxy(), "fMarkerSize")
      {};
      TPx_TAttMarker(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix     (""),
         obj          (director, parent, membername),
         fMarkerColor (director, obj.proxy(), "fMarkerColor"),
         fMarkerStyle (director, obj.proxy(), "fMarkerStyle"),
         fMarkerSize  (director, obj.proxy(), "fMarkerSize")
      {};
      TProxyHelper  ffPrefix;
      InjectProxyInterface();
      const TAttMarker* operator->() { return obj.ptr(); }
      TObjProxy<TAttMarker > obj;

      TShortProxy   fMarkerColor;
      TShortProxy   fMarkerStyle;
      TFloatProxy   fMarkerSize;
   };
   struct TPx_TAxis
      : public TObjProxy<TNamed >,
        public TObjProxy<TAttAxis >
   {
      TPx_TAxis(TProxyDirector* director,const char *top,const char *mid=0) :
         TObjProxy<TNamed >     (director, top, mid),
         TObjProxy<TAttAxis >   (director, top, mid),
         ffPrefix               (top,mid),
         obj                    (director, top, mid),
         fNbins                 (director, obj.proxy(), "fNbins"),
         fXmin                  (director, obj.proxy(), "fXmin"),
         fXmax                  (director, obj.proxy(), "fXmax"),
         fXbins                 (director, obj.proxy(), "fXbins"),
         fFirst                 (director, obj.proxy(), "fFirst"),
         fLast                  (director, obj.proxy(), "fLast"),
         fTimeDisplay           (director, obj.proxy(), "fTimeDisplay"),
         fTimeFormat            (director, obj.proxy(), "fTimeFormat"),
         fLabels                (director, obj.proxy(), "fLabels")
      {};
      TPx_TAxis(TProxyDirector* director, TProxy *parent, const char *membername) :
         TObjProxy<TNamed >     (director, parent, membername),
         TObjProxy<TAttAxis >   (director, parent, membername),
         ffPrefix               (""),
         obj                    (director, parent, membername),
         fNbins                 (director, obj.proxy(), "fNbins"),
         fXmin                  (director, obj.proxy(), "fXmin"),
         fXmax                  (director, obj.proxy(), "fXmax"),
         fXbins                 (director, obj.proxy(), "fXbins"),
         fFirst                 (director, obj.proxy(), "fFirst"),
         fLast                  (director, obj.proxy(), "fLast"),
         fTimeDisplay           (director, obj.proxy(), "fTimeDisplay"),
         fTimeFormat            (director, obj.proxy(), "fTimeFormat"),
         fLabels                (director, obj.proxy(), "fLabels")
      {};
      TProxyHelper            ffPrefix;
      InjectProxyInterface();
      const TAxis* operator->() { return obj.ptr(); }
      TObjProxy<TAxis > obj;

      TIntProxy               fNbins;
      TDoubleProxy            fXmin;
      TDoubleProxy            fXmax;
      TObjProxy<TArrayD >     fXbins;
      TIntProxy               fFirst;
      TIntProxy               fLast;
      TUCharProxy             fTimeDisplay;
      TObjProxy<TString >     fTimeFormat;
      TObjProxy<THashList >   fLabels;
   };
   struct TPx_TArrayD
      : public TObjProxy<TArray >
   {
      TPx_TArrayD(TProxyDirector* director,const char *top,const char *mid=0) :
         TObjProxy<TArray > (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         fArray             (director, obj.proxy(), "fArray")
      {};
      TPx_TArrayD(TProxyDirector* director, TProxy *parent, const char *membername) :
         TObjProxy<TArray > (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         fArray             (director, obj.proxy(), "fArray")
      {};
      TProxyHelper        ffPrefix;
      InjectProxyInterface();
      const TArrayD* operator->() { return obj.ptr(); }
      TObjProxy<TArrayD > obj;

      TArrayDoubleProxy   fArray;
   };
   struct TPx_TString
   {
      TPx_TString(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix    (top,mid),
         obj         (director, top, mid),
         fData       (director, obj.proxy(), "fData")
      {};
      TPx_TString(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix    (""),
         obj         (director, parent, membername),
         fData       (director, obj.proxy(), "fData")
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TString* operator->() { return obj.ptr(); }
      TObjProxy<TString > obj;

      TCharProxy   fData;
   };
   struct TPx_TList
      : public TObjProxy<TSeqCollection >
   {
      TPx_TList(TProxyDirector* director,const char *top,const char *mid=0) :
         TObjProxy<TSeqCollection >(director, top, mid),
         ffPrefix(top,mid),
         obj(director, top, mid)
      {};
      TPx_TList(TProxyDirector* director, TProxy *parent, const char *membername) :
         TObjProxy<TSeqCollection >(director, parent, membername),
         ffPrefix(""),
         obj(director, parent, membername)
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TList* operator->() { return obj.ptr(); }
      TObjProxy<TList > obj;

   };
   struct TPx_TH1
      : public TPx_TNamed,
        public TPx_TAttLine,
        public TPx_TAttFill,
        public TPx_TAttMarker
   {
      TPx_TH1(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TNamed         (director, top, mid),
         TPx_TAttLine       (director, top, mid),
         TPx_TAttFill       (director, top, mid),
         TPx_TAttMarker     (director, top, mid),
         ffPrefix           (top,mid),
         obj                (director, top, mid),
         fNcells            (director, obj.proxy(), "fNcells"),
         fXaxis             (director, obj.proxy(), "fXaxis"),
         fYaxis             (director, obj.proxy(), "fYaxis"),
         fZaxis             (director, obj.proxy(), "fZaxis"),
         fBarOffset         (director, obj.proxy(), "fBarOffset"),
         fBarWidth          (director, obj.proxy(), "fBarWidth"),
         fEntries           (director, obj.proxy(), "fEntries"),
         fTsumw             (director, obj.proxy(), "fTsumw"),
         fTsumw2            (director, obj.proxy(), "fTsumw2"),
         fTsumwx            (director, obj.proxy(), "fTsumwx"),
         fTsumwx2           (director, obj.proxy(), "fTsumwx2"),
         fMaximum           (director, obj.proxy(), "fMaximum"),
         fMinimum           (director, obj.proxy(), "fMinimum"),
         fNormFactor        (director, obj.proxy(), "fNormFactor"),
         fContour           (director, obj.proxy(), "fContour"),
         fSumw2             (director, obj.proxy(), "fSumw2"),
         fOption            (director, obj.proxy(), "fOption"),
         fFunctions         (director, obj.proxy(), "fFunctions"),
         fBufferSize        (director, obj.proxy(), "fBufferSize"),
         fBuffer            (director, obj.proxy(), "fBuffer")
      {};
      TPx_TH1(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TNamed         (director, parent, membername),
         TPx_TAttLine       (director, parent, membername),
         TPx_TAttFill       (director, parent, membername),
         TPx_TAttMarker     (director, parent, membername),
         ffPrefix           (""),
         obj                (director, parent, membername),
         fNcells            (director, obj.proxy(), "fNcells"),
         fXaxis             (director, obj.proxy(), "fXaxis"),
         fYaxis             (director, obj.proxy(), "fYaxis"),
         fZaxis             (director, obj.proxy(), "fZaxis"),
         fBarOffset         (director, obj.proxy(), "fBarOffset"),
         fBarWidth          (director, obj.proxy(), "fBarWidth"),
         fEntries           (director, obj.proxy(), "fEntries"),
         fTsumw             (director, obj.proxy(), "fTsumw"),
         fTsumw2            (director, obj.proxy(), "fTsumw2"),
         fTsumwx            (director, obj.proxy(), "fTsumwx"),
         fTsumwx2           (director, obj.proxy(), "fTsumwx2"),
         fMaximum           (director, obj.proxy(), "fMaximum"),
         fMinimum           (director, obj.proxy(), "fMinimum"),
         fNormFactor        (director, obj.proxy(), "fNormFactor"),
         fContour           (director, obj.proxy(), "fContour"),
         fSumw2             (director, obj.proxy(), "fSumw2"),
         fOption            (director, obj.proxy(), "fOption"),
         fFunctions         (director, obj.proxy(), "fFunctions"),
         fBufferSize        (director, obj.proxy(), "fBufferSize"),
         fBuffer            (director, obj.proxy(), "fBuffer")
      {};
      TProxyHelper        ffPrefix;
      InjectProxyInterface();
      const TH1* operator->() { return obj.ptr(); }
      TObjProxy<TH1 > obj;

      TIntProxy           fNcells;
      TPx_TAxis           fXaxis;
      TPx_TAxis           fYaxis;
      TPx_TAxis           fZaxis;
      TShortProxy         fBarOffset;
      TShortProxy         fBarWidth;
      TDoubleProxy        fEntries;
      TDoubleProxy        fTsumw;
      TDoubleProxy        fTsumw2;
      TDoubleProxy        fTsumwx;
      TDoubleProxy        fTsumwx2;
      TDoubleProxy        fMaximum;
      TDoubleProxy        fMinimum;
      TDoubleProxy        fNormFactor;
      TPx_TArrayD         fContour;
      TPx_TArrayD         fSumw2;
      TPx_TString         fOption;
      TPx_TList           fFunctions;
      TIntProxy           fBufferSize;
      TArrayDoubleProxy   fBuffer;
   };
   struct TPx_TArray
   {
      TPx_TArray(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix   (top,mid),
         obj        (director, top, mid),
         fN         (director, obj.proxy(), "fN")
      {};
      TPx_TArray(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix   (""),
         obj        (director, parent, membername),
         fN         (director, obj.proxy(), "fN")
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TArray* operator->() { return obj.ptr(); }
      TObjProxy<TArray > obj;

      TIntProxy   fN;
   };
   struct TPx_TArrayF
      : public TPx_TArray
   {
      TPx_TArrayF(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TArray        (director, top, mid),
         ffPrefix          (top,mid),
         obj               (director, top, mid),
         fArray            (director, obj.proxy(), "fArray")
      {};
      TPx_TArrayF(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TArray        (director, parent, membername),
         ffPrefix          (""),
         obj               (director, parent, membername),
         fArray            (director, obj.proxy(), "fArray")
      {};
      TProxyHelper       ffPrefix;
      InjectProxyInterface();
      const TArrayF* operator->() { return obj.ptr(); }
      TObjProxy<TArrayF > obj;

      TArrayFloatProxy   fArray;
   };
   struct TPx_TH1F
      : public TPx_TH1,
        public TPx_TArrayF
   {
      TPx_TH1F(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TH1(director, top, mid),
         TPx_TArrayF(director, top, mid),
         ffPrefix(top,mid),
         obj(director, top, mid)
      {};
      TPx_TH1F(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TH1(director, parent, membername),
         TPx_TArrayF(director, parent, membername),
         ffPrefix(""),
         obj(director, parent, membername)
      {};
      TProxyHelper ffPrefix;
      InjectProxyInterface();
      const TH1F* operator->() { return obj.ptr(); }
      TObjProxy<TH1F > obj;

   };
   struct TPx_TBits
   {
      TPx_TBits(TProxyDirector* director,const char *top,const char *mid=0) :
         ffPrefix          (top,mid),
         obj               (director, top, mid),
         fUniqueID         (director, ffPrefix, "fUniqueID"),
         fBits             (director, ffPrefix, "fBits"),
         fNbits            (director, ffPrefix, "fNbits"),
         fNbytes           (director, ffPrefix, "fNbytes"),
         fAllBits          (director, ffPrefix, "fAllBits")
      {};
      TPx_TBits(TProxyDirector* director, TProxy *parent, const char *membername) :
         ffPrefix          (""),
         obj               (director, parent, membername),
         fUniqueID         (director, ffPrefix, "fUniqueID"),
         fBits             (director, ffPrefix, "fBits"),
         fNbits            (director, ffPrefix, "fNbits"),
         fNbytes           (director, ffPrefix, "fNbytes"),
         fAllBits          (director, ffPrefix, "fAllBits")
      {};
      TProxyHelper       ffPrefix;
      InjectProxyInterface();
      const TBits* operator->() { return obj.ptr(); }
      TObjProxy<TBits > obj;

      TUIntProxy         fUniqueID;
      TUIntProxy         fBits;
      TUIntProxy         fNbits;
      TUIntProxy         fNbytes;
      TArrayUCharProxy   fAllBits;
   };
   struct TPx_Event
      : public TPx_TObject
   {
      TPx_Event(TProxyDirector* director,const char *top,const char *mid=0) :
         TPx_TObject               (director, top, mid),
         ffPrefix                  (top,mid),
         obj                       (director, top, mid),
         fType                     (director, "fType[20]"),
         fNtrack                   (director, "fNtrack"),
         fNseg                     (director, "fNseg"),
         fNvertex                  (director, "fNvertex"),
         fFlag                     (director, "fFlag"),
         fTemperature              (director, "fTemperature"),
         fMeasures                 (director, "fMeasures[10]"),
         fMatrix                   (director, "fMatrix[4][4]"),
         fClosestDistance          (director, "fClosestDistance"),
         fEvtHdr                   (director, "fEvtHdr"),
         fTracks                   (director, "fTracks"),
         fHighPt                   (director, "fHighPt"),
         fMuons                    (director, "fMuons"),
         fLastTrack                (director, "fLastTrack"),
         fWebHistogram             (director, "fWebHistogram"),
         fH                        (director, "fH"),
         fTriggerBits              (director, "fTriggerBits")
      {};
      TPx_Event(TProxyDirector* director, TProxy *parent, const char *membername) :
         TPx_TObject               (director, parent, membername),
         ffPrefix                  (""),
         obj                       (director, parent, membername),
         fType                     (director, "fType[20]"),
         fNtrack                   (director, "fNtrack"),
         fNseg                     (director, "fNseg"),
         fNvertex                  (director, "fNvertex"),
         fFlag                     (director, "fFlag"),
         fTemperature              (director, "fTemperature"),
         fMeasures                 (director, "fMeasures[10]"),
         fMatrix                   (director, "fMatrix[4][4]"),
         fClosestDistance          (director, "fClosestDistance"),
         fEvtHdr                   (director, "fEvtHdr"),
         fTracks                   (director, "fTracks"),
         fHighPt                   (director, "fHighPt"),
         fMuons                    (director, "fMuons"),
         fLastTrack                (director, "fLastTrack"),
         fWebHistogram             (director, "fWebHistogram"),
         fH                        (director, "fH"),
         fTriggerBits              (director, "fTriggerBits")
      {};
      TProxyHelper               ffPrefix;
      InjectProxyInterface();
      TProxy obj;

      TArrayCharProxy            fType;
      TIntProxy                  fNtrack;
      TIntProxy                  fNseg;
      TIntProxy                  fNvertex;
      TUIntProxy                 fFlag;
      TFloatProxy                fTemperature;
      TArrayIntProxy             fMeasures;
      TArray2Proxy<Float_t,4 >   fMatrix;
      TArrayFloatProxy           fClosestDistance;
      TPx_EventHeader            fEvtHdr;
      TClaPx_Track               fTracks;
      TPx_TRefArray              fHighPt;
      TPx_TRefArray              fMuons;
      TPx_TRef                   fLastTrack;
      TPx_TRef                   fWebHistogram;
      TPx_TH1F                   fH;
      TPx_TBits                  fTriggerBits;
   };


   // Proxy for each of the branches and leaves of the tree
   TPx_Event                  event;
   TPx_TObject                baseTObject;
   TArrayCharProxy            fType;
   TIntProxy                  fNtrack;
   TIntProxy                  fNseg;
   TIntProxy                  fNvertex;
   TUIntProxy                 fFlag;
   TFloatProxy                fTemperature;
   TArrayIntProxy             fMeasures;
   TArray2Proxy<Float_t,4 >   fMatrix;
   TArrayFloatProxy           fClosestDistance;
   TPx_EventHeader            fEvtHdr;
   TClaPx_Track               fTracks;
   TPx_TRefArray              fHighPt;
   TPx_TRefArray              fMuons;
   TPx_TRef                   fLastTrack;
   TPx_TRef                   fWebHistogram;
   TPx_TH1F                   fH;
   TPx_TBits                  fTriggerBits;


   gensel(TTree *tree=0) : 
      fChain(0),
      fHelper(0),
      fInput(0),
      htemp(0),
      fDirector(tree,-1),
      event                     (&fDirector,"event"),
      baseTObject               (&fDirector,"TObject"),
      fType                     (&fDirector,"fType[20]"),
      fNtrack                   (&fDirector,"fNtrack"),
      fNseg                     (&fDirector,"fNseg"),
      fNvertex                  (&fDirector,"fNvertex"),
      fFlag                     (&fDirector,"fFlag"),
      fTemperature              (&fDirector,"fTemperature"),
      fMeasures                 (&fDirector,"fMeasures[10]"),
      fMatrix                   (&fDirector,"fMatrix[4][4]"),
      fClosestDistance          (&fDirector,"fClosestDistance"),
      fEvtHdr                   (&fDirector,"fEvtHdr"),
      fTracks                   (&fDirector,"fTracks"),
      fHighPt                   (&fDirector,"fHighPt"),
      fMuons                    (&fDirector,"fMuons"),
      fLastTrack                (&fDirector,"fLastTrack"),
      fWebHistogram             (&fDirector,"fWebHistogram"),
      fH                        (&fDirector,"fH"),
      fTriggerBits              (&fDirector,"fTriggerBits")
      { }
   ~gensel();
   void    Begin(::TTree *tree);
   void    Init(::TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Int_t entry);
   Bool_t  ProcessCut(Int_t entry);
   void    ProcessFill(Int_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    Terminate();

   ClassDef(gensel,0);


//inject the user's code
#include "script.C"
};

#endif


#ifdef __MAKECINT__
#pragma link C++ class gensel::TPx_TObject-;
#pragma link C++ class gensel::TPx_EventHeader-;
#pragma link C++ class gensel::TClaPx_TBits-;
#pragma link C++ class gensel::TClaPx_Track-;
#pragma link C++ class gensel::TPx_TCollection-;
#pragma link C++ class gensel::TPx_TSeqCollection-;
#pragma link C++ class gensel::TPx_TNamed-;
#pragma link C++ class gensel::TPx_TProcessID-;
#pragma link C++ class gensel::TPx_TRefArray-;
#pragma link C++ class gensel::TPx_TObject_1-;
#pragma link C++ class gensel::TPx_TRef-;
#pragma link C++ class gensel::TPx_TAttLine-;
#pragma link C++ class gensel::TPx_TAttFill-;
#pragma link C++ class gensel::TPx_TAttMarker-;
#pragma link C++ class gensel::TPx_TAxis-;
#pragma link C++ class gensel::TPx_TArrayD-;
#pragma link C++ class gensel::TPx_TString-;
#pragma link C++ class gensel::TPx_TList-;
#pragma link C++ class gensel::TPx_TH1-;
#pragma link C++ class gensel::TPx_TArray-;
#pragma link C++ class gensel::TPx_TArrayF-;
#pragma link C++ class gensel::TPx_TH1F-;
#pragma link C++ class gensel::TPx_TBits-;
#pragma link C++ class gensel::TPx_Event-;
#pragma link C++ class gensel;
#endif


gensel::~gensel() {
   // destructor. Clean up helpers.

   delete fHelper;
   delete fInput;
}

void gensel::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain = tree;
   fDirector.SetTree(fChain);
   fHelper = new TSelectorDraw();
   fInput  = new TList();
   fInput->Add(new TNamed("varexp","0.0")); // Fake a double size histogram
   fInput->Add(new TNamed("selection",""));
   fHelper->SetInputList(fInput);
}

Bool_t gensel::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   fDirector.SetTree(fChain);
   
   return kTRUE;
}
   
void gensel::Begin(TTree *tree)
{
   // Function called before starting the event loop.
   // Initialize the tree branches.

   Init(tree);

   TString option = GetOption();
   fHelper->SetOption(option);
   fHelper->Begin(tree);
   htemp = (TH1*)fHelper->GetObject();
   htemp->SetTitle("script.C");
   fObject = htemp;

}

Bool_t gensel::Process(Int_t entry)
{
   // Processing function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).
   // Return kFALSE to stop processing.

   return kTRUE;
}

Bool_t gensel::ProcessCut(Int_t entry)
{
   // Selection function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // Return kFALSE as soon as a bad entry is detected.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).

   return kTRUE;
}

void gensel::ProcessFill(Int_t entry)
{
   // Function called for selected entries only.
   // Entry is the entry number in the current tree.
   // Read branches not processed in ProcessCut() and fill histograms.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).
   fDirector.SetReadEntry(entry);
   htemp->Fill(script());

}

void gensel::Terminate()
{
   // Function called at the end of the event loop.
   Int_t drawflag = (htemp && htemp->GetEntries()>0);
   
   if (!drawflag && !fOption.Contains("goff") && !fOption.Contains("same")) {
      gPad->Clear();
      return;
  }
   if (fOption.Contains("goff")) drawflag = false;
   if (drawflag) htemp->Draw(fOption);

}
