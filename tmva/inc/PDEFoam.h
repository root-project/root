// @(#)root/foam:$Name: not supported by cvs2svn $:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoam, PDEFoamCell, PDEFoamIntegrand, PDEFoamMaxwt, PDEFoamVect,    *
 *          TFDISTR                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Collection of helper classes to be used with MethodPDEFoam                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *                                                                                * 
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoam
#define ROOT_TMVA_PDEFoam

#include <iosfwd>

#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TH2D
#include "TH2D.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif
#ifndef ROOT_TVectorT
#include "TVectorT.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TRefArray
#include "TRefArray.h"
#endif
#ifndef ROOT_TMethodCall
#include "TMethodCall.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif
#ifndef ROOT_TRef
#include "TRef.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TRandom3;

namespace TMVA {
   class PDEFoam;
   class PDEFoamCell;
   class PDEFoamIntegrand;
   class PDEFoamVect;
   class PDEFoamMaxwt;
   class MsgLogger;

   enum EKernel { kNone, kGaus, kLinN };
   enum ETargetSelection { kMean, kMpv };
   enum ECellType { kAll, kActive, kInActive };
   enum EFoamType { kSeparate, kDiscr, kMonoTarget, kMultiTarget };
   // possible values, saved in foam cells
   // kNev           : number of events (saved in cell element 0)
   // kDiscriminator : discriminator (saved in cell element 0)
   // kTarget0       : target 0 (saved in cell element 0)
   // kMeanValue     : mean sampling value (saved in fIntegral)
   // kRms           : rms of sampling distribution (saved in fDriver)
   // kRmsOvMean     : rms/mean of sampling distribution (saved in fDriver and fIntegral)
   // kDensity       : number of events/cell volume
   enum ECellValue { kNev, kDiscriminator, kTarget0, kMeanValue, kRms, kRmsOvMean, kDensity };
   // options for filling density (used in Density() to build up foam)
   // kEVENT_DENSITY : use event density for foam buildup
   // kDISCRIMINATOR : use N_sig/(N_sig + N_bg) for foam buildup
   // kTARGET        : use GetTarget(0) for foam build up
   enum TDensityCalc { kEVENT_DENSITY, kDISCRIMINATOR, kTARGET };
}

//////////////////////////////////////////////////////////////
//                                                          //
// Class PDEFoamIntegrand is an Abstract class representing //
// n-dimensional real positive integrand function           //
//                                                          //
//////////////////////////////////////////////////////////////

namespace TMVA {

   class PDEFoamIntegrand : public ::TObject  {

   public:
      PDEFoamIntegrand();
      virtual ~PDEFoamIntegrand() { }
      virtual Double_t Density(Int_t ndim, Double_t *) = 0;
      virtual Double_t Density(Int_t ndim, Double_t *, Double_t &) = 0;

      ClassDef(PDEFoamIntegrand,1); //n-dimensional real positive integrand of FOAM
   }; // end of PDEFoamIntegrand
} // namespace TMVA


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TFDISTR is a child class of PDEFoamIntegrand and contains Methods to     //
// create a probability density by filling in events.                       //
// The main function is Density() which provides the event density at a     //
// given point during the foam build-up (sampling).                         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

namespace TMVA {

   // class definition of underlying density
   class TFDISTR : public PDEFoamIntegrand
   {
   private:
      Int_t fDim;               // number of dimensions 
      Float_t *fXmin;           //[fDim] minimal value of phase space in all dimension 
      Float_t *fXmax;           //[fDim] maximal value of phase space in all dimension 
      Float_t fVolFrac;         // volume fraction (with respect to total phase space
      BinarySearchTree *fBst;   // Binary tree to find events within a volume
      TDensityCalc fDensityCalc;// method of density calculation
     
   protected:
      Int_t fSignalClass;      // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      Int_t fBackgroundClass;  // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }                       

   public:
      TFDISTR();
      virtual ~TFDISTR();
     
      // Getter and setter for VolFrac option
      void SetVolumeFraction(Float_t vfr){fVolFrac=vfr; return;}
      Float_t GetVolumeFraction(){return fVolFrac;}
     
      // set foam dimension (mandatory before foam build-up!)
      void SetDim(Int_t idim){ 
         fDim = idim; 
	 if (fXmin) delete [] fXmin;
	 if (fXmax) delete [] fXmax;
         fXmin = new Float_t[fDim];
         fXmax = new Float_t[fDim];
         return;
      }
     
      // set foam boundaries
      void SetXmin(Int_t idim,Float_t wmin){fXmin[idim]=wmin; return;}
      void SetXmax(Int_t idim,Float_t wmax){fXmax[idim]=wmax; return;}
     
      // transformation functions for event variable into foam boundaries
      // reason: foam allways has boundaries [0, 1]
      Float_t VarTransform(Int_t idim, Float_t x){        // transform [xmin, xmax] --> [0, 1]
         Float_t b=fXmax[idim]-fXmin[idim]; 
         return (x-fXmin[idim])/b;
      }
      Float_t VarTransformInvers(Int_t idim, Float_t x){  // transform [0, 1] --> [xmin, xmax]
         Float_t b=fXmax[idim]-fXmin[idim]; 
         return x*b + fXmin[idim];
      }
     
      // debug function
      void PrintDensity();
     
      // density build-up functions
      void Initialize(Int_t ndim = 2);
      void FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights=kFALSE );
     
      // Dominik Dannheim 10.Jan.2008
      // new method to fill edge histograms directly, without MC sampling
      void FillEdgeHist(Int_t nDim, TObjArray *myfHistEdg , Double_t *cellPosi, Double_t *cellSize, Double_t *ceSum, Double_t ceVol);
     
      // main function used by PDEFoam
      // returns density at a given point by range searching in BST
      Double_t Density(int nDim, Double_t *Xarg, Double_t &event_density);
      Double_t Density(int nDim, Double_t *Xarg){ 
         Double_t event_density = 0; 
         return Density(nDim, Xarg, event_density);
      };

      // helper functions on BST
      UInt_t GetNEvents(PDEFoamCell* cell); // gets number of events in cell (from fBst)
      Bool_t CellRMSCut(PDEFoamCell* cell, Float_t rms_cut, Int_t nbin); // calc cell rms and compare with rms_cut

      // Getters and setters for foam filling method
      void SetDensityCalc( TDensityCalc dc ){ fDensityCalc = dc; };
      Bool_t FillDiscriminator(){ return fDensityCalc == kDISCRIMINATOR; }
      Bool_t FillTarget0()      { return fDensityCalc == kTARGET;        }
      Bool_t FillEventDensity() { return fDensityCalc == kEVENT_DENSITY; }
     
      void SetSignalClass( Int_t cls )     { fSignalClass = cls;  } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      void SetBackgroundClass( Int_t cls ) { fBackgroundClass = cls;  } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      TH2D* MakeHistogram(Int_t nbinsx, Int_t nbinsy);
     
      ClassDef(TFDISTR,3) //Class for Event density
         };  //end of TFDISTR
     
}  // namespace TMVA

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// PDEFoam is the child class of the multi-dimensional general purpose      //
// Monte Carlo event generator (integrator) PDEFoam.                          //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

namespace TMVA {
   
   class PDEFoamCell;
   class PDEFoam;

   std::ostream& operator<< ( std::ostream& os, const PDEFoam& pdefoam );
   std::istream& operator>> ( std::istream& istr,     PDEFoam& pdefoam );
   
   class PDEFoam : public TObject {
   protected:
      // COMPONENTS //
      //-------------- Input parameters
      TString fName;             // Name of a given instance of the FOAM class
      TString fVersion;          // Actual version of the FOAM like (1.01m)
      TString fDate;             // Release date of FOAM
      Int_t   fDim;              // Dimension of the integration/simulation space
      Int_t   fNCells;           // Maximum number of cells
      Int_t   fRNmax;            // Maximum No. of the rand. numb. requested at once
      //-------------------
      Int_t   fOptDrive;         // Optimization switch =1,2 for variance or maximum weight optimization
      Int_t   fChat;             // Chat=0,1,2 chat level in output, Chat=1 normal level
      Int_t   fOptRej;           // Switch =0 for weighted events; =1 for unweighted events in MC
      //-------------------
      Int_t   fNBin;             // No. of bins in the edge histogram for cell MC exploration
      Int_t   fNSampl;           // No. of MC events, when dividing (exploring) cell
      Int_t   fEvPerBin;         // Maximum number of effective (wt=1) events per bin
      //-------------------  MULTI-BRANCHING ---------------------
      Int_t  *fMaskDiv;          //! [fDim] Dynamic Mask for  cell division
      Int_t  *fInhiDiv;          //! [fDim] Flags for inhibiting cell division
      Int_t   fOptPRD;           //  Option switch for predefined division, for quick check
      PDEFoamVect **fXdivPRD;    //! Lists of division values encoded in one vector per direction
      //-------------------  GEOMETRY ----------------------------
      Int_t   fNoAct;            // Number of active cells
      Int_t   fLastCe;           // Index of the last cell
      PDEFoamCell **fCells;      // [fNCells] Array of ALL cells
      //------------------ M.C. generation----------------------------
      PDEFoamMaxwt   *fMCMonit;  // Monitor of the MC weight for measuring MC efficiency
      Double_t   fMaxWtRej;      // Maximum weight in rejection for getting wt=1 events
      TRefArray *fCellsAct;      // Array of pointers to active cells, constructed at the end of foam build-up
      Double_t  *fPrimAcu;       // [fNoAct] Array of cumulative probability of all active cells
      TObjArray *fHistEdg;       // Histograms of wt, one for each cell edge
      TObjArray *fHistDbg;       // Histograms of wt, for debug
      TH1D      *fHistWt;        // Histogram of the MC wt

      Double_t *fMCvect;         // [fDim] Generated MC vector for the outside user
      Double_t  fMCwt;           // MC weight
      Double_t *fRvec;           // [fRNmax] random number vector from r.n. generator fDim+1 maximum elements
      //----------- Procedures
      PDEFoamIntegrand *fRho;    //! Pointer to the user-defined integrand function/distribution
      TMethodCall *fMethodCall;  //! ROOT's pointer to user-defined global distribution function
      TRandom3        *fPseRan;  // Pointer to user-defined generator of pseudorandom numbers
      //----------- Statistics and MC results
      Long_t   fNCalls;          // Total number of the function calls
      Long_t   fNEffev;          // Total number of effective events (wt=1) in the foam buildup
      Double_t fSumWt, fSumWt2;  // Total sum of wt and wt^2
      Double_t fSumOve;          // Total Sum of overveighted events
      Double_t fNevGen;          // Total number of the generated MC events
      Double_t fWtMax, fWtMin;   // Maximum/Minimum MC weight
      Double_t fPrime;           // Primary integral R' (R=R'<wt>)
      Double_t fMCresult;        // True Integral R from MC series
      Double_t fMCerror;         // and its error
      //----------  working space for CELL exploration -------------
      Double_t *fAlpha;          // [fDim] Internal parameters of the hyperrectangle
      // ---------  PDE-Foam specific variables
      Double_t *fXmin;         // [fDim] minimum for variable transform
      Double_t *fXmax;         // [fDim] maximum for variable transform
      UInt_t fNElements;       // number of variables in every cell
      Bool_t fCutNmin;         // true: activate cut on minimal number of events in cell
      UInt_t fNmin;            // minimal number of events in cell to split cell
      Bool_t fCutRMSmin;       // true:  peek cell with max. RMS for next split
      Double_t fRMSmin;        // activate cut: minimal RMS in cell to split cell
      Float_t fVolFrac;        // volume fraction (with respect to total phase space
      TFDISTR *fDistr;         //! density from extern
      Timer *fTimer;           // timer for graphical output
      TObjArray *fVariableNames;// collection of all variable names
      Int_t fSignalClass;      // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      Int_t fBackgroundClass;  // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }                       

      //////////////////////////////////////////////////////////////////////////////////////////////
      //                                     METHODS                                              //
      //////////////////////////////////////////////////////////////////////////////////////////////
   private:
      Double_t Sqr(Double_t x) const { return x*x;}      // Square function

   protected:
      // override  PDEFoam::OutputGrow(Bool_t) for nicer TMVA output
      virtual void OutputGrow(Bool_t finished = false ); 

      // weight result from function with kernel
      Double_t WeightGaus(PDEFoamCell*, std::vector<Float_t>, UInt_t dim=0); 
      Double_t WeightLinNeighbors( std::vector<Float_t> txvec, ECellValue cv );

   public:
      PDEFoam();                  // Default constructor (used only by ROOT streamer)
      PDEFoam(const TString&);    // Principal user-defined constructor
      virtual ~PDEFoam();         // Default destructor
      PDEFoam(const PDEFoam&);    // Copy Constructor  NOT USED

      // Initialization
      virtual void Initialize(Bool_t CreateCellElements);     // Initialisation of foam
      virtual void Initialize(TRandom3*, PDEFoamIntegrand *); // Alternative initialization method, backward compatibility
      virtual void InitCells(Bool_t CreateCellElements);      // Initialisation of all foam cells
      virtual Int_t CellFill(Int_t, PDEFoamCell*);// Allocates new empty cell and return its index
      virtual void Explore(PDEFoamCell *Cell);    // Exploration of the new cell, determine <wt>, wtMax etc.
      virtual void Carver(Int_t&,Double_t&,Double_t&);// Determines the best edge, wt_max reduction
      virtual void Varedu(Double_t [], Int_t&, Double_t&,Double_t&); // Determines the best edge, variace reduction
      virtual void MakeAlpha();                 // Provides random point inside hyperrectangle
      virtual void Grow();                      // build up foam
      virtual Long_t PeekMax();                 // peek cell with max. driver integral
      virtual Int_t  Divide(PDEFoamCell *);     // Divide iCell into two daughters; iCell retained, taged as inactive
      virtual void MakeActiveList();            // Creates table of active cells
      virtual void GenerCel2(PDEFoamCell *&);   // Chose an active cell the with probability ~ Primary integral
      // Generation
      virtual Double_t Eval(Double_t *xRand, Double_t &event_density); // evaluate distribution on point 'xRand'
      virtual Double_t Eval(Double_t *xRand){   // evaluate distribution on point 'xRand'
	 Double_t event_density = 0;
	 return Eval(xRand, event_density);
      };
      virtual void     MakeEvent();             // Makes (generates) single MC event
      virtual void     GetMCvect(Double_t *);   // Provides generated randomly MC vector
      virtual void     GetMCwt(Double_t &);     // Provides generated MC weight
      virtual Double_t GetMCwt();               // Provides generates MC weight
      virtual Double_t MCgenerate(Double_t *MCvect);// All three above function in one
      // Finalization
      virtual void GetIntegMC(Double_t&, Double_t&);  // Provides Integrand and abs. error from MC run
      virtual void GetIntNorm(Double_t&, Double_t&);  // Provides normalization Inegrand
      virtual void GetWtParams(Double_t, Double_t&, Double_t&, Double_t&);// Provides MC weight parameters
      virtual void Finalize(  Double_t&, Double_t&);    // Prints summary of MC integration
      virtual PDEFoamIntegrand  *GetRho(){return fRho;} // Gets pointer of the distribut. (after restoring from disk)
      virtual TRandom3*GetPseRan() const {return fPseRan;}   // Gets pointer of r.n. generator (after restoring from disk)
      virtual void SetRhoInt(void *Rho);              // Set new integrand distr. in interactive mode
      virtual void SetRho(PDEFoamIntegrand *Rho);     // Set new integrand distr. in compiled mode
      virtual void ResetRho(PDEFoamIntegrand *Rho);              // Set new distribution, delete old
      virtual void SetPseRan(TRandom3*PseRan){fPseRan=PseRan;}   // Set new r.n. generator
      virtual void ResetPseRan(TRandom3*PseRan);                 // Set new r.n.g, delete old
      // Getters and Setters
      virtual void SetkDim(Int_t kDim){ // Sets dimension of cubical space
	 fDim  = kDim;
	 if (fXmin) delete [] fXmin;
	 if (fXmax) delete [] fXmax;
         fXmin = new Double_t[GetTotDim()];
         fXmax = new Double_t[GetTotDim()];
      }
      virtual void SetnCells(Long_t nCells){fNCells =nCells;}  // Sets maximum number of cells
      virtual void SetnSampl(Long_t nSampl){fNSampl =nSampl;}  // Sets no of MC events in cell exploration
      virtual void SetnBin(Int_t nBin){fNBin = nBin;}          // Sets no of bins in histogs in cell exploration
      virtual void SetChat(Int_t Chat){fChat = Chat;}          // Sets option Chat, chat level
      virtual void SetOptRej(Int_t OptRej){fOptRej =OptRej;}   // Sets option for MC rejection
      virtual void SetOptDrive(Int_t OptDrive){fOptDrive =OptDrive;}   // Sets optimization switch
      virtual void SetEvPerBin(Int_t EvPerBin){fEvPerBin =EvPerBin;}   // Sets max. no. of effective events per bin
      virtual void SetMaxWtRej(Double_t MaxWtRej){fMaxWtRej=MaxWtRej;} // Sets max. weight for rejection
      virtual void SetInhiDiv(Int_t, Int_t );            // Set inhibition of cell division along certain edge
      virtual void SetXdivPRD(Int_t, Int_t, Double_t[]); // Set predefined division points
      // Getters and Setters
      virtual const char *GetVersion() const {return fVersion.Data();}// Get version of the FOAM
      virtual Int_t    GetTotDim() const { return fDim;}              // Get total dimension
      virtual Double_t GetPrimary() const {return fPrime;}            // Get value of primary integral R'
      virtual void GetPrimary(Double_t &prime) {prime = fPrime;}      // Get value of primary integral R'
      virtual Long_t GetnCalls() const {return fNCalls;}              // Get total no. of the function calls
      virtual Long_t GetnEffev() const {return fNEffev;}              // Get total no. of effective wt=1 events
      virtual TString GetFoamName() const {return fName;}             // Get name of foam
      // Debug
      virtual void CheckAll(Int_t);     // Checks correctness of the entire data structure in the FOAM object
      virtual void PrintCells();        // Prints content of all cells
      virtual void LinkCells(void);     // Void function for backward compatibility

      // getter and setter to activate cut options
      void   CutNmin(Bool_t cut )   { fCutNmin = cut;    }
      Bool_t CutNmin()              { return fCutNmin;   }
      void   CutRMSmin(Bool_t cut ) { fCutRMSmin = cut;  }
      Bool_t CutRMSmin()            { return fCutRMSmin; }

      // getter and setter for cut values
      void     SetNmin(UInt_t val)     { fNmin=val;      }
      UInt_t   GetNmin()               { return fNmin;   }
      void     SetRMSmin(Double_t val) { fRMSmin=val;    }
      Double_t GetRMSmin()             { return fRMSmin; }

      // foam output operators
      friend std::ostream& operator<< ( std::ostream& os, const PDEFoam& pdefoam );
      friend std::istream& operator>> ( std::istream& istr,     PDEFoam& pdefoam );

      void ReadStream(istream &);         // read foam from stream
      void PrintStream(ostream  &) const; // write foam from stream
      void ReadXML( void* parent );       // read foam variables from xml
      void AddXMLTo( void* parent );      // write foam variables to xml
      
      // getters/ setters for foam boundaries
      void SetXmin(Int_t idim, Double_t wmin){ 
         fXmin[idim]=wmin;
         fDistr->SetXmin(idim, wmin);
         return; 
      }
      void SetXmax(Int_t idim, Double_t wmax){ 
         fXmax[idim]=wmax;
         fDistr->SetXmax(idim, wmax);
         return; 
      }
      Double_t GetXmin(Int_t idim){return fXmin[idim];}
      Double_t GetXmax(Int_t idim){return fXmax[idim];}

      // getter/ setter for variable name
      void AddVariableName(const char *s){
	 TObjString *os = new TObjString(s);
	 AddVariableName(os);
      };
      void AddVariableName(TObjString *s){
	 fVariableNames->Add(s);
      };
      TObjString* GetVariableName(Int_t idx){
	 return dynamic_cast<TObjString*>(fVariableNames->At(idx));
      };

      // transformation functions for event variable into foam boundaries
      // reason: foam allways has boundaries [0, 1]
      Float_t VarTransform(Int_t idim, Float_t x){        // transform [xmin, xmax] --> [0, 1]
         Float_t b=fXmax[idim]-fXmin[idim]; 
         return (x-fXmin[idim])/b;
      }
      std::vector<Float_t> VarTransform(std::vector<Float_t> invec){
         std::vector<Float_t> outvec;
         for(UInt_t i=0; i<invec.size(); i++)
            outvec.push_back(VarTransform(i, invec.at(i)));
         return outvec;
      }
      Float_t VarTransformInvers(Int_t idim, Float_t x){  // transform [0, 1] --> [xmin, xmax]
         Float_t b=fXmax[idim]-fXmin[idim]; 
         return x*b + fXmin[idim];
      }
      std::vector<Float_t> VarTransformInvers(std::vector<Float_t> invec){
         std::vector<Float_t> outvec;
         for(UInt_t i=0; i<invec.size(); i++)
            outvec.push_back(VarTransformInvers(i, invec.at(i)));
         return outvec;
      }
      
      // projection method
      virtual TH2D* Project2(Int_t idim1, Int_t idim2, const char *opt="nev", const char *ker="kNone", UInt_t maxbins=0);

      // Project foam by creating MC events
      virtual TH2D* ProjectMC(Int_t idim1, Int_t idim2, Int_t nevents, Int_t nbin);  

      // Draw a 1-dim histogram
      virtual TH1D* Draw1Dim(const char *opt, Int_t nbin);

      // Generates C++ code (root macro) for drawing foam with boxes (only 2-dim!)
      virtual void RootPlot2dim( const TString& filename, std::string what, 
                                 Bool_t CreateCanvas = kTRUE, Bool_t colors = kTRUE, Bool_t log_colors = kFALSE  );

      // init TObject pointer on cells
      void ResetCellElements(Bool_t allcells = false);

      // low level functions to access a certain cell value
      TVectorD* GetCellElements(std::vector<Float_t>);       // return cell elements of cell with given coordinates
      Double_t GetCellElement(PDEFoamCell *cell, UInt_t i);  // get Element 'i' in cell 'cell'
      void SetCellElement(PDEFoamCell *cell, UInt_t i, Double_t value); // set Element 'i' in cell 'cell' to value 'value'
      void SetNElements(UInt_t numb){ fNElements = numb; }   // init every cell element (TVectorD*)
      UInt_t GetNElements(){ return fNElements; }            // returns number of elements, saved on every cell
      
      void DisplayCellContent(void); // debug function

      // functions to fill created cells with certain values
      void FillFoamCells(const Event* ev, EFoamType ft, Bool_t NoNegWeights=kFALSE);
      
      // functions to calc discriminators/ mean targets for every cell
      // using filled cell values
      void CalcCellDiscr();
      void CalcCellTarget();

      // helper functions to access cell data
      Double_t GetCellMean(std::vector<Float_t> xvec);
      Double_t GetCellRMS(std::vector<Float_t> xvec);
      Double_t GetCellEntries(std::vector<Float_t> xvec);
      Double_t GetCellDiscr(std::vector<Float_t> xvec, EKernel kernel=kNone);
      Double_t GetCellDiscrError(std::vector<Float_t> xvec);
      Double_t GetCellDensity(std::vector<Float_t> xvec, EKernel kernel=kNone);

      Double_t GetCellMean(PDEFoamCell* cell);
      Double_t GetCellRMS(PDEFoamCell* cell);
      Double_t GetCellRMSovMean(PDEFoamCell* cell);
      Double_t GetCellEntries(PDEFoamCell* cell);
      Double_t GetCellEvents(PDEFoamCell* cell);
      Double_t GetCellDiscr(PDEFoamCell* cell);
      Double_t GetCellDiscrError(PDEFoamCell* cell);
      Double_t GetCellRegValue0(PDEFoamCell* cell); // returns regression value (regression method 1)
      Double_t GetCellDensity(PDEFoamCell*);

      // returns regression value (mono target regression)
      Double_t GetCellRegValue0(std::vector<Float_t>, EKernel kernel=kNone);
      // returns regression value i, given all variables (multi target regression)
      Double_t GetProjectedRegValue(UInt_t i, std::vector<Float_t> vals, EKernel kernel=kNone, ETargetSelection ts=kMean);
      Double_t GetCellTarget(UInt_t target_number, std::vector<Float_t> tvals, ETargetSelection ts);

      // helper/ debug functions
      Double_t GetSumCellIntg();           // test function. calculates sum over all cell->GetIntg()
      Double_t GetSumCellElements(UInt_t); // test function. calculates sum over all cell->GetCellElement()
      UInt_t   GetNActiveCells();          // returns number of active cells
      UInt_t   GetNInActiveCells();        // returns number of not active cells
      UInt_t   GetNCells();                // returns number of cells
      PDEFoamCell* GetRootCell();          // get pointer to root cell
      Int_t    GetSumCellMemory(ECellType ct = kAll); // return memory consumtion of all cells
      void     CheckCells(Bool_t remove_empty_cells=false); // check all cells with respect to critical values
      void     RemoveEmptyCell(Int_t iCell); // removes iCell if its volume is zero
      void     PrintCellElements();          // print all cells with its elements

      // find cell according to given event variables
      PDEFoamCell* FindCell(std::vector<Float_t>); //!

      // helper functions to include TFDISTR into the PDEFoam class
      // set VolFrac to PDEFoam
      void SetPDEFoamVolumeFraction( Double_t vfr) { fVolFrac = vfr; }
      // get VolFrac from PDEFoam
      Double_t GetPDEFoamVolumeFraction() const { return fVolFrac; }
      void SetVolumeFraction(Double_t);    // set VolFrac to TFDISTR
      void FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights=kFALSE );
      void Create(Bool_t CreateCellElements=false); // create PDEFoam
      void Init();                    // initialize TFDISTR
      void PrintDensity();            // debug output
      void SetFoamType(EFoamType ft);

      void SetSignalClass( Int_t cls )     { fSignalClass = cls; fDistr->SetSignalClass( cls ); } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      void SetBackgroundClass( Int_t cls ) { fBackgroundClass = cls; fDistr->SetBackgroundClass( cls ); } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      //////////////////////////////////////////////////////////////////////////////////////////////
      ClassDef(PDEFoam,3)   // General purpose self-adapting binning
   }; // end of PDEFoam 

}  // namespace TMVA


////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
// Class PDEFoamCell used in PDEFoam                                              //
//                                                                                //
// Objects of this class are hyperrectangular cells organized in the binary tree. //
// Special algoritm for encoding relalive positioning of the cells                //
// saves total memory allocation needed for the system of cells.                  //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////

namespace TMVA {

   class PDEFoamVect;

   class PDEFoamCell : public TObject {

      //   static, the same for all cells!
   private:
      Short_t  fDim;                   // Dimension of the vector space
      //   MEMBERS

   private:
      //--- linked tree organization ---
      Int_t    fSerial;                // Serial number
      Int_t    fStatus;                // Status (active, inactive)
      TRef     fParent;                // Pointer to parent cell
      TRef     fDaught0;               // Pointer to daughter 1
      TRef     fDaught1;               // Pointer to daughter 2
      //--- M.C. sampling and choice of the best edge ---

   private:
      Double_t fXdiv;                  // Factor for division
      Int_t    fBest;                  // Best Edge for division
      //--- Integrals of all kinds ---
      Double_t fVolume;                // Cartesian Volume of cell
      Double_t fIntegral;              // Integral over cell (estimate from exploration)
      Double_t fDrive;                 // Driver  integral, only for cell build-up
      Double_t fPrimary;               // Primary integral, only for MC generation
      //----------  working space for the user --------------
      TObject *fElement;               // may set by the user to save some data in this cell

      //////////////////////////////////////////////////////////////////////////////////////
      //                           METHODS                                                //
      //////////////////////////////////////////////////////////////////////////////////////
   public:
      PDEFoamCell();                          // Default Constructor for ROOT streamers
      PDEFoamCell(Int_t);                     // User Constructor
      PDEFoamCell(PDEFoamCell &);             // Copy Constructor
      virtual ~PDEFoamCell();                 // Destructor
      void  Fill(Int_t, PDEFoamCell*, PDEFoamCell*, PDEFoamCell*);    // Assigns values of attributes
      PDEFoamCell&  operator=(const PDEFoamCell&);       // Substitution operator (never used)
      //--------------- Geometry ----------------------------------
      Double_t  GetXdiv() const { return fXdiv;}          // Pointer to Xdiv
      Int_t     GetBest() const { return fBest;}          // Pointer to Best
      void      SetBest(Int_t    Best){ fBest =Best;}     // Set Best edge candidate
      void      SetXdiv(Double_t Xdiv){ fXdiv =Xdiv;}     // Set x-division for best edge cand.
      void      GetHcub(  PDEFoamVect&, PDEFoamVect&) const;  // Get position and size vectors (h-cubical subspace)
      void      GetHSize( PDEFoamVect& ) const;             // Get size only of cell vector  (h-cubical subspace)
      //--------------- Integrals/Volumes -------------------------
      void      CalcVolume();                             // Calculates volume of cell
      Double_t  GetVolume() const { return fVolume;}      // Volume of cell
      Double_t  GetIntg() const { return fIntegral;}      // Get Integral
      Double_t  GetDriv() const { return fDrive;}         // Get Drive
      Double_t  GetPrim() const { return fPrimary;}       // Get Primary
      void      SetIntg(Double_t Intg){ fIntegral=Intg;}  // Set true integral
      void      SetDriv(Double_t Driv){ fDrive   =Driv;}  // Set driver integral
      void      SetPrim(Double_t Prim){ fPrimary =Prim;}  // Set primary integral
      //--------------- linked tree organization ------------------
      Int_t     GetStat() const { return fStatus;}        // Get Status
      void      SetStat(Int_t Stat){ fStatus=Stat;}       // Set Status
      PDEFoamCell* GetPare() const { return (PDEFoamCell*) fParent.GetObject(); }  // Get Pointer to parent cell
      PDEFoamCell* GetDau0() const { return (PDEFoamCell*) fDaught0.GetObject(); } // Get Pointer to 1-st daughter vertex
      PDEFoamCell* GetDau1() const { return (PDEFoamCell*) fDaught1.GetObject(); } // Get Pointer to 2-nd daughter vertex
      void      SetDau0(PDEFoamCell* Daug){ fDaught0 = Daug;}  // Set pointer to 1-st daughter
      void      SetDau1(PDEFoamCell* Daug){ fDaught1 = Daug;}  // Set pointer to 2-nd daughter
      void      SetPare(PDEFoamCell* Pare){ fParent  = Pare;}  // Set pointer to parent
      void      SetSerial(Int_t Serial){ fSerial=Serial;}    // Set serial number
      Int_t     GetSerial() const { return fSerial;}         // Get serial number
      //--- other ---
      void Print(Option_t *option) const ;                   // Prints cell content
      //--- getter and setter for user variable ---
      void SetElement(TObject* fobj){ fElement = fobj; }     // Set user variable
      TObject* GetElement(){ return fElement; }              // Get pointer to user varibale
      ////////////////////////////////////////////////////////////////////////////
      ClassDef(PDEFoamCell,1)  //Single cell of FOAM
   }; // end of PDEFoamCell
} // namespace TMVA



//////////////////////////////////////////////////////////////////
//                                                              //
// Small auxiliary class for controlling MC weight.             //
//                                                              //
//////////////////////////////////////////////////////////////////

namespace TMVA {

   class PDEFoamMaxwt : public TObject {

   private:
      Double_t  fNent;      // No. of MC events
      Int_t     fnBin;      // No. of bins on the weight distribution
      Double_t  fwmax;      // Maximum analyzed weight

   protected:
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }                          
   
   public:

      TH1D   *fWtHst1;      // Histogram of the weight wt
      TH1D   *fWtHst2;      // Histogram of wt filled with wt

   public:
      // constructor
      PDEFoamMaxwt();                          // NOT IMPLEMENTED (NEVER USED)
      PDEFoamMaxwt(Double_t, Int_t);           // Principal Constructor
      PDEFoamMaxwt(PDEFoamMaxwt &From);        // Copy constructor
      virtual ~PDEFoamMaxwt();                 // Destructor
      void Reset();                            // Reset
      PDEFoamMaxwt& operator=(const PDEFoamMaxwt &);    // operator =
      void Fill(Double_t);
      void Make(Double_t, Double_t&);
      void GetMCeff(Double_t, Double_t&, Double_t&);  // get MC efficiency= <w>/wmax

      ClassDef(PDEFoamMaxwt,1); //Controlling of the MC weight (maximum weight)
   }; // end of PDEFoamMaxwt
} // namespace TMVA



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Auxiliary class PDEFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the PDEFoam cells                       //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

namespace TMVA {
   ///////////////////////////////////////////////////////////////////////////////
   class PDEFoamVect : public TObject {

   private:
      Int_t       fDim;                     // Dimension
      Double_t   *fCoords;                  // [fDim] Coordinates
      PDEFoamVect  *fNext;                  // pointer for tree construction
      PDEFoamVect  *fPrev;                  // pointer for tree construction

   protected:
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }                       

   public:
      // constructor
      PDEFoamVect();                                 // Constructor
      PDEFoamVect(Int_t);                            // USER Constructor
      PDEFoamVect(const PDEFoamVect &);              // Copy constructor
      virtual ~PDEFoamVect();                        // Destructor

      //////////////////////////////////////////////////////////////////////////////
      //                     Overloading operators                                //
      //////////////////////////////////////////////////////////////////////////////
      PDEFoamVect& operator =( const PDEFoamVect& ); // = operator; Substitution
      Double_t & operator[]( Int_t );                // [] provides POINTER to coordinate
      PDEFoamVect& operator =( Double_t [] );        // LOAD IN entire double vector
      PDEFoamVect& operator =( Double_t );           // LOAD IN double number
      //////////////////////////   OTHER METHODS    //////////////////////////////////
      PDEFoamVect& operator+=( const  PDEFoamVect& );  // +=; add vector u+=v  (FAST)
      PDEFoamVect& operator-=( const  PDEFoamVect& );  // +=; add vector u+=v  (FAST)
      PDEFoamVect& operator*=( const  Double_t&  );    // *=; mult. by scalar v*=x (FAST)
      PDEFoamVect  operator+ ( const  PDEFoamVect& );  // +;  u=v+s, NEVER USE IT, SLOW!!!
      PDEFoamVect  operator- ( const  PDEFoamVect& );  // -;  u=v-s, NEVER USE IT, SLOW!!!
      void       Print(Option_t *option) const;    // Prints vector
      void       PrintList();                      // Prints vector and the following linked list
      Int_t      GetDim() const { return fDim; }   // Returns dimension
      Double_t   GetCoord(Int_t i) const { return fCoords[i]; }   // Returns coordinate

      ClassDef(PDEFoamVect,1) //n-dimensional vector with dynamical allocation
   }; // end of PDEFoamVect
}  // namespace TMVA

#endif
