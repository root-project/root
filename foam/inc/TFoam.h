// $Id: TFoam.h,v 1.3 2005/04/05 11:56:00 psawicki Exp $

#ifndef ROOT_TFoam
#define ROOT_TFoam

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TFoam is the main class of the multi-dimensional general purpose         //
// Monte Carlo event generator (integrator) FOAM.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#include "TH1.h"
#include "TRefArray.h"
#include "TMethodCall.h"
#include "TRandom.h"

#include "TFoamIntegrand.h"
#include "TFoamMaxwt.h"
#include "TFoamVect.h"
#include "TFoamCell.h"

class TFoam : public TObject {
 private:
  // COMPONENTS //
  //-------------- Input parameters
  Char_t  fName[128];        // Name of a given instance of the mFOAM class
  Char_t  fVersion[10];      // Actual version of the mFOAM like (1.01m)
  Char_t  fDate[40];         // Release date of mFOAM
  Int_t   fkDim;             // Dimension of the integration/simulation space
  Int_t   fnCells;           // Maximum number of cells
  Int_t   fRNmax;            // Maximum No. of the rand. numb. requested at once
  //-------------------
  Int_t   fOptDrive;         // Optimization switch =1,2 for variance or maximum weight optimization
  Int_t   fChat;             // Chat=0,1,2 chat level in output, Chat=1 normal level
  Int_t   fOptRej;           // Switch =0 for weighted events; =1 for unweighted events in MC
  //-------------------
  Int_t   fnBin;             // No. of bins in the edge histogram for cell MC exploration
  Int_t   fnSampl;           // No. of MC events, when dividing (exploring) cell
  Int_t   fEvPerBin;         // Maximum number of effective (wt=1) events per bin
  //-------------------  MULTI-BRANCHING ---------------------
  Int_t  *fMaskDiv;          //! [fkDim] Dynamic Mask for  cell division
  Int_t  *fInhiDiv;          //! [fkDim] Flags for inhibiting cell division
  Int_t   fOptPRD;           //  Option switch for predefined division, for quick check
  TFoamVect **fXdivPRD;      //! Lists of division values encoded in one vector per direction
  //-------------------  GEOMETRY ----------------------------
  Int_t   fNoAct;            // Number of active cells
  Int_t   fLastCe;           // Index of the last cell
  TFoamCell **fCells;           // [fnCells] Array of ALL cells
  //------------------ M.C. generation----------------------------
  TFoamMaxwt   *fMCMonit;    // Monitor of the MC weight for measuring MC efficiency
  Double_t   fMaxWtRej;      // Maximum weight in rejection for getting wt=1 events
  TRefArray *fCellsAct;      // Array of pointers to active cells, constructed at the end of foam build-up
  Double_t  *fPrimAcu;       // [fNoAct] Array of cumulative probability of all active cells
  TObjArray *fHistEdg;       // Histograms of wt, one for each cell edge
  TObjArray *fHistDbg;       // Histograms of wt, for debug 
  TH1D      *fHistWt;        // Histogram of the MC wt

  Double_t  *fMCvect;        // [fkDim] Generated MC vector for the outside user
  Double_t  fMCwt;           // MC weight
  Double_t *fRvec;           // [fRNmax] random number vector from r.n. generator fkDim+1 maximum elements
  //----------- Procedures 
  TFoamIntegrand *fRho;      // Pointer to the user-defined integrand function/distribution 
  TMethodCall *fMethodCall;  //! ROOT's pointer to user-defined global distribution function
  TRandom         *fPseRan;  // Pointer to user-defined generator of pseudorandom numbers
  //----------- Statistics and MC results
  Long_t   fnCalls;          // Total number of the function calls
  Long_t   fnEffev;          // Total number of effective events (wt=1) in the foam buildup
  Double_t fSumWt, fSumWt2;  // Total sum of wt and wt^2
  Double_t fSumOve;          // Total Sum of overveighted events
  Double_t fNevGen;          // Total number of the generated MC events
  Double_t fWtMax, fWtMin;   // Maximum/Minimum MC weight
  Double_t fPrime;           // Primary integral R' (R=R'<wt>)
  Double_t fMCresult;        // True Integral R from MC series
  Double_t fMCerror;         // and its error
  //----------  working space for CELL exploration -------------
  Double_t *fAlpha;          // [fkDim] Internal parameters of the hyperrectangle
  //////////////////////////////////////////////////////////////////////////////////////////////
  //                                     METHODS                                              //
  //////////////////////////////////////////////////////////////////////////////////////////////
 public:
  TFoam();                          // Default constructor (used only by ROOT streamer)
  TFoam(const Char_t*);             // Principal user-defined constructor
  virtual ~TFoam();                 // Default destructor
  TFoam(const TFoam&);              // Copy Constructor  NOT USED
  // Initialization
  void Initialize();                // Initialization of the mFOAM (grid, cells, etc), mandatory!
  void Initialize(TRandom *, TFoamIntegrand *); // Alternative initialization method, backward compatibility 
  void InitCells(void);             // Initializes first cells inside original cube
  Int_t  CellFill(Int_t, TFoamCell*);  // Allocates new empty cell and return its index
  void Explore(TFoamCell *Cell);       // Exploration of the new cell, determine <wt>, wtMax etc.
  void Carver(Int_t&,Double_t&,Double_t&);// Determines the best edge, wt_max reduction
  void Varedu(Double_t [], Int_t&, Double_t&,Double_t&); // Determines the best edge, variace reduction
  void MakeAlpha(void);             // Provides random point inside hyperrectangle
  void Grow(void);                  // Adds new cells to mFOAM object until buffer is full
  Long_t PeekMax(void);             // Choose one active cell, used by Grow and also in MC generation
  Int_t  Divide(TFoamCell *);       // Divide iCell into two daughters; iCell retained, taged as inactive
  void MakeActiveList(void);        // Creates table of active cells
  void GenerCel2(TFoamCell *&);     // Chose an active cell the with probability ~ Primary integral
  // Generation
  Double_t Eval(Double_t *);        // Evaluates value of the distribution function
  void     MakeEvent(void);         // Makes (generates) single MC event
  void     GetMCvect(Double_t *);   // Provides generated randomly MC vector
  void     GetMCwt(Double_t &);     // Provides generated MC weight
  Double_t GetMCwt(void);           // Provides generates MC weight
  Double_t MCgenerate(Double_t *MCvect);// All three above function in one
  // Finalization
  void GetIntegMC(Double_t&, Double_t&);// Provides Integrand and abs. error from MC run
  void GetIntNorm(Double_t&, Double_t&);// Provides normalization Inegrand
  void GetWtParams(const Double_t, Double_t&, Double_t&, Double_t&);// Provides MC weight parameters
  void Finalize(  Double_t&, Double_t&);  // Prints summary of MC integration
  TFoamIntegrand  *GetRho(){return fRho;};// Gets pointer of the distribut. (after restoring from disk)
  TRandom *GetPseRan(){return fPseRan;};  // Gets pointer of r.n. generator (after restoring from disk)
  void SetRhoInt(void *Rho);              // Set new integrand distr. in interactive mode
  void SetRho(TFoamIntegrand *Rho);       // Set new integrand distr. in compiled mode
  void ResetRho(TFoamIntegrand *Rho);                // Set new distribution, delete old
  void SetPseRan(TRandom *PseRan){fPseRan=PseRan;};  // Set new r.n. generator
  void ResetPseRan(TRandom *PseRan);                 // Set new r.n.g, delete old 
  // Getters and Setters
  void SetkDim(Int_t kDim){fkDim = kDim;};         // Sets dimension of cubical space
  void SetnCells(Long_t nCells){fnCells =nCells;}; // Sets maximum number of cells
  void SetnSampl(Long_t nSampl){fnSampl =nSampl;}; // Sets no of MC events in cell exploration
  void SetnBin(Int_t nBin){fnBin = nBin;};         // Sets no of bins in histogs in cell exploration
  void SetChat(Int_t Chat){fChat = Chat;};         // Sets option Chat, chat level
  void SetOptRej(Int_t OptRej){fOptRej =OptRej;};  // Sets option for MC rejection 
  void SetOptDrive(Int_t OptDrive){fOptDrive =OptDrive;}; // Sets optimization switch
  void SetEvPerBin(Int_t EvPerBin){fEvPerBin =EvPerBin;}; // Sets max. no. of effective events per bin
  void SetMaxWtRej(const Double_t MaxWtRej){fMaxWtRej=MaxWtRej;}; // Sets max. weight for rejection
  void SetInhiDiv(Int_t, Int_t );            // Set inhibition of cell division along certain edge
  void SetXdivPRD(Int_t, Int_t, Double_t[]); // Set predefined division points
  // Getters and Setters
  Char_t * GetVersion(void ){return fVersion;};      // Get version of the mFOAM
  Int_t    GetTotDim(void ){ return fkDim;};         // Get total dimension
  Double_t GetPrimary(void ){return fPrime;};        // Get value of primary integral R'
  void GetPrimary(Double_t &Prime){Prime = fPrime;}; // Get value of primary integral R'
  Long_t GetnCalls(void){return fnCalls;};           // Get total no. of the function calls
  Long_t GetnEffev(void){return fnEffev;};           // Get total no. of effective wt=1 events
  // Debug
  void CheckAll(const Int_t);   // Checks correctness of the entire data structure in the mFOAM object
  void PrintCells(void);        // Prints content of all cells
  void RootPlot2dim(Char_t*);   // Generates C++ code for drawing foam
  // Inline
 private:
  Double_t Sqr( const Double_t x ){ return x*x;};     // Square function
  //////////////////////////////////////////////////////////////////////////////////////////////
  ClassDef(TFoam,1);   // General purpose self-adapting Monte Carlo event generator
};
#endif
