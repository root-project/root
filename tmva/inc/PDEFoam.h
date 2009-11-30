
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoam                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class for PDEFoam object                                                  *
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
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TRandom3
#include "TRandom3.h"
#endif

#ifndef ROOT_TMVA_PDEFoamDistr
#include "TMVA/PDEFoamDistr.h"
#endif
#ifndef ROOT_TMVA_PDEFoamVect
#include "TMVA/PDEFoamVect.h"
#endif
#ifndef ROOT_TMVA_PDEFoamCell
#include "TMVA/PDEFoamCell.h"
#endif

namespace TMVA {
   class PDEFoam;
   class PDEFoamCell;

   enum EKernel { kNone, kGaus, kLinN };
   enum ETargetSelection { kMean, kMpv };
   enum ECellType { kAll, kActive, kInActive };

   // enum type for possible foam cell values
   // kNev           : number of events (saved in cell element 0)
   // kDiscriminator : discriminator (saved in cell element 0)
   // kDiscriminatorError : error on discriminator (saved in cell element 1)
   // kTarget0       : target 0 (saved in cell element 0)
   // kTargetError   : error on target 0 (saved in cell element 1)
   // kMeanValue     : mean sampling value (saved in fIntegral)
   // kRms           : rms of sampling distribution (saved in fDriver)
   // kRmsOvMean     : rms/mean of sampling distribution (saved in 
   //                  fDriver and fIntegral)
   // kDensity       : number of events/cell volume
   enum ECellValue { kNev, kDiscriminator, kDiscriminatorError, kTarget0, 
                     kTarget0Error, kMeanValue, kRms, kRmsOvMean, kDensity };
}

namespace TMVA {

   std::ostream& operator<< ( std::ostream& os, const PDEFoam& pdefoam );
   std::istream& operator>> ( std::istream& istr,     PDEFoam& pdefoam );

   class PDEFoam : public TObject {
   protected:
      // COMPONENTS //
      //-------------- Input parameters
      TString fName;             // Name of a given instance of the FOAM class
      Int_t   fDim;              // Dimension of the integration/simulation space
      Int_t   fNCells;           // Maximum number of cells
      //-------------------
      Int_t   fNBin;             // No. of bins in the edge histogram for cell MC exploration
      Int_t   fNSampl;           // No. of MC events, when dividing (exploring) cell
      Int_t   fEvPerBin;         // Maximum number of effective (wt=1) events per bin
      //-------------------  MULTI-BRANCHING ---------------------
      Int_t  *fMaskDiv;          //! [fDim] Dynamic Mask for cell division
      Int_t  *fInhiDiv;          //! [fDim] Flags for inhibiting cell division
      //-------------------  GEOMETRY ----------------------------
      Int_t   fNoAct;            // Number of active cells
      Int_t   fLastCe;           // Index of the last cell
      PDEFoamCell **fCells;      // [fNCells] Array of ALL cells
      //------------------ M.C. generation----------------------------
      TObjArray *fHistEdg;       // Histograms of wt, one for each cell edge
      Double_t *fRvec;           // [fDim] random number vector from r.n. generator fDim+1 maximum elements
      //----------- Procedures
      TRandom3        *fPseRan;  // Pointer to user-defined generator of pseudorandom numbers
      //----------  working space for CELL exploration -------------
      Double_t *fAlpha;          // [fDim] Internal parameters of the hyperrectangle
      // ---------  PDE-Foam specific variables
      EFoamType fFoamType;     // type of foam
      Double_t *fXmin;         // [fDim] minimum for variable transform
      Double_t *fXmax;         // [fDim] maximum for variable transform
      UInt_t fNElements;       // number of variables in every cell
      Bool_t fCutNmin;         // true: activate cut on minimal number of events in cell
      UInt_t fNmin;            // minimal number of events in cell to split cell
      Bool_t fCutRMSmin;       // true: peek cell with max. RMS for next split
      Double_t fRMSmin;        // activate cut: minimal RMS in cell to split cell
      Float_t fVolFrac;        // volume fraction (with respect to total phase space
      PDEFoamDistr *fDistr;    //! distribution of training events
      Timer *fTimer;           // timer for graphical output
      TObjArray *fVariableNames;// collection of all variable names
      Int_t fSignalClass;      // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      Int_t fBackgroundClass;  // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }

      /////////////////////////////////////////////////////////////////
      //                            METHODS                          //
      /////////////////////////////////////////////////////////////////
   private:
      Double_t Sqr(Double_t x) const { return x*x;}      // Square function

   protected:
      // ---------- TMVA console output

      void OutputGrow(Bool_t finished = false ); // nice TMVA console output

      // ---------- Weighting functions for kernels

      Float_t WeightGaus(PDEFoamCell*, std::vector<Float_t>, UInt_t dim=0);

      Double_t WeightLinNeighbors( std::vector<Float_t> txvec, ECellValue cv, 
                                   Int_t dim1=-1, Int_t dim2=-1, 
                                   Bool_t TreatEmptyCells=kFALSE );
      
      // ---------- Foam build-up functions

      // Internal foam initialization functions
      void InitCells(Bool_t CreateCellElements);      // Initialisation of all foam cells
      Int_t CellFill(Int_t, PDEFoamCell*);// Allocates new empty cell and return its index
      void Explore(PDEFoamCell *Cell);    // Exploration of the new cell, determine <wt>, wtMax etc.
      void Varedu(Double_t [], Int_t&, Double_t&,Double_t&); // Determines the best edge, variace reduction
      void MakeAlpha();             // Provides random point inside hyperrectangle
      void Grow();                  // build up foam
      Long_t PeekMax();             // peek cell with max. driver integral
      Int_t  Divide(PDEFoamCell *); // Divide iCell into two daughters; iCell retained, taged as inactive
      Double_t Eval(Double_t *xRand, Double_t &event_density); // evaluate distribution on point 'xRand'

      // ---------- Cell value access functions

      // low level functions to access a certain cell value
      TVectorD* GetCellElements(std::vector<Float_t>);       // return cell elements of cell with given coordinates
      Double_t GetCellElement(PDEFoamCell *cell, UInt_t i);  // get Element 'i' in cell 'cell'
      void SetCellElement(PDEFoamCell *cell, UInt_t i, Double_t value); // set Element 'i' in cell 'cell' to value 'value'

      // helper functions to access cell data
      Double_t GetCellValue(PDEFoamCell*, ECellValue);

      // specific function used during evaluation; determines, whether a cell value is undefined
      Bool_t   CellValueIsUndefined( PDEFoamCell* );

      // finds cell according to given event variables
      PDEFoamCell* FindCell(std::vector<Float_t>); //!
      std::vector<TMVA::PDEFoamCell*> FindCells(std::vector<Float_t>); //!

      // find cells, which fit a given event vector
      void FindCellsRecursive(std::vector<Float_t>, PDEFoamCell*, 
                              std::vector<PDEFoamCell*> &);
      
      // calculates the mean/ mpv target values for a given event 'tvals'
      std::vector<Float_t> GetCellTargets( std::vector<Float_t> tvals, ETargetSelection ts );
      // get number of events in cell during foam build-up
      Double_t GetBuildUpCellEvents(PDEFoamCell* cell);
      
      // ---------- Public functions ----------------------------------
   public:
      PDEFoam();                  // Default constructor (used only by ROOT streamer)
      PDEFoam(const TString&);    // Principal user-defined constructor
      virtual ~PDEFoam();         // Default destructor
      PDEFoam(const PDEFoam&);    // Copy Constructor  NOT USED

      // ---------- Foam creation functions

      void Init();                    // initialize PDEFoamDistr
      void FillBinarySearchTree( const Event* ev, Bool_t NoNegWeights=kFALSE );
      void Create(Bool_t CreateCellElements=false);     // build-up foam

      // function to fill created cell with given value
      void FillFoamCells(const Event* ev, Bool_t NoNegWeights=kFALSE);

      // functions to calc discriminators/ mean targets for every cell
      // using filled cell values
      void CalcCellDiscr();
      void CalcCellTarget();

      // init TObject pointer on cells
      void ResetCellElements(Bool_t allcells = false);

      // ---------- Getters and Setters

      void SetkDim(Int_t kDim); // Sets dimension of cubical space
      void SetnCells(Long_t nCells){fNCells =nCells;}  // Sets maximum number of cells
      void SetnSampl(Long_t nSampl){fNSampl =nSampl;}  // Sets no of MC events in cell exploration
      void SetnBin(Int_t nBin){fNBin = nBin;}          // Sets no of bins in histogs in cell exploration
      void SetEvPerBin(Int_t EvPerBin){fEvPerBin =EvPerBin;} // Sets max. no. of effective events per bin
      void SetInhiDiv(Int_t, Int_t ); // Set inhibition of cell division along certain edge
      void SetNElements(UInt_t numb){fNElements = numb;} // init every cell element (TVectorD*)
      void SetPDEFoamVolumeFraction(Double_t vfr){fVolFrac = vfr;} // set VolFrac to PDEFoam
      void SetVolumeFraction(Double_t); // set VolFrac to PDEFoamDistr
      void SetFoamType(EFoamType ft);   // set foam type

      void SetSignalClass( Int_t cls )     { fSignalClass = cls; fDistr->SetSignalClass( cls ); } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      void SetBackgroundClass( Int_t cls ) { fBackgroundClass = cls; fDistr->SetBackgroundClass( cls ); } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      Int_t    GetTotDim()    const {return fDim;  } // Get total dimension
      TString  GetFoamName()  const {return fName; } // Get name of foam
      UInt_t   GetNElements() const {return fNElements; } // returns number of elements, saved on every cell
      Double_t GetPDEFoamVolumeFraction() const {return fVolFrac;} // get VolFrac from PDEFoam
      EFoamType GetFoamType()      const {return fFoamType;}; // get foam type
      UInt_t   GetNActiveCells()   const {return fNoAct;}; // returns number of active cells
      UInt_t   GetNInActiveCells() const {return GetNCells()-GetNActiveCells();}; // returns number of not active cells
      UInt_t   GetNCells()         const {return fNCells;};   // returns number of cells
      PDEFoamCell* GetRootCell()   const {return fCells[0];}; // get pointer to root cell

      // Getters and Setters for user cut options
      void     CutNmin(Bool_t cut )    { fCutNmin = cut;    }
      Bool_t   CutNmin()               { return fCutNmin;   }
      void     CutRMSmin(Bool_t cut )  { fCutRMSmin = cut;  }
      Bool_t   CutRMSmin()             { return fCutRMSmin; }
      void     SetNmin(UInt_t val)     { fNmin=val;      }
      UInt_t   GetNmin()               { return fNmin;   }
      void     SetRMSmin(Double_t val) { fRMSmin=val;    }
      Double_t GetRMSmin()             { return fRMSmin; }

      // Getters and Setters for foam boundaries
      void SetXmin(Int_t idim, Double_t wmin);
      void SetXmax(Int_t idim, Double_t wmax);
      Double_t GetXmin(Int_t idim){return fXmin[idim];}
      Double_t GetXmax(Int_t idim){return fXmax[idim];}

      // Getters and Setters for variable names
      void AddVariableName(const char *s) { AddVariableName(new TObjString(s)); };
      void AddVariableName(TObjString *s) { fVariableNames->Add(s); };
      TObjString* GetVariableName(Int_t idx) {return dynamic_cast<TObjString*>(fVariableNames->At(idx));};

      // ---------- Transformation functions for event variables into foam boundaries
      // reason: foam allways has boundaries [0, 1]

      Float_t VarTransform(Int_t idim, Float_t x);       // transform [xmin, xmax] --> [0, 1]
      std::vector<Float_t> VarTransform(std::vector<Float_t> invec);
      Float_t VarTransformInvers(Int_t idim, Float_t x); // transform [0, 1] --> [xmin, xmax]
      std::vector<Float_t> VarTransformInvers(std::vector<Float_t> invec);

      // ---------- Debug functions

      void     CheckAll(Int_t);  // Checks correctness of the entire data structure in the FOAM object
      void     PrintCells();     // Prints content of all cells
      void     CheckCells(Bool_t remove_empty_cells=false);   // check all cells with respect to critical values
      void     RemoveEmptyCell(Int_t iCell); // removes iCell if its volume is zero
      void     PrintCellElements();          // print all cells with its elements

      // ---------- Foam output

      friend std::ostream& operator<< ( std::ostream& os, const PDEFoam& pdefoam );
      friend std::istream& operator>> ( std::istream& istr,     PDEFoam& pdefoam );

      void ReadStream(istream &);         // read  foam from stream
      void PrintStream(ostream  &) const; // write foam from stream
      void ReadXML( void* parent );       // read  foam variables from xml
      void AddXMLTo( void* parent );      // write foam variables to xml

      // ---------- Foam projection methods

      // project foam to two-dimensional histogram
      TH2D* Project2(Int_t idim1, Int_t idim2, const char *opt="nev", 
                     const char *ker="kNone", UInt_t maxbins=0);

      // helper function for Project2()
      Double_t GetProjectionCellValue( PDEFoamCell* cell, 
                                       Int_t idim1, Int_t idim2, ECellValue cv );

      // Project one-dimensional foam to a 1-dim histogram
      TH1D* Draw1Dim(const char *opt, Int_t nbin);

      // Generates C++ code (root macro) for drawing foam with boxes (only 2-dim!)
      void RootPlot2dim( const TString& filename, std::string what,
                         Bool_t CreateCanvas = kTRUE, Bool_t colors = kTRUE, 
                         Bool_t log_colors = kFALSE  );
      
      // ---------- Foam evaluation functions

      // get cell value for a given event
      Double_t GetCellValue(std::vector<Float_t>, ECellValue);

      // helper functions to access cell data with kernel
      Double_t GetCellDiscr(std::vector<Float_t> xvec, EKernel kernel=kNone);
      Double_t GetCellDensity(std::vector<Float_t> xvec, EKernel kernel=kNone);

      // calc mean cell value of neighbor cells
      Double_t GetAverageNeighborsValue(std::vector<Float_t> txvec, ECellValue cv);

      // returns regression value (mono target regression)
      Double_t GetCellRegValue0(std::vector<Float_t>, EKernel kernel=kNone);

      // returns regression value i, given all variables (multi target regression)
      std::vector<Float_t> GetProjectedRegValue(std::vector<Float_t> vals, EKernel kernel=kNone, ETargetSelection ts=kMean);

      // ---------- ROOT class definition
      ClassDef(PDEFoam,3)
   }; // end of PDEFoam 

}  // namespace TMVA

// ---------- Inline functions

//_____________________________________________________________________
inline Float_t TMVA::PDEFoam::VarTransform(Int_t idim, Float_t x) 
{
   // transform variable x from [xmin, xmax] --> [0, 1]
   return (x-fXmin[idim])/(fXmax[idim]-fXmin[idim]);
}

//_____________________________________________________________________
inline std::vector<Float_t> TMVA::PDEFoam::VarTransform(std::vector<Float_t> invec)
{
   // transform vector invec from [xmin, xmax] --> [0, 1]
   std::vector<Float_t> outvec;
   for(UInt_t i=0; i<invec.size(); i++)
      outvec.push_back(VarTransform(i, invec.at(i)));
   return outvec;
}

//_____________________________________________________________________
inline Float_t TMVA::PDEFoam::VarTransformInvers(Int_t idim, Float_t x)
{ 
   // transform variable x from [0, 1] --> [xmin, xmax]
   return x*(fXmax[idim]-fXmin[idim]) + fXmin[idim];
}

//_____________________________________________________________________
inline std::vector<Float_t> TMVA::PDEFoam::VarTransformInvers(std::vector<Float_t> invec)
{
   // transform vector invec from [0, 1] --> [xmin, xmax]
   std::vector<Float_t> outvec;
   for(UInt_t i=0; i<invec.size(); i++)
      outvec.push_back(VarTransformInvers(i, invec.at(i)));
   return outvec;
}

#endif
