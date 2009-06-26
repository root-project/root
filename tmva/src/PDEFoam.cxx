// @(#)root/foam:$Name: not supported by cvs2svn $:$Id$
// Authors: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoam, PDEFoamCell, PDEFoamIntegrand, PDEFoamMaxwt, PDEFoamVect,    *
 *          TFDISTR                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementations                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                * 
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of PDEFoam
//
// The PDEFoam method is an
// extension of the PDERS method, which uses self-adapting binning to
// divide the multi-dimensional phase space in a finite number of
// hyper-rectangles (boxes).
//
// For a given number of boxes, the binning algorithm adjusts the size
// and position of the boxes inside the multidimensional phase space,
// minimizing the variance of the signal and background densities inside
// the boxes. The binned density information is stored in binary trees,
// allowing for a very fast and memory-efficient classification of
// events.
//
// The implementation of the PDEFoam is based on the monte-carlo
// integration package PDEFoam included in the analysis package ROOT.
//_______________________________________________________________________


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <assert.h>

#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TMVA/PDEFoam.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"

#include "TStyle.h"
#include "TObject.h"
#include "TH1.h"
#include "TMath.h"
#include "TVectorT.h"
#include "TRandom3.h"
#include "TRefArray.h"
#include "TMethodCall.h"
#include "G__ci.h"
#include "TColor.h"
#include "TSystem.h"

ClassImp(TMVA::PDEFoam)

static const Float_t gHigh= FLT_MAX;
static const Float_t gVlow=-FLT_MAX;

using namespace std;

#define SW2 setprecision(7) << setw(12)

//________________________________________________________________________________________________
TMVA::PDEFoam::PDEFoam() :
   fLogger(new MsgLogger("PDEFoam"))
{
   // Default constructor for streamer, user should not use it.
   fDim      = 0;
   fNoAct    = 0;
   fNCells   = 0;
   fRNmax    = 0;
   fMaskDiv  = 0;
   fInhiDiv  = 0;
   fXdivPRD  = 0;
   fCells    = 0;
   fAlpha    = 0;
   fCellsAct = 0;
   fPrimAcu  = 0;
   fHistEdg  = 0;
   fHistWt   = 0;
   fHistDbg  = 0;
   fMCMonit  = 0;
   fRho      = 0;  // Integrand function
   fMCvect   = 0;
   fRvec     = 0;
   fPseRan   = 0;  // generator of pseudorandom numbers

   fXmin      = 0;
   fXmax      = 0;
   fNElements = 0;
   fCutNmin   = kTRUE;
   fNmin      = 100;  // only used, when fCutMin == kTRUE
   fCutRMSmin = kFALSE;
   fRMSmin    = 1.0;

   SetPDEFoamVolumeFraction(-1.);

   fSignalClass     = -1;
   fBackgroundClass = -1;

   fDistr = new TFDISTR();
   fDistr->SetSignalClass( fSignalClass );
   fDistr->SetBackgroundClass( fBackgroundClass );

   fTimer = new Timer(fNCells, "PDEFoam", kTRUE);
   fVariableNames = new TObjArray();
}

//_________________________________________________________________________________________________
TMVA::PDEFoam::PDEFoam(const TString& Name) :
   fLogger(new MsgLogger("PDEFoam"))
{
   // User constructor, to be employed by the user

   if(strlen(Name)  >129) {
      Log() << kFATAL << "Name too long " << Name.Data() << Endl;
   }
   fName=Name;                                            // Class name
   fDate="  Release date:  2005.04.10";                   // Release date
   fVersion= "1.02M";                                     // Release version
   fMaskDiv  = 0;             // Dynamic Mask for  cell division, h-cubic
   fInhiDiv  = 0;             // Flag allowing to inhibit cell division in certain projection/edge
   fXdivPRD  = 0;             // Lists of division values encoded in one vector per direction
   fCells    = 0;
   fAlpha    = 0;
   fCellsAct = 0;
   fPrimAcu  = 0;
   fHistEdg  = 0;
   fHistWt   = 0;
   fHistDbg  = 0;
   fDim     = 0;                // dimension of hyp-cubical space
   fNCells   = 1000;             // Maximum number of Cells,    is usually re-set
   fNSampl   = 200;              // No of sampling when dividing cell
   fOptPRD   = 0;                // General Option switch for PRedefined Division, for quick check
   fOptDrive = 2;                // type of Drive =1,2 for TrueVol,Sigma,WtMax
   fChat     = 1;                // Chat=0,1,2 chat level in output, Chat=1 normal level
   fOptRej   = 1;                // OptRej=0, wted events; OptRej=1, wt=1 events
   //------------------------------------------------------
   fNBin     = 8;                // binning of edge-histogram in cell exploration
   fEvPerBin =25;                // maximum no. of EFFECTIVE event per bin, =0 option is inactive
   //------------------------------------------------------
   fNCalls = 0;                  // No of function calls
   fNEffev = 0;                  // Total no of eff. wt=1 events in build=up
   fLastCe =-1;                  // Index of the last cell
   fNoAct  = 0;                  // No of active cells (used in MC generation)
   fWtMin = gHigh;               // Minimal weight
   fWtMax = gVlow;               // Maximal weight
   fMaxWtRej =1.10;              // Maximum weight in rejection for getting wt=1 events
   fPseRan   = 0;                // Initialize private copy of random number generator
   fMCMonit  = 0;                // MC efficiency monitoring
   fRho = 0;                     // pointer to abstract class providing function to integrate
   fMethodCall=0;                // ROOT's pointer to global distribution function

   fXmin      = 0;
   fXmax      = 0;
   fCutNmin   = kFALSE;
   fCutRMSmin = kFALSE;
   SetPDEFoamVolumeFraction(-1.);

   fSignalClass     = -1;
   fBackgroundClass = -1;

   fDistr = new TFDISTR();
   fDistr->SetSignalClass( fSignalClass );
   fDistr->SetBackgroundClass( fBackgroundClass );

   fTimer = new Timer(fNCells, "PDEFoam", kTRUE);
   fVariableNames = new TObjArray();

   Log().SetSource( "PDEFoam" );
}

//_______________________________________________________________________________________________
TMVA::PDEFoam::~PDEFoam()
{
   delete fVariableNames;
   delete fTimer;
   delete fDistr;
   if (fXmin) delete [] fXmin;  fXmin=0; 
   if (fXmax) delete [] fXmax;  fXmax=0;

   // Default destructor
   Int_t i;

   if(fCells!= 0) {
      for(i=0; i<fNCells; i++) delete fCells[i]; // PDEFoamCell*[]
      delete [] fCells;
   }
   delete [] fRvec;    //double[]
   delete [] fAlpha;   //double[]
   delete [] fMCvect;  //double[]
   delete [] fPrimAcu; //double[]
   delete [] fMaskDiv; //int[]
   delete [] fInhiDiv; //int[]
 
   if( fXdivPRD!= 0) {
      for(i=0; i<fDim; i++) delete fXdivPRD[i]; // PDEFoamVect*[]
      delete [] fXdivPRD;
   }
   delete fMCMonit;
   delete fHistWt;

   delete fLogger;
}

//_____________________________________________________________________________________________
TMVA::PDEFoam::PDEFoam(const PDEFoam &From): 
   TObject(From),
   fLogger( new MsgLogger("PDEFoam"))
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_______________________________________________________________________________________
void TMVA::PDEFoam::Initialize(Bool_t CreateCellElements)
{
   // Basic initialization of FOAM invoked by the user.
   // IMPORTANT: Random number generator and the distribution object has to be
   // provided using SetPseRan and SetRho prior to invoking this initializator!

   Bool_t addStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   Int_t i;

   if(fPseRan==0) Log() << kFATAL << "Random number generator not set" << Endl;
   if(fRho==0 && fMethodCall==0 ) Log() << kFATAL << "Distribution function not set" << Endl;
   if(fDim==0) Log() << kFATAL << "Zero dimension not allowed" << Endl;

   /////////////////////////////////////////////////////////////////////////
   //                   ALLOCATE SMALL LISTS                              //
   //  it is done globally, not for each cell, to save on allocation time //
   /////////////////////////////////////////////////////////////////////////
   fRNmax= fDim+1;
   fRvec = new Double_t[fRNmax];   // Vector of random numbers
   if(fRvec==0)  Log() << kFATAL << "Cannot initialize buffer fRvec" << Endl;

   if(fDim>0){
      fAlpha = new Double_t[fDim];    // sum<1 for internal parametrization of the simplex
      if(fAlpha==0)  Log() << kFATAL << "Cannot initialize buffer fAlpha" << Endl;
   }
   fMCvect = new Double_t[fDim]; // vector generated in the MC run
   if(fMCvect==0)  Log() << kFATAL << "Cannot initialize buffer fMCvect" << Endl;

   //====== List of directions inhibited for division
   if(fInhiDiv == 0){
      fInhiDiv = new Int_t[fDim];
      for(i=0; i<fDim; i++) fInhiDiv[i]=0;
   }
   //====== Dynamic mask used in Explore for edge determination
   if(fMaskDiv == 0){
      fMaskDiv = new Int_t[fDim];
      for(i=0; i<fDim; i++) fMaskDiv[i]=1;
   }
   //====== List of predefined division values in all directions (initialized as empty)
   if(fXdivPRD == 0){
      fXdivPRD = new PDEFoamVect*[fDim];
      for(i=0; i<fDim; i++)  fXdivPRD[i]=0; // Artificially extended beyond fDim
   }
   //====== Initialize list of histograms
   fHistWt  = new TH1D("HistWt","Histogram of MC weight",100,0.0, 1.5*fMaxWtRej); // MC weight
   fHistEdg = new TObjArray(fDim);           // Initialize list of histograms
   TString hname;
   TString htitle;
   for(i=0;i<fDim;i++){
      hname=fName+TString("_HistEdge_");
      hname+=i;
      htitle=TString("Edge Histogram No. ");
      htitle+=i;
      (*fHistEdg)[i] = new TH1D(hname.Data(),htitle.Data(),fNBin,0.0, 1.0); // Initialize histogram for each edge
      ((TH1D*)(*fHistEdg)[i])->Sumw2();
   }
   //======  extra histograms for debug purposes
   fHistDbg = new TObjArray(fDim);         // Initialize list of histograms
   for(i=0;i<fDim;i++){
      hname=fName+TString("_HistDebug_");
      hname+=i;
      htitle=TString("Debug Histogram ");
      htitle+=i;
      (*fHistDbg)[i] = new TH1D(hname.Data(),htitle.Data(),fNBin,0.0, 1.0); // Initialize histogram for each edge
   }

   // ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| //
   //                     BUILD-UP of the FOAM                            //
   // ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| //
   //
   //        Define and explore root cell(s)
   InitCells(CreateCellElements);
   //        PrintCells(); Log()<<" ===== after InitCells ====="<<Endl;
   Grow();
   //        PrintCells(); Log()<<" ===== after Grow      ====="<<Endl;

   MakeActiveList(); // Final Preperations for the M.C. generation

   // Preperations for the M.C. generation
   fSumWt  = 0.0;               // M.C. generation sum of Wt
   fSumWt2 = 0.0;               // M.C. generation sum of Wt**2
   fSumOve = 0.0;               // M.C. generation sum of overweighted
   fNevGen = 0.0;               // M.C. generation sum of 1d0
   fWtMax  = gVlow;               // M.C. generation maximum wt
   fWtMin  = gHigh;               // M.C. generation minimum wt
   fMCresult=fCells[0]->GetIntg(); // M.C. Value of INTEGRAL,temporary assignment
   fMCresult=fCells[0]->GetIntg(); // M.C. Value of INTEGRAL,temporary assignment
   fMCerror =fCells[0]->GetIntg(); // M.C. Value of ERROR   ,temporary assignment
   fMCMonit = new PDEFoamMaxwt(5.0,1000);  // monitoring M.C. efficiency

   if(fChat==2) PrintCells();
   TH1::AddDirectory(addStatus);
} // Initialize

//_____________________________________________________________________________________________
void TMVA::PDEFoam::Initialize(TRandom3*PseRan, PDEFoamIntegrand *fun )
{
   // Basic initialization of FOAM invoked by the user. Mandatory!
   // ============================================================
   // This method starts the process of the cell build-up.
   // User must invoke Initialize with two arguments or Initialize without arguments.
   // This is done BEFORE generating first MC event and AFTER allocating FOAM object
   // and reseting (optionally) its internal parameters/switches.
   //
   // This method invokes several other methods:
   // ==========================================
   // InitCells initializes memory storage for cells and begins exploration process
   // from the root cell. The empty cells are allocated/filled using  CellFill.
   // The procedure Grow which loops over cells, picks up the cell with the biggest
   // ``driver integral'', see Comp. Phys. Commun. 152 152 (2003) 55 for explanations,
   // with the help of PeekMax procedure. The chosen cell is split using Divide.
   // Subsequently, the procedure Explore called by the Divide
   // (and by InitCells for the root cell) does the most important
   // job in the FOAM object build-up: it performs a small MC run for each
   // newly allocated daughter cell.
   // Explore calculates how profitable the future split of the cell will be
   // and defines the optimal cell division geometry with the help of Carver or Varedu
   // procedures, for maximum weight or variance optimization respectively.
   // All essential results of the exploration are written into
   // the explored cell object. At the very end of the foam build-up,
   // Finally, MakeActiveList is invoked to create a list of pointers to
   // all active cells, for the purpose of the quick access during the MC generation.
   // The procedure Explore employs MakeAlpha to generate random coordinates
   // inside a given cell with the uniform distribution.

   SetPseRan(PseRan);
   SetRho(fun);
   Initialize(kFALSE);
}

//_______________________________________________________________________________________
void TMVA::PDEFoam::InitCells(Bool_t CreateCellElements)
{
   // Internal subprogram used by Initialize.
   // It initializes "root part" of the FOAM of the tree of cells.

   Int_t i;

   fLastCe =-1;                             // Index of the last cell
   if(fCells!= 0) {
      for(i=0; i<fNCells; i++) delete fCells[i];
      delete [] fCells;
   }
   //
   fCells = new PDEFoamCell*[fNCells];
   for(i=0;i<fNCells;i++){
      fCells[i]= new PDEFoamCell(fDim); // Allocate BIG list of cells
      fCells[i]->SetSerial(i);
   }
   if(fCells==0) Log() << kFATAL << "Cannot initialize CELLS" << Endl;

   // create cell elemets
   if (CreateCellElements)
      ResetCellElements(true);

   /////////////////////////////////////////////////////////////////////////////
   //              Single Root Hypercube                                      //
   /////////////////////////////////////////////////////////////////////////////
   CellFill(1,   0);  //  0-th cell ACTIVE

   // Exploration of the root cell(s)
   for(Long_t iCell=0; iCell<=fLastCe; iCell++){
      Explore( fCells[iCell] );               // Exploration of root cell(s)
   }
}//InitCells

//_______________________________________________________________________________________
Int_t TMVA::PDEFoam::CellFill(Int_t Status, PDEFoamCell *parent)
{
   // Internal subprogram used by Initialize.
   // It initializes content of the newly allocated active cell.

   PDEFoamCell *cell;
   if (fLastCe==fNCells){
      Log() << kFATAL << "Too many cells" << Endl;
   }
   fLastCe++;   // 0-th cell is the first
   if (Status==1) fNoAct++;

   cell = fCells[fLastCe];

   cell->Fill(Status, parent, 0, 0);

   cell->SetBest( -1);         // pointer for planning division of the cell
   cell->SetXdiv(0.5);         // factor for division
   Double_t xInt2,xDri2;
   if(parent!=0){
      xInt2  = 0.5*parent->GetIntg();
      xDri2  = 0.5*parent->GetDriv();
      cell->SetIntg(xInt2);
      cell->SetDriv(xDri2);
   }else{
      cell->SetIntg(0.0);
      cell->SetDriv(0.0);
   }
   return fLastCe;
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::Explore(PDEFoamCell *cell)
{
   // Internal subprogram used by Initialize.
   // It explores newly defined cell with help of special short MC sampling.
   // As a result, estimates of kTRUE and drive volume is defined/determined
   // Average and dispersion of the weight distribution will is found along
   // each edge and the best edge (minimum dispersion, best maximum weight)
   // is memorized for future use.
   // The optimal division point for eventual future cell division is
   // determined/recorded. Recorded are also minimum and maximum weight etc.
   // The volume estimate in all (inactive) parent cells is updated.
   // Note that links to parents and initial volume = 1/2 parent has to be
   // already defined prior to calling this routine.
   //
   // This function is overridden from the original PDEFoam::Explore()
   // to provide an extra option:   Filling edge histograms directly from the 
   // input distributions, w/o MC sampling if fNSampl == 0
   // Warning:  This option is not tested jet!

   Double_t wt, dx, xBest, yBest;
   Double_t intOld, driOld;

   Long_t iev;
   Double_t nevMC;
   Int_t i, j, k;
   Int_t nProj, kBest;
   Double_t ceSum[5], xproj;

   Double_t event_density = 0;
   Double_t totevents     = 0;
   Double_t toteventsOld  = 0;

   PDEFoamVect  cellSize(fDim);
   PDEFoamVect  cellPosi(fDim);

   cell->GetHcub(cellPosi,cellSize);

   PDEFoamCell  *parent;

   Double_t *xRand = new Double_t[fDim];

   Double_t *volPart=0;

   cell->CalcVolume();
   dx = cell->GetVolume();
   intOld = cell->GetIntg(); //memorize old values,
   driOld = cell->GetDriv(); //will be needed for correcting parent cells
   if (CutNmin())
      toteventsOld = GetCellEvents(cell);

   /////////////////////////////////////////////////////
   //    Special Short MC sampling to probe cell      //
   /////////////////////////////////////////////////////
   ceSum[0]=0;
   ceSum[1]=0;
   ceSum[2]=0;
   ceSum[3]=gHigh;  //wtmin
   ceSum[4]=gVlow;  //wtmax
   //
   for (i=0;i<fDim;i++) ((TH1D *)(*fHistEdg)[i])->Reset(); // Reset histograms
   fHistWt->Reset();
   //

   Double_t nevEff=0.;

   // DD 10.Jan.2008
   // use new routine that fills edge histograms directly from the 
   // input distributions, w/o MC sampling
   if (fNSampl==0) {
      Double_t *cellPosiarr = new Double_t[Int_t(fDim)];
      Double_t *cellSizearr = new Double_t[Int_t(fDim)];
      for (Int_t idim=0; idim<fDim; idim++) {
         cellPosiarr[idim]=cellPosi[idim];
         cellSizearr[idim]=cellSize[idim];
      }
      // not jet implemented:
      // fRho->FillEdgeHist(fDim, fHistEdg, cellPosiarr, cellSizearr, ceSum, dx);
      delete[] cellPosiarr;
      delete[] cellSizearr;
   }   
   else {
      // ||||||||||||||||||||||||||BEGIN MC LOOP|||||||||||||||||||||||||||||
      for (iev=0;iev<fNSampl;iev++){
         MakeAlpha();               // generate uniformly vector inside hypercube
       
         if (fDim>0){
            for (j=0; j<fDim; j++)
               xRand[j]= cellPosi[j] +fAlpha[j]*(cellSize[j]);
         }
       
         wt         = dx*Eval(xRand, event_density);
	 totevents += dx*event_density;
       
         nProj = 0;
         if (fDim>0) {
            for (k=0; k<fDim; k++) {
               xproj =fAlpha[k];
               ((TH1D *)(*fHistEdg)[nProj])->Fill(xproj,wt);
               nProj++;
            }
         }
         //
         fNCalls++;
         ceSum[0] += wt;    // sum of weights
         ceSum[1] += wt*wt; // sum of weights squared
         ceSum[2]++;        // sum of 1
         if (ceSum[3]>wt) ceSum[3]=wt;  // minimum weight;
         if (ceSum[4]<wt) ceSum[4]=wt;  // maximum weight
         // test MC loop exit condition
         nevEff = ceSum[0]*ceSum[0]/ceSum[1];
         if ( nevEff >= fNBin*fEvPerBin) break;
      }   // ||||||||||||||||||||||||||END MC LOOP|||||||||||||||||||||||||||||
      totevents /= fNSampl;
   }

   // make shure that, if root cell is explored, more than zero
   // events were found.
   if (cell==fCells[0] && ceSum[0]<=0.0){
      if (ceSum[0]==0.0)
	 Log() << kFATAL << "No events were found during exploration of " 
	       << "root cell.  Please check PDEFoam parameters nSampl " 
	       << "and VolFrac." << Endl;
      else
	 Log() << kWARNING << "Negative number of events found during "
	       << "exploration of root cell" << Endl;
   }

   //------------------------------------------------------------------
   //---  predefine logics of searching for the best division edge ---
   for (k=0; k<fDim;k++){
      fMaskDiv[k] =1;                       // default is all
      if ( fInhiDiv[k]==1) fMaskDiv[k] =0; // inhibit some...
   }
   // Note that predefined division below overrule inhibition above
   kBest=-1;
   Double_t rmin,rmax,rdiv;
   if (fOptPRD) {          // quick check
      for (k=0; k<fDim; k++) {
         rmin= cellPosi[k];
         rmax= cellPosi[k] +cellSize[k];
         if ( fXdivPRD[k] != 0) {
            Int_t n= (fXdivPRD[k])->GetDim();
            for (j=0; j<n; j++) {
               rdiv=(*fXdivPRD[k])[j];
               // check predefined divisions is available in this cell
               if ( (rmin +1e-99 <rdiv) && (rdiv< rmax -1e-99)) {
                  kBest=k;
                  xBest= (rdiv-cellPosi[k])/cellSize[k] ;
                  goto ee05;
               }
            }
         }
      }//k
   }
 ee05:
   //------------------------------------------------------------------
   fNEffev += (Long_t)nevEff;
   nevMC          = ceSum[2];
   Double_t intTrue = ceSum[0]/(nevMC+0.000001);
   Double_t intDriv=0.;
   Double_t intPrim=0.;
   
   switch(fOptDrive){
   case 1:                       // VARIANCE REDUCTION
      if (kBest == -1) Varedu(ceSum,kBest,xBest,yBest); // determine the best edge,
      if (CutRMSmin())
         intDriv =sqrt( ceSum[1]/nevMC -intTrue*intTrue ); // Older ansatz, numerically not bad
      else
         intDriv =sqrt(ceSum[1]/nevMC) -intTrue; // Foam build-up, sqrt(<w**2>) -<w>
      intPrim =sqrt(ceSum[1]/nevMC);          // MC gen. sqrt(<w**2>) =sqrt(<w>**2 +sigma**2)
      break;
   case 2:                       // WTMAX  REDUCTION
      if (kBest == -1) Carver(kBest,xBest,yBest);  // determine the best edge
      intDriv =ceSum[4] -intTrue; // Foam build-up, wtmax-<w>
      intPrim =ceSum[4];          // MC generation, wtmax!
      break;
   default:
      Log() << kFATAL << "Wrong fOptDrive = " << Endl;
   }//switch
   //=================================================================================
   //hist_Neff_distrib.Fill( fLastCe/2.0+0.01, nevEff+0.01);  //
   //hist_kBest_distrib.Fill( kBest+0.50, 1.0 ); //  debug
   //hist_xBest_distrib.Fill( xBest+0.01, 1.0 ); //  debug
   //=================================================================================
   cell->SetBest(kBest);
   cell->SetXdiv(xBest);
   cell->SetIntg(intTrue);
   cell->SetDriv(intDriv);
   cell->SetPrim(intPrim);
   if (CutNmin())
      SetCellElement(cell, 0, totevents);
   
   // correct/update integrals in all parent cells to the top of the tree
   Double_t  parIntg, parDriv;
   for (parent = cell->GetPare(); parent!=0; parent = parent->GetPare()){
      parIntg = parent->GetIntg();
      parDriv = parent->GetDriv();
      parent->SetIntg( parIntg   +intTrue -intOld );
      parent->SetDriv( parDriv   +intDriv -driOld );
      if (CutNmin())
	 SetCellElement( parent, 0, GetCellEvents(parent) + totevents - toteventsOld);
   }
   delete [] volPart;
   delete [] xRand;

}

//______________________________________________________________________________________
void TMVA::PDEFoam::Varedu(Double_t ceSum[5], Int_t &kBest, Double_t &xBest, Double_t &yBest)
{
   // Internal subrogram used by Initialize.
   // In determines the best edge candidate and the position of the cell division plane
   // in case of the variance reduction for future cell division,
   // using results of the MC exploration run stored in fHistEdg

   Double_t nent   = ceSum[2];
   Double_t swAll  = ceSum[0];
   Double_t sswAll = ceSum[1];
   Double_t ssw    = sqrt(sswAll)/sqrt(nent);
   //
   Double_t swIn,swOut,sswIn,sswOut,xLo,xUp;
   kBest =-1;
   xBest =0.5;
   yBest =1.0;
   Double_t maxGain=0.0;
   // Now go over all projections kProj
   for(Int_t kProj=0; kProj<fDim; kProj++) {
      if( fMaskDiv[kProj]) {
         // initialize search over bins
         Double_t sigmIn =0.0; Double_t sigmOut =0.0;
         Double_t sswtBest = gHigh;
         Double_t gain =0.0;
         Double_t xMin=0.0; Double_t xMax=0.0;
         // Double loop over all pairs jLo<jUp
         for(Int_t jLo=1; jLo<=fNBin; jLo++) {
            Double_t aswIn=0;  Double_t asswIn=0;
            for(Int_t jUp=jLo; jUp<=fNBin;jUp++) {
	      aswIn  +=     ((TH1D *)(*fHistEdg)[kProj])->GetBinContent(jUp);
	      asswIn += Sqr(((TH1D *)(*fHistEdg)[kProj])->GetBinError(  jUp));
              xLo=(jLo-1.0)/fNBin;
               xUp=(jUp*1.0)/fNBin;
               swIn  =        aswIn/nent;
               swOut = (swAll-aswIn)/nent;
               sswIn = sqrt(asswIn)       /sqrt(nent*(xUp-xLo))     *(xUp-xLo);
               sswOut= sqrt(sswAll-asswIn)/sqrt(nent*(1.0-xUp+xLo)) *(1.0-xUp+xLo);
               if( (sswIn+sswOut) < sswtBest) {
                  sswtBest = sswIn+sswOut;
                  gain     = ssw-sswtBest;
                  sigmIn   = sswIn -swIn;  // Debug
                  sigmOut  = sswOut-swOut; // Debug
                  xMin    = xLo;
                  xMax    = xUp;
               }
            }//jUp
         }//jLo
         Int_t iLo = (Int_t) (fNBin*xMin);
         Int_t iUp = (Int_t) (fNBin*xMax);
         //----------DEBUG printout
         //Log()<<"@@@@@  xMin xMax = "<<xMin   <<" "<<xMax<<"  iLo= "<<iLo<<"  iUp= "<<iUp;
         //Log()<<"  sswtBest/ssw= "<<sswtBest/ssw<<"  Gain/ssw= "<< Gain/ssw<<Endl;
         //----------DEBUG auxilary Plot
	 //         for(Int_t iBin=1;iBin<=fNBin;iBin++) {
	 //            if( ((iBin-0.5)/fNBin > xMin) && ((iBin-0.5)/fNBin < xMax) ){
	 //               ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin,sigmIn/(xMax-xMin));
	 //            } else {
	 //               ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin,sigmOut/(1-xMax+xMin));
	 //            }
	 //         }
         if(gain>=maxGain) {
            maxGain=gain;
            kBest=kProj; // <--- !!!!! The best edge
            xBest=xMin;
            yBest=xMax;
            if(iLo == 0)     xBest=yBest; // The best division point
            if(iUp == fNBin) yBest=xBest; // this is not really used
         }
      }
   } //kProj
   //----------DEBUG printout
   //Log()<<"@@@@@@@>>>>> kBest= "<<kBest<<"  maxGain/ssw= "<< maxGain/ssw<<Endl;
   if( (kBest >= fDim) || (kBest<0) ) Log() << kFATAL << "Something wrong with kBest" << Endl;
}          //PDEFoam::Varedu

//________________________________________________________________________________________
void TMVA::PDEFoam::Carver(Int_t &kBest, Double_t &xBest, Double_t &yBest)
{
   // Internal subrogram used by Initialize.
   // Determines the best edge-candidate and the position of the division plane
   // for the future cell division, in the case of the optimization of the maximum weight.
   // It exploits results of the cell MC exploration run stored in fHistEdg.

   Int_t    kProj,iBin;
   Double_t carve,carvTot,carvMax,carvOne,binMax,binTot,primTot,primMax;
   Int_t    jLow,jUp,iLow,iUp;
   Double_t theBin;
   Int_t    jDivi; // TEST
   Int_t j;

   Double_t *bins  = new Double_t[fNBin];      // bins of histogram for single  PROJECTION
   if(bins==0)    Log() << kFATAL << "Cannot initialize buffer Bins " << Endl;

   kBest =-1;
   xBest =0.5;
   yBest =1.0;
   carvMax = gVlow;
   primMax = gVlow;
   for(kProj=0; kProj<fDim; kProj++)
      if( fMaskDiv[kProj] ){
         //if( kProj==1 ){
         //Log()<<"==================== Carver histogram: kProj ="<<kProj<<"==================="<<Endl;
         //((TH1D *)(*fHistEdg)[kProj])->Print("all");
         binMax = gVlow;
         for(iBin=0; iBin<fNBin;iBin++){
            bins[iBin]= ((TH1D *)(*fHistEdg)[kProj])->GetBinContent(iBin+1);
            binMax = TMath::Max( binMax, bins[iBin]);       // Maximum content/bin
         }
         if(binMax < 0 ) {       //case of empty cell
            delete [] bins;
            return;
         }
         carvTot = 0.0;
         binTot  = 0.0;
         for(iBin=0;iBin<fNBin;iBin++){
            carvTot = carvTot + (binMax-bins[iBin]);     // Total Carve (more stable)
            binTot  +=bins[iBin];
         }
         primTot = binMax*fNBin;
         //Log() <<"Carver:  CarvTot "<<CarvTot<< "    primTot "<<primTot<<Endl;
         jLow =0;
         jUp  =fNBin-1;
         carvOne = gVlow;
         Double_t yLevel = gVlow;
         for(iBin=0; iBin<fNBin;iBin++) {
            theBin = bins[iBin];
            //-----  walk to the left and find first bin > theBin
            iLow = iBin;
            for(j=iBin; j>-1; j-- ) {
               if(theBin< bins[j]) break;
               iLow = j;
            }
            //iLow = iBin;
            //if(iLow>0)     while( (theBin >= bins[iLow-1])&&(iLow >0) ){iLow--;} // horror!!!
            //------ walk to the right and find first bin > theBin
            iUp  = iBin;
            for(j=iBin; j<fNBin; j++) {
               if(theBin< bins[j]) break;
               iUp = j;
            }
            //iUp  = iBin;
            //if(iUp<fNBin-1) while( (theBin >= bins[iUp+1])&&( iUp<fNBin-1 ) ){iUp++;} // horror!!!
            //
            carve = (iUp-iLow+1)*(binMax-theBin);
            if( carve > carvOne) {
               carvOne = carve;
               jLow = iLow;
               jUp  = iUp;
               yLevel = theBin;
            }
         }//iBin
         if( carvTot > carvMax) {
            carvMax   = carvTot;
            primMax   = primTot;
            //Log() <<"Carver:   primMax "<<primMax<<Endl;
            kBest = kProj;    // Best edge
            xBest = ((Double_t)(jLow))/fNBin;
            yBest = ((Double_t)(jUp+1))/fNBin;
            if(jLow == 0 )       xBest = yBest;
            if(jUp  == fNBin-1) yBest = xBest;
            // division ratio in units of 1/fNBin, testing
            jDivi = jLow;
            if(jLow == 0 )     jDivi=jUp+1;
         }
         //======  extra histograms for debug purposes
         //Log()<<"kProj= "<<kProj<<" jLow= "<<jLow<<" jUp= "<<jUp<<Endl;
         for(iBin=0;    iBin<fNBin;  iBin++)
            ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin+1,binMax);
         for(iBin=jLow; iBin<jUp+1;   iBin++)
            ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin+1,yLevel);
      }//kProj
   if( (kBest >= fDim) || (kBest<0) ) Log() << kFATAL << "Something wrong with kBest" << Endl;
   delete [] bins;
}          //PDEFoam::Carver

//______________________________________________________________________________________________
void TMVA::PDEFoam::MakeAlpha()
{
   // Internal subrogram used by Initialize.
   // Provides random vector Alpha  0< Alpha(i) < 1
   Int_t k;
   if(fDim<1) return;

   // simply generate and load kDim uniform random numbers
   fPseRan->RndmArray(fDim,fRvec);   // kDim random numbers needed
   for(k=0; k<fDim; k++) fAlpha[k] = fRvec[k];
} //MakeAlpha

//_____________________________________________________________________________________________
Long_t TMVA::PDEFoam::PeekMax()
{
   // Internal subprogram used by Initialize.
   // It finds cell with maximal driver integral for the purpose of the division.
   // This function is overridden by the PDEFoam Class to apply cuts
   // on Nmin and RMSmin during cell buildup.

   Long_t iCell = -1;

   Long_t  i;
   Double_t  drivMax, driv;
   Bool_t bCutNmin = kTRUE;
   Bool_t bCutRMS  = kTRUE;
   //   drivMax = gVlow;
   drivMax = 0;  // only split cells if gain>0 (this also avoids splitting at cell boundary)
   for(i=0; i<=fLastCe; i++) {//without root
      if( fCells[i]->GetStat() == 1 ) {
	 driv =  TMath::Abs( fCells[i]->GetDriv());
	 // apply RMS-cut for all options
	 if (CutRMSmin()){  
	    // calc error on rms, but how?
	    bCutRMS = driv > GetRMSmin() /*&& driv > driv_err*/;
	    Log() << kINFO << "rms[cell "<<i<<"]=" << driv << Endl;
	 }

	 // apply Nmin-cut
	 if (CutNmin())
	    bCutNmin = GetCellEvents(fCells[i]) > GetNmin();

	 // choose cell
	 if(driv > drivMax && bCutNmin && bCutRMS) {
	    drivMax = driv;
	    iCell = i;
	 }
      }
   }
   //   cout << "drivMax: " << drivMax << " iCell: " << iCell << " xdiv: " 
   //	<< fCells[iCell]->GetXdiv() << " GetDriv: " << fCells[iCell]->GetDriv() << endl;

   if (iCell == -1){
     if (!bCutNmin)
	 Log() << kVERBOSE << "Warning: No cell with more than " << GetNmin() << " events found!" << Endl;
      else if (!bCutRMS)
	 Log() << kVERBOSE << "Warning: No cell with RMS/mean > " << GetRMSmin() << " found!" << Endl;
      else
	 Log() << kWARNING << "Warning: PDEFoam::PeekMax: no more candidate cells (drivMax>0) found for further splitting." << Endl;
   }

   return(iCell);
}

//_____________________________________________________________________________________________
Int_t TMVA::PDEFoam::Divide(PDEFoamCell *cell)
{
   // Internal subrogram used by Initialize.
   // It divides cell iCell into two daughter cells.
   // The iCell is retained and tagged as inactive, daughter cells are appended
   // at the end of the buffer.
   // New vertex is added to list of vertices.
   // List of active cells is updated, iCell removed, two daughters added
   // and their properties set with help of MC sampling (PDEFoam_Explore)
   // Returns Code RC=-1 of buffer limit is reached,  fLastCe=fnBuf.

   Double_t xdiv;
   Int_t   kBest;

   if(fLastCe+1 >= fNCells) Log() << kFATAL << "Buffer limit is reached, fLastCe=fnBuf" << Endl;

   cell->SetStat(0); // reset cell as inactive
   fNoAct--;

   xdiv  = cell->GetXdiv();
   kBest = cell->GetBest();
   if( kBest<0 || kBest>=fDim ) Log() << kFATAL << "Wrong kBest" << Endl;

   //////////////////////////////////////////////////////////////////
   //           define two daughter cells (active)                 //
   //////////////////////////////////////////////////////////////////

   Int_t d1 = CellFill(1,   cell);
   Int_t d2 = CellFill(1,   cell);
   cell->SetDau0((fCells[d1]));
   cell->SetDau1((fCells[d2]));
   Explore( (fCells[d1]) );
   Explore( (fCells[d2]) );
   return 1;
} // PDEFoam_Divide

//_________________________________________________________________________________________
void TMVA::PDEFoam::MakeActiveList()
{
   // Internal subrogram used by Initialize.
   // It finds out number of active cells fNoAct,
   // creates list of active cell fCellsAct and primary cumulative fPrimAcu.
   // They are used during the MC generation to choose randomly an active cell.

   Long_t n, iCell;
   Double_t sum;
   // flush previous result
   if(fPrimAcu  != 0) delete [] fPrimAcu;
   if(fCellsAct != 0) delete fCellsAct;

   // Allocate tables of active cells
   fCellsAct = new TRefArray();

   // Count Active cells and find total Primary
   // Fill-in tables of active cells

   fPrime = 0.0; n = 0;
   for(iCell=0; iCell<=fLastCe; iCell++) { 
      if (fCells[iCell]->GetStat()==1) {
         fPrime += fCells[iCell]->GetPrim();
         fCellsAct->Add(fCells[iCell]);
         n++;
      }
   }

   if(fNoAct != n)  Log() << kFATAL << "Wrong fNoAct              " << Endl;
   if(fPrime == 0.) Log() << kFATAL << "Integrand function is zero" << Endl;

   fPrimAcu  = new  Double_t[fNoAct]; // cumulative primary for MC generation
   if( fCellsAct==0 || fPrimAcu==0 ) Log() << kFATAL << "Cant allocate fCellsAct or fPrimAcu" << Endl;

   sum =0.0;
   for(iCell=0; iCell<fNoAct; iCell++) {
      sum = sum + ( (PDEFoamCell *) (fCellsAct->At(iCell)) )->GetPrim()/fPrime;
      fPrimAcu[iCell]=sum;
   }

} //MakeActiveList

//___________________________________________________________________________________________
void TMVA::PDEFoam::GenerCel2(PDEFoamCell *&pCell)
{
   // Internal subprogram.
   // Return randomly chosen active cell with probability equal to its
   // contribution into total driver integral using interpolation search.

   Long_t  lo, hi, hit;
   Double_t fhit, flo, fhi;
   Double_t random;

   random=fPseRan->Rndm();
   lo  = 0;              hi =fNoAct-1;
   flo = fPrimAcu[lo];  fhi=fPrimAcu[hi];
   while(lo+1<hi) {
      hit = lo + (Int_t)( (hi-lo)*(random-flo)/(fhi-flo)+0.5);
      if (hit<=lo)
         hit = lo+1;
      else if(hit>=hi)
         hit = hi-1;
      fhit=fPrimAcu[hit];
      if (fhit>random) {
         hi = hit;
         fhi = fhit;
      } else {
         lo = hit;
         flo = fhit;
      }
   }
   if (fPrimAcu[lo]>random)
      pCell = (PDEFoamCell *) fCellsAct->At(lo);
   else
      pCell = (PDEFoamCell *) fCellsAct->At(hi);
}       // PDEFoam::GenerCel2

//__________________________________________________________________________________________
Double_t TMVA::PDEFoam::Eval(Double_t *xRand, Double_t &event_density)
{
   // Internal subprogram.
   // Evaluates distribution to be generated.

   Double_t result = 0;

   if(!fRho) {   //interactive mode
      Log() << kFATAL << "No binary tree given!" << Endl;
   } else {       //compiled mode
      result=fRho->Density(fDim,xRand,event_density);
   }

   return result;
}

//_____________________________________________________________________________________________
void TMVA::PDEFoam::Grow()
{
   // Internal subrogram used by Initialize.
   // It grow new cells by the binary division process.
   // This function is overridden by the PDEFoam class to stop the foam buildup process
   // if one of the cut conditions stop the cell split.

   fTimer->Init(fNCells);

   Long_t iCell;
   PDEFoamCell* newCell;

   while ( (fLastCe+2) < fNCells ) {  // this condition also checked inside Divide
      iCell   = PeekMax();            // peek up cell with maximum driver integral
      if ( (iCell<0) || (iCell>fLastCe) ) {
	 Log() << kVERBOSE << "Break: "<< fLastCe+1 << " cells created" << Endl;
	 // remove remaining empty cells
	 for (Long_t jCell=fLastCe+1; jCell<fNCells; jCell++)
	    delete fCells[jCell];
	 fNCells = fLastCe+1;
	 break;
      }
      newCell = fCells[iCell];

      OutputGrow();

      if ( Divide( newCell )==0) break;  // and divide it into two
   }
   OutputGrow( kTRUE );
   CheckAll(1);   // set arg=1 for more info
      
   Log() << kINFO << GetNActiveCells() << " active cells created" << Endl;
}// Grow

//___________________________________________________________________________________________
void TMVA::PDEFoam::MakeEvent(void)
{
   // User subprogram.
   // It generates randomly point/vector according to user-defined distribution.
   // Prior initialization with help of Initialize() is mandatory.
   // Generated MC point/vector is available using GetMCvect and the MC weight with GetMCwt.
   // MC point is generated with wt=1 or with variable weight, see OptRej switch.

   Int_t      j;
   Double_t   wt,dx,mcwt;
   PDEFoamCell *rCell;
   //
   //********************** MC LOOP STARS HERE **********************
 ee0:
   GenerCel2(rCell);   // choose randomly one cell

   MakeAlpha();

   PDEFoamVect  cellPosi(fDim); PDEFoamVect  cellSize(fDim);
   rCell->GetHcub(cellPosi,cellSize);
   for(j=0; j<fDim; j++)
      fMCvect[j]= cellPosi[j] +fAlpha[j]*cellSize[j];
   dx = rCell->GetVolume();      // Cartesian volume of the Cell
   //  weight average normalized to PRIMARY integral over the cell

   wt=dx*Eval(fMCvect);

   mcwt = wt / rCell->GetPrim();  // PRIMARY controls normalization
   fNCalls++;
   fMCwt   =  mcwt;
   // accumulation of statistics for the main MC weight
   fSumWt  += mcwt;           // sum of Wt
   fSumWt2 += mcwt*mcwt;      // sum of Wt**2
   fNevGen++;                 // sum of 1d0
   fWtMax  =  TMath::Max(fWtMax, mcwt);   // maximum wt
   fWtMin  =  TMath::Min(fWtMin, mcwt);   // minimum wt
   fMCMonit->Fill(mcwt);
   fHistWt->Fill(mcwt,1.0);          // histogram
   //*******  Optional rejection ******
   if(fOptRej == 1) {
      Double_t random;
      random=fPseRan->Rndm();
      if( fMaxWtRej*random > fMCwt) goto ee0;  // Wt=1 events, internal rejection
      if( fMCwt<fMaxWtRej ) {
         fMCwt = 1.0;                  // normal Wt=1 event
      } else {
         fMCwt = fMCwt/fMaxWtRej;    // weight for overweighted events! kept for debug
         fSumOve += fMCwt-fMaxWtRej; // contribution of overweighted
      }
   }
   //********************** MC LOOP ENDS HERE **********************
} // MakeEvent

//_________________________________________________________________________________
void TMVA::PDEFoam::GetMCvect(Double_t *MCvect)
{
   // User may get generated MC point/vector with help of this method

   for ( Int_t k=0 ; k<fDim ; k++) *(MCvect +k) = fMCvect[k];
}//GetMCvect

//___________________________________________________________________________________
Double_t TMVA::PDEFoam::GetMCwt(void)
{
   // User may get weight MC weight using this method

   return(fMCwt);
}
//___________________________________________________________________________________
void TMVA::PDEFoam::GetMCwt(Double_t &mcwt)
{
   // User may get weight MC weight using this method

   mcwt=fMCwt;
}

//___________________________________________________________________________________
Double_t TMVA::PDEFoam::MCgenerate(Double_t *MCvect)
{
   // User subprogram which generates MC event and returns MC weight

   MakeEvent();
   GetMCvect(MCvect);
   return(fMCwt);
}//MCgenerate

//___________________________________________________________________________________
void TMVA::PDEFoam::GetIntegMC(Double_t &mcResult, Double_t &mcError)
{
   // User subprogram.
   // It provides the value of the integral calculated from the averages of the MC run
   // May be called after (or during) the MC run.

   Double_t mCerelat;
   mcResult = 0.0;
   mCerelat = 1.0;
   if (fNevGen>0) {
      mcResult = fPrime*fSumWt/fNevGen;
      mCerelat = sqrt( fSumWt2/(fSumWt*fSumWt) - 1/fNevGen);
   }
   mcError = mcResult *mCerelat;
}//GetIntegMC

//____________________________________________________________________________________
void  TMVA::PDEFoam::GetIntNorm(Double_t& IntNorm, Double_t& Errel )
{
   // User subprogram.
   // It returns NORMALIZATION integral to be combined with the average weights
   // and content of the histograms in order to get proper absolute normalization
   // of the integrand and distributions.
   // It can be called after initialization, before or during the MC run.

   if(fOptRej == 1) {    // Wt=1 events, internal rejection
      Double_t intMC,errMC;
      GetIntegMC(intMC,errMC);
      IntNorm = intMC;
      Errel   = errMC;
   } else {                // Wted events, NO internal rejection
      IntNorm = fPrime;
      Errel   = 0;
   }
}//GetIntNorm

//______________________________________________________________________________________
void  TMVA::PDEFoam::GetWtParams(Double_t eps, Double_t &aveWt, Double_t &wtMax, Double_t &sigma)
{
   // May be called optionally after the MC run.
   // Returns various parameters of the MC weight for efficiency evaluation

   Double_t mCeff, wtLim;
   fMCMonit->GetMCeff(eps, mCeff, wtLim);
   wtMax = wtLim;
   aveWt = fSumWt/fNevGen;
   sigma = sqrt( fSumWt2/fNevGen -aveWt*aveWt );
}//GetmCeff

//_______________________________________________________________________________________
void TMVA::PDEFoam::Finalize(Double_t& IntNorm, Double_t& Errel)
{
   // May be called optionally by the user after the MC run.
   // It provides normalization and also prints some information/statistics on the MC run.

   GetIntNorm(IntNorm,Errel);
   Double_t mcResult,mcError;
   GetIntegMC(mcResult,mcError);
//    Double_t mCerelat= mcError/mcResult;

//    if(fChat>0) {
//       Double_t eps = 0.0005;
//       Double_t mCeff, mcEf2, wtMax, aveWt, sigma;
//       GetWtParams(eps, aveWt, wtMax, sigma);
//       mCeff=0;
//       if(wtMax>0.0) mCeff=aveWt/wtMax;
//       mcEf2 = sigma/aveWt;
//       Double_t driver = fCells[0]->GetDriv();

//       BXOPE;
//       BXTXT("****************************************");
//       BXTXT("******  TMVA::PDEFoam::Finalize    ******");
//       BXTXT("****************************************");
//       BX1I("    NevGen",fNevGen, "Number of generated events in the MC generation   ");
//       BX1I("    nCalls",fNCalls, "Total number of function calls                    ");
//       BXTXT("----------------------------------------");
//       BX1F("     AveWt",aveWt,    "Average MC weight                      ");
//       BX1F("     WtMin",fWtMin,  "Minimum MC weight (absolute)           ");
//       BX1F("     WtMax",fWtMax,  "Maximum MC weight (absolute)           ");
//       BXTXT("----------------------------------------");
//       BX1F("    XPrime",fPrime,  "Primary total integral, R_prime        ");
//       BX1F("    XDiver",driver,   "Driver  total integral, R_loss         ");
//       BXTXT("----------------------------------------");
//       BX2F("    IntMC", mcResult,  mcError,      "Result of the MC Integral");
//       BX1F(" mCerelat", mCerelat,  "Relative error of the MC integral      ");
//       BX1F(" <w>/WtMax",mCeff,     "MC efficiency, acceptance rate");
//       BX1F(" Sigma/<w>",mcEf2,     "MC efficiency, variance/ave_wt");
//       BX1F("     WtMax",wtMax,     "WtMax(esp= 0.0005)            ");
//       BX1F("     Sigma",sigma,     "variance of MC weight         ");
//       if(fOptRej==1) {
//          Double_t avOve=fSumOve/fSumWt;
//          BX1F("<OveW>/<W>",avOve,     "Contrib. of events wt>MaxWtRej");
//       }
//       BXCLO;
//    }
}  // Finalize

//__________________________________________________________________________________________
void TMVA::PDEFoam::ResetPseRan(TRandom3*PseRan)
{
   // User may optionally reset random number generator using this method
   // Usually it is done when FOAM object is restored from the disk.
   // IMPORTANT: this method deletes existing  random number generator registered in the FOAM object.
   // In particular such an object is created by the streamer during the disk-read operation.

   if(fPseRan) {
      Info("ResetPseRan", "Resetting random number generator  \n");
      delete fPseRan;
   }
   SetPseRan(PseRan);
}

//__________________________________________________________________________________________
void TMVA::PDEFoam::SetRho(PDEFoamIntegrand *fun)
{
   // User may use this method to set (register) random number generator used by
   // the given instance of the FOAM event generator. Note that single r.n. generator
   // may serve several FOAM objects.

   if (fun)
      fRho=fun;
   else
      Log() << kFATAL << "Bad function" << Endl;
}

//__________________________________________________________________________________________
void TMVA::PDEFoam::ResetRho(PDEFoamIntegrand *fun)
{
   // User may optionally reset the distribution using this method
   // Usually it is done when FOAM object is restored from the disk.
   // IMPORTANT: this method deletes existing  distribution object registered in the FOAM object.
   // In particular such an object is created by the streamer diring the disk-read operation.
   // This method is used only in very special cases, because the distribution in most cases
   // should be "owned" by the FOAM object and should not be replaced by another one after initialization.

   if(fRho) {
      Info("ResetRho", "!!! Resetting distribution function  !!!\n");
      delete fRho;
   }
   SetRho(fun);
}

//__________________________________________________________________________________________
void TMVA::PDEFoam::SetRhoInt(void *fun)
{
   // User may use this to set pointer to the global function (not descending
   // from PDEFoamIntegrand) serving as a distribution for FOAM.
   // It is useful for simple interactive applications.
   // Note that persistency for FOAM object will not work in the case of such
   // a distribution.

   const Char_t *namefcn = G__p2f2funcname(fun); //name of integrand function
   if(namefcn) {
      fMethodCall=new TMethodCall();
      fMethodCall->InitWithPrototype(namefcn, "Int_t, Double_t *");
   }
   fRho=0;
}

//_____________________________________________________________________________________
void  TMVA::PDEFoam::SetInhiDiv(Int_t iDim, Int_t InhiDiv)
{
   // This can be called before Initialize, after setting kDim
   // It defines which variables are excluded in the process of the cell division.
   // For example 'FoamX->SetInhiDiv(1, 1);' inhibits division of y-variable.

   if(fDim==0) Log() << kFATAL << "SetInhiDiv: fDim=0" << Endl;
   if(fInhiDiv == 0) {
      fInhiDiv = new Int_t[ fDim ];
      for(Int_t i=0; i<fDim; i++) fInhiDiv[i]=0;
   }
   //
   if( ( 0<=iDim) && (iDim<fDim)) {
      fInhiDiv[iDim] = InhiDiv;
   } else
      Log() << kFATAL << "Wrong iDim" << Endl;
}//SetInhiDiv

//______________________________________________________________________________________
void  TMVA::PDEFoam::SetXdivPRD(Int_t iDim, Int_t len, Double_t xDiv[])
{
   // This should be called before Initialize, after setting  kDim
   // It predefines values of the cell division for certain variable iDim.

   Int_t i;

   if(fDim<=0)    Log() << kFATAL << "fDim=0" << Endl;
   if(   len<1 )  Log() << kFATAL << "len<1"  << Endl;
   // allocate list of pointers, if it was not done before
   if(fXdivPRD == 0) {
      fXdivPRD = new PDEFoamVect*[fDim];
      for(i=0; i<fDim; i++)  fXdivPRD[i]=0;
   }
   // set division list for direction iDim in H-cubic space!!!
   if( ( 0<=iDim) && (iDim<fDim)) {
      fOptPRD =1;      // !!!!
      if( fXdivPRD[iDim] != 0)
         Log() << kFATAL << "Second allocation of XdivPRD not allowed" << Endl;
      fXdivPRD[iDim] = new PDEFoamVect(len); // allocate list of division points
      for(i=0; i<len; i++) {
         (*fXdivPRD[iDim])[i]=xDiv[i]; // set list of division points
      }
   } else {
      Log() << kFATAL << "Wrong iDim" << Endl;
   }
   // Priting predefined division points
   Log()<<" SetXdivPRD, idim= "<<iDim<<"  len= "<<len<<"   "<<Endl;
   for(i=0; i<len; i++) {
      Log()<< (*fXdivPRD[iDim])[i] <<"  ";
   }
   Log()<<Endl;
   for(i=0; i<len; i++)  Log()<< xDiv[i] <<"   ";
   Log()<<Endl;
   //
}//SetXdivPRD

//_______________________________________________________________________________________
void TMVA::PDEFoam::CheckAll(Int_t level)
{
   //  User utility, miscellaneous and debug.
   //  Checks all pointers in the tree of cells. This is useful autodiagnostic.
   //  level=0, no printout, failures causes STOP
   //  level=1, printout, failures lead to WARNINGS only

   Int_t errors, warnings;
   PDEFoamCell *cell;
   Long_t iCell;

   errors = 0; warnings = 0;
   if (level==1) Log() <<  "Performing consistency checks for created foam" << Endl;
   for(iCell=1; iCell<=fLastCe; iCell++) {
      cell = fCells[iCell];
      //  checking general rules
      if( ((cell->GetDau0()==0) && (cell->GetDau1()!=0) ) ||
          ((cell->GetDau1()==0) && (cell->GetDau0()!=0) ) ) {
         errors++;
         if (level==1) Log() << kFATAL << "ERROR: Cell's no %d has only one daughter " << iCell << Endl;
      }
      if( (cell->GetDau0()==0) && (cell->GetDau1()==0) && (cell->GetStat()==0) ) {
         errors++;
         if (level==1) Log() << kFATAL << "ERROR: Cell's no %d  has no daughter and is inactive " << iCell << Endl;
      }
      if( (cell->GetDau0()!=0) && (cell->GetDau1()!=0) && (cell->GetStat()==1) ) {
         errors++;
         if (level==1) Log() << kFATAL << "ERROR: Cell's no %d has two daughters and is active " << iCell << Endl;
      }

      // checking parents
      if( (cell->GetPare())!=fCells[0] ) { // not child of the root
         if ( (cell != cell->GetPare()->GetDau0()) && (cell != cell->GetPare()->GetDau1()) ) {
            errors++;
            if (level==1) Log() << kFATAL << "ERROR: Cell's no %d parent not pointing to this cell " << iCell << Endl;
         }
      }

      // checking daughters
      if(cell->GetDau0()!=0) {
         if(cell != (cell->GetDau0())->GetPare()) {
            errors++;
            if (level==1)  Log() << kFATAL << "ERROR: Cell's no %d daughter 0 not pointing to this cell " << iCell << Endl;
         }
      }
      if(cell->GetDau1()!=0) {
         if(cell != (cell->GetDau1())->GetPare()) {
            errors++;
            if (level==1) Log() << kFATAL << "ERROR: Cell's no %d daughter 1 not pointing to this cell " << iCell << Endl;
         }
      }
     if(cell->GetVolume()<1E-50) {
         errors++;
         if(level==1) Log() << kFATAL << "ERROR: Cell no. %d has Volume of <1E-50" << iCell << Endl;
     }
    }// loop after cells;

   // Check for cells with Volume=0
   for(iCell=0; iCell<=fLastCe; iCell++) {
      cell = fCells[iCell];
      if( (cell->GetStat()==1) && (cell->GetVolume()<1E-11) ) {
         errors++;
         if(level==1) Log() << kFATAL << "ERROR: Cell no. " << iCell << " is active but Volume is 0 " <<  Endl;
      }
   }
   // summary
   if(level==1){
     Log() << kINFO << "Check has found " << errors << " errors and " << warnings << " warnings." << Endl;
   }
   if(errors>0){
      Info("CheckAll","Check - found total %d  errors \n",errors);
   }
} // Check

//________________________________________________________________________________________
void TMVA::PDEFoam::PrintCells(void)
{
   // Prints geometry of ALL cells of the FOAM

   Long_t iCell;

   for(iCell=0; iCell<=fLastCe; iCell++) {
      Log()<<"Cell["<<iCell<<"]={ ";
      Log()<<"  "<< fCells[iCell]<<"  ";  // extra DEBUG
      Log()<<Endl;
      fCells[iCell]->Print("1");
      Log()<<"}"<<Endl;
   }
}

//________________________________________________________________________________________
void TMVA::PDEFoam::LinkCells()
{
   // Void function for backward compatibility

   Info("LinkCells", "VOID function for backward compatibility \n");
   return;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetSumCellElements(UInt_t i)
{
   // returns the sum of all cell elements with index i of all active cells
   Double_t intg = 0;
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (fCells[iCell]->GetStat())
         intg += GetCellElement(fCells[iCell], i);
   }
   return intg;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetSumCellIntg()
{
   // returns the sum of all cell integrals of all active cells
   Double_t intg = 0;
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (fCells[iCell]->GetStat())
         intg += fCells[iCell]->GetIntg();
   }
   return intg;
}

//_______________________________________________________________________________________________
UInt_t TMVA::PDEFoam::GetNActiveCells()
{
   // returns number of active cells
   UInt_t count = 0;
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (fCells[iCell]->GetStat())
         count++;
   }
   return count;
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::RemoveEmptyCell( Int_t iCell )
{
   // This function removes a cell iCell, which has a volume equal to zero.
   // It works the following way:
   // 1) find grandparent to iCell
   // 2) set daughter of the grandparent cell to the sister of iCell
   // Result:
   // iCell and its parent are alone standing ==> will be removed

   // get cell volume
   Double_t volume = fCells[iCell]->GetVolume();

   if (!fCells[iCell]->GetStat() || volume>0){
      Log() << "<RemoveEmptyCell>: cell " << iCell 
              << "is not active or has volume>0 ==> doesn't need to be removed" << Endl;
      return;
   }

   // get parent and grandparent Cells
   PDEFoamCell *pCell  = fCells[iCell]->GetPare();
   PDEFoamCell *ppCell = fCells[iCell]->GetPare()->GetPare();

   // get neighbour (sister) to iCell
   PDEFoamCell *sCell;
   if (pCell->GetDau0() == fCells[iCell])
      sCell = pCell->GetDau1();
   else
      sCell = pCell->GetDau0();

   // cross check
   if (pCell->GetIntg() != sCell->GetIntg())
      Log() << kWARNING << "<RemoveEmptyCell> Error: cell integrals are not equal!" 
              << " Intg(parent cell)=" << pCell->GetIntg() 
              << " Intg(sister cell)=" << sCell->GetIntg() 
              << Endl;

   // set daughter of grandparent to sister of iCell
   if (ppCell->GetDau0() == pCell)
      ppCell->SetDau0(sCell);
   else
      ppCell->SetDau1(sCell);

   // set parent of sister to grandparent of of iCell
   sCell->SetPare(ppCell);

   // now iCell and its parent are alone ==> set them inactive
   fCells[iCell]->SetStat(0);
   pCell->SetStat(0);
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::CheckCells( Bool_t remove_empty_cells )
{
   // debug function: checks all cells with respect to critical
   // values, f.e. cell volume, ...

   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!fCells[iCell]->GetStat())
         continue;

      Double_t volume = fCells[iCell]->GetVolume();
      if (volume<1e-10){
         if (volume<=0){
            Log() << kWARNING << "Critical: cell volume negative or zero! volume=" 
                    << volume << " cell number: " << iCell << Endl;
            // fCells[iCell]->Print("1"); // debug output
            if (remove_empty_cells){
               Log() << kWARNING << "Remove cell " << iCell << Endl;
               RemoveEmptyCell(iCell);
            }
         }
         else {
            Log() << kWARNING << "Cell volume close to zero! volume=" 
                    << volume << " cell number: " << iCell << Endl;
         }
      }
   }   
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::PrintCellElements()
{
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!fCells[iCell]->GetStat()) continue;
      
      std::cout << ">>> cell[" << iCell << "] elements: ";
      for (UInt_t i=0; i<GetNElements(); i++)
         std::cout << "[" << i << "; " << GetCellElement(fCells[iCell], i) << "] ";
      std::cout << std::endl;
   }
}

//_______________________________________________________________________________________________
UInt_t TMVA::PDEFoam::GetNInActiveCells()
{
   // returns number of not active cells
   return GetNCells()-GetNActiveCells();
}

//_______________________________________________________________________________________________
UInt_t TMVA::PDEFoam::GetNCells()
{
   // returns the total number of foam cells (active and not active cells)
   return fNCells;
}

//_______________________________________________________________________________________________
Int_t TMVA::PDEFoam::GetSumCellMemory( ECellType ct )
{
   // returns total used memory of cells in bytes
   // The parameter 'ct' describes the cell type: 
   // - kAll - get memory usage of all cells
   // - kActive - get memory usage of all active cells
   // - kInActive - get memory usage of all non-active cells

   UInt_t count = 0;
   for (Long_t iCell=0; iCell<fNCells; iCell++) {
      Int_t cellsize = sizeof(*(fCells[iCell]));

      if (ct==kAll)
         count += cellsize;
      else if (ct==kActive && fCells[iCell]->GetStat() && iCell<fLastCe)
         count += cellsize;
      else if (ct==kInActive && 
               ((!(fCells[iCell]->GetStat()) && iCell<fLastCe) || iCell>=fLastCe) )
         count += cellsize;
   }
   return count;
}

//_______________________________________________________________________________________________
TMVA::PDEFoamCell* TMVA::PDEFoam::GetRootCell()
{
   // returns pointer to root cell
   return fCells[0];
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::ResetCellElements(Bool_t allcells)
{
   // creates a TVectorD object with fNElements in every cell
   // and initializes them by zero.
   // The TVectorD object is used to store classification or 
   // regression data in every foam cell.
   //
   // Parameter:
   //   allcells == true  : create TVectorD on every cell
   //   allcells == false : create TVectorD on active cells with 
   //                       cell index <= fLastCe (default)
   
   if (!fCells || GetNElements()==0) return;

   // delete all old cell elements
   Log() << kVERBOSE << "Delete old cell elements" << Endl;
   for(Long_t iCell=0; iCell<fNCells; iCell++) {
      if (fCells[iCell]->GetElement() != 0){
	 delete dynamic_cast<TVectorD*>(fCells[iCell]->GetElement());
         fCells[iCell]->SetElement(0);
      }
   }

   if (allcells){
      Log() << kVERBOSE << "Reset new cell elements to " 
	    << GetNElements() << " value(s) per cell" << Endl;
      // create new cell elements on every cell
      for(Long_t iCell=0; iCell<fNCells; iCell++) {
	 TVectorD *elem = new TVectorD(GetNElements());

	 for (UInt_t i=0; i<GetNElements(); i++)
	    (*elem)(i) = 0.;

	 fCells[iCell]->SetElement(elem);
      }   
   } else {
      Log() << kVERBOSE << "Reset active cell elements to " 
	    << GetNElements() << " value(s) per cell" << Endl;
      // create new cell elements (only active cells with 
      // cell index <= fLastCe)
      for(Long_t iCell=0; iCell<=fLastCe; iCell++) {
	 // skip inactive cells
	 if (!(fCells[iCell]->GetStat()))
	    continue;

	 TVectorD *elem = new TVectorD(GetNElements());

	 for (UInt_t i=0; i<GetNElements(); i++)
	    (*elem)(i) = 0.;

	 fCells[iCell]->SetElement(elem);
      }   
   }
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::DisplayCellContent()
{
   // prints out general cell informations
   Double_t total_counts = 0;
   Double_t max = GetCellEntries(fCells[0]);
   Double_t min = GetCellEntries(fCells[0]);

   // Reset Cell Integrals
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      total_counts += GetCellEntries(fCells[iCell]);
      
      if (GetCellEntries(fCells[iCell]) < min)
         min = GetCellEntries(fCells[iCell]);
      if (GetCellEntries(fCells[iCell]) > max)
         max = GetCellEntries(fCells[iCell]);
   }

   Log() << kVERBOSE << "DEBUG: Total Events in Foam: " << total_counts << Endl;
   Log() << kVERBOSE << "DEBUG: min cell entry: " << min << Endl;
   Log() << kVERBOSE << "DEBUG: max cell entry: " << max << Endl;
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::CalcCellTarget()
{
   // Calculate average cell target in every cell and save them to the cell.
   // This function is called when the Mono target regression option is set.

   // loop over cells
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!(fCells[iCell]->GetStat()))
         continue;

      Double_t N_ev  = GetCellElement(fCells[iCell], 0); // get number of events
      Double_t tar   = GetCellElement(fCells[iCell], 1); // get sum of targets

      if (N_ev > 1e-10){
         SetCellElement(fCells[iCell], 0, tar/N_ev);  // set average target
         SetCellElement(fCells[iCell], 1, 0.0 );      // ???
      }
      else {
         SetCellElement(fCells[iCell], 0, 0.0 ); // set mean target
         SetCellElement(fCells[iCell], 1, 1.  ); // set mean target error
      }
   }
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::CalcCellDiscr()
{
   // Calc discriminator and its error for every cell and save it to the cell.
   // This function is called when the fSigBgSeparated==False option is set.

   // loop over cells
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!(fCells[iCell]->GetStat()))
         continue;

      Double_t N_sig = GetCellElement(fCells[iCell], 0); // get number of signal events
      Double_t N_bg  = GetCellElement(fCells[iCell], 1); // get number of bg events

      if (N_sig<0.) {
	Log() << kWARNING << "Negative number of signal events in cell " << iCell 
	      << ": " << N_sig << ". Set to 0." << Endl;
	N_sig=0.;
      }
      if (N_bg<0.) {
	Log() << kWARNING << "Negative number of background events in cell " << iCell 
	      << ": " << N_bg << ". Set to 0." << Endl;
	N_bg=0.;
      }


      if (N_sig+N_bg > 1e-10){
         SetCellElement(fCells[iCell], 0, N_sig/(N_sig+N_bg));  // set discriminator
         SetCellElement(fCells[iCell], 1, TMath::Sqrt( TMath::Power ( N_sig/TMath::Power(N_sig+N_bg,2),2)*N_sig +
                                                       TMath::Power ( N_bg /TMath::Power(N_sig+N_bg,2),2)*N_bg ) ); // set discriminator error

      }
      else {
         SetCellElement(fCells[iCell], 0, 0.5); // set discriminator
         SetCellElement(fCells[iCell], 1, 1. ); // set discriminator error
      }
   }
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDiscr( std::vector<Float_t> xvec, EKernel kernel )
{
   // Get discriminator saved in cell (previously calculated in CalcCellDiscr())
   // which encloses the coordinates given in xvec.
   // This function is used, when the fSigBgSeparated==False option is set
   // (unified foams).

   Double_t result = 0.;

   // transform xvec
   std::vector<Float_t> txvec = VarTransform(xvec);

   // find cell
   PDEFoamCell *cell= FindCell(txvec);

   if (!cell) return -999.;

   if (kernel == kNone) result = GetCellDiscr(cell);
   else if (kernel == kGaus) {
      Double_t norm = 0.;

      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_discr = GetCellDiscr(fCells[iCell]);
         Double_t gau        = WeightGaus(fCells[iCell], txvec);

         result += gau * cell_discr;
         norm   += gau;
      }

      result /= norm;
   } 
   else if (kernel == kLinN) {
      result = WeightLinNeighbors(txvec, kDiscriminator);
   }
   else {
      Log() << kFATAL << "GetCellDiscr: ERROR: wrong kernel!" << Endl;
      result = -999.0;
   }

   return result;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDiscrError( std::vector<Float_t> xvec )
{
   // Return discriminator error of cell (previously calculated in CalcCellDiscr()).
   // This function is used, when the fSigBgSeparated==False option is set
   // (unified foams).

   // find cell
   PDEFoamCell *cell= FindCell(VarTransform(xvec));
   if (!cell) return -999.;

   return GetCellDiscrError(cell);
}

//_______________________________________________________________________________________________
void TMVA::PDEFoam::FillFoamCells(const Event* ev, EFoamType ft, Bool_t NoNegWeights)
{
   // This function fills an event into the foam.
   //
   // In case of ft==kMonoTarget this function prepares the
   // calculation of the average target value in every cell.  Note,
   // that only target 0 is saved in the cell!
   //
   // In case of ft==kDiscr this function prepares the calculation of
   // the cell discriminator in every cell.
   //
   // If 'NoNegWeights' is true, an event with negative weight will
   // not be filled into the foam.  (Default value: false)

   std::vector<Float_t> values  = ev->GetValues();
   std::vector<Float_t> targets = ev->GetTargets();
   Float_t weight               = ev->GetOriginalWeight();
   
   if(NoNegWeights && weight<=0)
      return;

   if (ft == kMultiTarget)
      values.insert(values.end(), targets.begin(), targets.end());

   // find corresponding foam cell
   PDEFoamCell *cell = FindCell(VarTransform(values));
   if (!cell) {
      Log() << kFATAL << "<PDEFoam::FillFoamCells>: No cell found!" << Endl; 
      return;
   }

   // Add events to cell
   if (ft == kSeparate || ft == kMultiTarget){
      // 0. Element: Number of events
      // 1. Element: RMS
      SetCellElement(cell, 0, GetCellElement(cell, 0) + weight);
      SetCellElement(cell, 1, GetCellElement(cell, 1) + weight*weight);
   } else if (ft == kDiscr){
      // 0. Element: Number of signal events
      // 1. Element: Number of background events times normalization
      if (ev->IsSignal())
	 SetCellElement(cell, 0, GetCellElement(cell, 0) + weight);
      else
	 SetCellElement(cell, 1, GetCellElement(cell, 1) + weight);
   } else if (ft == kMonoTarget){
      // 0. Element: Number of events
      // 1. Element: Target 0
      SetCellElement(cell, 0, GetCellElement(cell, 0) + weight);
      SetCellElement(cell, 1, GetCellElement(cell, 1) + weight*targets.at(0));
   }   
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellMean( std::vector<Float_t> xvec )
{  // Return mean of event distribution in cell that encloses xvec

   PDEFoamCell *cell = FindCell(VarTransform(xvec));

   if (!cell) {
      Log() << kFATAL << "<GetCellMean> ERROR: No cell found! " << Endl; 
      return -999.;
   }

   return GetCellMean(cell);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellRMS( std::vector<Float_t> xvec )
{  // Get RMS event distribution in cell that encloses xvec

   PDEFoamCell *cell = FindCell(VarTransform(xvec));

   if (!cell) {
      Log() << kFATAL << "<GetCellRMS> ERROR: No cell found! " << Endl; 
      return -999.;
   }

   return GetCellRMS(cell);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellRegValue0( std::vector<Float_t> xvec, EKernel kernel )
{  
   // Get regression value 0 from cell that contains xvec.
   // This function is used when the MultiTargetRegression==False option is set.

   Double_t result = 0.;

   std::vector<Float_t> txvec = VarTransform(xvec);
   PDEFoamCell *cell          = FindCell(txvec);

   if (!cell) {
      Log() << kFATAL << "<GetCellRegValue0> ERROR: No cell found!" << Endl; 
      return -999.;
   }

   if (kernel == kNone){
      result = GetCellRegValue0(cell);
   }
   else if (kernel == kGaus){
      // return gaus weighted cell density

      Double_t norm = 0.;
      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_val = GetCellRegValue0(fCells[iCell]);
         Double_t gau      = WeightGaus(fCells[iCell], txvec);

         result += gau * cell_val;
         norm   += gau;
      }
      result /= norm;
   }
   else if (kernel == kLinN) {
      result = WeightLinNeighbors(txvec, kTarget0);
   }
   else {
      Log() << kFATAL << "ERROR: unknown kernel!" << Endl; 
      return -999.;
   }

   return result;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellTarget( UInt_t target_number, std::vector<Float_t> tvals, ETargetSelection ts )
{
   // This function is used when the MultiTargetRegression==True
   // option is set.  It calculates the mean target or most probable
   // target value with the id 'target_number' if 'tvals' variables
   // are given ('tvals' does not contain targets)
   //
   // Parameters:
   // - tvals - transformed event variables (element of [0,1]) (no targets!)
   // - target_number - number of target (i.e. 0= first target)
   // - ts - method of target selection (kMean, kMpv)
   //
   // Result: 
   // mean target or most probable target over all cells which first
   // coordinates enclose 'tvals'
   
   Double_t result       = 0.;
   Double_t norm         = 0.;
   Double_t max_dens     = 0.;
   UInt_t   cell_count   = 0;
   const Double_t xsmall = 1.e-7; 

   if (ts==kMpv)
      result = fXmin[target_number+tvals.size()];

   // loop over all cells
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!(fCells[iCell]->GetStat())) continue;

      // find cells which first coordinates (non-target coordinates)
      // fit with given variables 'tvals'
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      fCells[iCell]->GetHcub(cellPosi, cellSize);

      Bool_t CellOK = kTRUE;
      for (UInt_t k=0; k<tvals.size(); k++){
         if (!( tvals.at(k) >  cellPosi[k]-xsmall && 
		tvals.at(k) <= cellPosi[k]+cellSize[k]+xsmall )){
            CellOK = kFALSE;
            break;
         }
      }
      if (!CellOK)
         continue;
      cell_count++;

      // cell fits -> start projection
      Double_t cell_density = GetCellDensity(fCells[iCell]);

      if (ts==kMean){
	 // sum cell density times cell center
	 result += cell_density * VarTransformInvers(target_number+tvals.size(), 
						     cellPosi[target_number+tvals.size()]
						     +0.5*cellSize[target_number+tvals.size()]);
	 norm   += cell_density;
      } else {
	 // get cell center with maximum event density
	 if (cell_density > max_dens){
	    max_dens = cell_density;
	    result = VarTransformInvers(target_number+tvals.size(), 
					cellPosi[target_number+tvals.size()]
					+0.5*cellSize[target_number+tvals.size()]);
	 }
      }

   } // cell loop

   if (cell_count<1 || (ts==kMean && norm<1.0e-15))
      return (fXmax[target_number+tvals.size()]-fXmin[target_number+tvals.size()])/2.;

   // calc mean cell density
   if (ts==kMean)
      result /= norm;

   return result;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetProjectedRegValue( UInt_t target_number, std::vector<Float_t> vals, 
                                              EKernel kernel, ETargetSelection ts )
{
   // This function is used when the MultiTargetRegression==True option is set.
   // Returns regression value i, given the event variables 'vals'.
   // Note: number of foam dimensions = number of variables + number of targets
   //
   // Parameters:
   // - target_number - number of target (i.e. target_number=0 specifies the first target)
   // - vals - event variables (no targets)
   // - kernel - used kernel (None or Gaus)
   // - ts - method of target selection (Mean or Mpv)

   if (target_number+vals.size() > UInt_t(GetTotDim())){
      Log() << kFATAL << "ERROR: wrong dimension given!" << Endl;
      return 0;
   }

   // checkt whether vals are within foam borders.
   // if not -> push it into foam
   const Double_t xsmall = 1.e-7; 
   for (UInt_t l=0; l<vals.size(); l++) {
      if (vals.at(l) <= fXmin[l]){
         vals.at(l) = fXmin[l] + xsmall;
      }
      else if (vals.at(l) >= fXmax[l]){
         vals.at(l) = fXmax[l] - xsmall;
      }
   }
   
   // transform variables (vals)
   std::vector<Float_t> txvec = VarTransform(vals);

   // result value and norm
   Double_t result = 0.;

   // choose kernel
   /////////////////// no kernel //////////////////
   if (kernel == kNone)
      result = GetCellTarget(target_number, txvec, ts);
   ////////////////////// Gaus kernel //////////////////////
   else if (kernel == kGaus){

      Double_t norm = 0.;

      // loop over all active cells to calc gaus weighted target values
      for (Long_t ice=0; ice<=fLastCe; ice++) {
         if (!(fCells[ice]->GetStat())) continue;

	 // weight with gaus only in non-target dimensions!
         Double_t gau = WeightGaus(fCells[ice], txvec, vals.size());

         PDEFoamVect cellPosi(GetTotDim()), cellSize(GetTotDim());
         fCells[ice]->GetHcub(cellPosi, cellSize);

         // fill new vector with coordinates of new cell
         std::vector<Float_t> new_vec;
         for (UInt_t k=0; k<txvec.size(); k++)
            new_vec.push_back(cellPosi[k] + 0.5*cellSize[k]);

         Double_t val = GetCellTarget(target_number, new_vec, ts);
         result += gau * val;
         norm   += gau;
      }

      if (norm<1.0e-20){
         // evtl. change algorithm -> find nearest cell! (analogus to FindCell()?)
         Log() << kWARNING << "Warning: norm too small!" << Endl;
         return 0.;
      }

      result /= norm; // this variable now contains gaus weighted average target value
      
   } // end gaus kernel
   else {
      Log() << kFATAL << "<GetProjectedRegValue> ERROR: unsupported kernel!" << Endl;
      result = 0.;
   }

   return result;
} 

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellEntries( std::vector<Float_t> xvec )
{  // Returns number of events in cell that encloses xvec

   PDEFoamCell *cell= FindCell(VarTransform(xvec));

   if (!cell) {
      Log() << kFATAL << "<GetCellEntries> No cell found! " << Endl; 
      return -999.;
   }

   return GetCellEntries(cell);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDensity( PDEFoamCell *cell )
{
   // Returns cell density = number of events in cell / cell volume

   Double_t volume  = cell->GetVolume();
   Double_t entries = GetCellEntries(cell);
   if (entries <= 0)
      return 0.;
   if (volume > 1.0e-10 ){
      return GetCellEntries(cell)/volume;
   }
   else {
      if (volume<=0){
         cell->Print("1"); // debug output
         Log() << kWARNING << "<GetCellDensity(cell)>: ERROR: cell volume negative or zero!"
                 << " ==> return cell density 0!" 
                 << " cell volume=" << volume 
                 << " cell entries=" << GetCellEntries(cell) << Endl;
         return 0;
      }
      else {
         Log() << kWARNING << "<GetCellDensity(cell)>: WARNING: cell volume close to zero!" 
                 << " cell volume: " << volume << Endl;
      }
   }
   return GetCellEntries(cell)/volume;   
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDensity( std::vector<Float_t> xvec, EKernel kernel )
{ 
   // Returns density (=number of entries / volume) of cell that encloses 'xvec'.
   // This function is called by GetMvaValue() in case of two separated foams
   // (signal and background).
   // 'kernel' can be either kNone or kGaus.

   Double_t result = 0;
   std::vector<Float_t> txvec = VarTransform(xvec);
   PDEFoamCell *cell          = FindCell(txvec);

   if (!cell) {
      Log() << kFATAL << "<GetCellDensity(event)> ERROR: No cell found!" << Endl; 
      return -999.;
   }

   if (kernel == kNone){
      // return cell entries over cell volume
      return GetCellDensity(cell);
   }
   else if (kernel == kGaus){
      // return gaus weighted cell density

      Double_t norm = 0.;

      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_dens = GetCellDensity(fCells[iCell]);
         Double_t gau       = WeightGaus(fCells[iCell], txvec);

         result += gau * cell_dens;
         norm   += gau;
      }

      result /= norm;
   }
   else if (kernel == kLinN){
      result = WeightLinNeighbors(txvec, kDensity);
   }
   else {
      Log() << kFATAL << "<GetCellDensity(event)> ERROR: unknown kernel!" << Endl; 
      return -999.;
   }

   return result;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellRegValue0( PDEFoamCell* cell )
{
   // Returns the content of cell element 0, in which the average target value 
   // (target 0) is saved.
   // Only used if MultiTargetRegression==False.
   return GetCellElement(cell, 0);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDiscr( PDEFoamCell* cell )
{
   // Returns the content of cell element 0, in which the cell discriminator is saved.
   // Only used if fSigBgSeparated==False.
   return GetCellElement(cell, 0);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellDiscrError( PDEFoamCell* cell )
{
   // Returns the content of cell element 1, in which the cell discriminator error is saved.
   // Only used if fSigBgSeparated==False.
   return GetCellElement(cell, 1);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellMean( PDEFoamCell* cell )
{
   // Returns the mean of the event distribution in the 'cell'.
   // Only used if fSigBgSeparated==True.
   return cell->GetIntg();
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellRMS( PDEFoamCell* cell )
{
   // Returns the RMS of the event distribution in the 'cell'.
   // Only used if fSigBgSeparated==True.
   return cell->GetDriv();
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellRMSovMean( PDEFoamCell* cell )
{
   // returns RMS/Mean for given cell
   if (GetCellMean(cell)!=0)
      return GetCellRMS(cell)/GetCellMean(cell);
   else
      return 0;
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellEntries( PDEFoamCell* cell )
{
   // Returns the number of events, saved in the 'cell'.
   // Only used if fSigBgSeparated==True.
   return GetCellElement(cell, 0);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellEvents( PDEFoamCell* cell )
{
   // Returns the number of events, saved in the 'cell' during foam build-up.
   // Only used during foam build-up!
   return GetCellElement(cell, 0);
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::WeightLinNeighbors( std::vector<Float_t> txvec, ECellValue cv )
{ 
   // results the cell value, corresponding to txvec, weighted by the
   // neighor cells via a linear function
   //
   // Parameters
   //  - txvec - event vector, transformed to interval [0,1]
   //  - cv - cell value to be weighted

   Double_t result = 0.;
   const Double_t xoffset = 1.e-6; 

   if (txvec.size() != UInt_t(GetTotDim()))
      Log() << kFATAL << "Wrong dimension of event variable!" << Endl;

   PDEFoamCell *cell= FindCell(txvec);
   PDEFoamVect cellSize(GetTotDim());
   PDEFoamVect cellPosi(GetTotDim());
   cell->GetHcub(cellPosi, cellSize);

   for (Int_t dim=0; dim<GetTotDim(); dim++) {
      std::vector<Float_t> ntxvec = txvec;
      Double_t mindist;
      PDEFoamCell *mindistcell = 0;
      mindist = (txvec[dim]-cellPosi[dim])/cellSize[dim];
      if (mindist<0.5) { // left neighbour 
	 ntxvec[dim] = cellPosi[dim]-xoffset;
	 mindistcell = FindCell(ntxvec);
      } else { // right neighbour
	 mindist=1-mindist;
	 ntxvec[dim] = cellPosi[dim]+cellSize[dim]+xoffset;
	 mindistcell = FindCell(ntxvec);	 
      }
      Double_t cellval = 0;
      Double_t mindistcellval = 0;
      if (cv==kDiscriminator){
	 cellval        = GetCellDiscr(cell);
	 mindistcellval = GetCellDiscr(mindistcell);
      } else if (cv==kTarget0){
	 cellval        = GetCellRegValue0(cell);
	 mindistcellval = GetCellRegValue0(mindistcell);
      } else if (cv==kDensity){
	 cellval        = GetCellDensity(cell);
	 mindistcellval = GetCellDensity(mindistcell);
      } else if (cv==kNev){
	 cellval        = GetCellEntries(cell);
	 mindistcellval = GetCellEntries(mindistcell);
      } else if (cv==kMeanValue){
	 cellval        = GetCellMean(cell);
	 mindistcellval = GetCellMean(mindistcell);
      } else if (cv==kRms){
	 cellval        = GetCellRMS(cell);
	 mindistcellval = GetCellRMS(mindistcell);
      } else if (cv==kRmsOvMean){
	 cellval        = GetCellRMSovMean(cell);
	 mindistcellval = GetCellRMSovMean(mindistcell);
      } else {
	 Log() << kFATAL << "<WeightLinNeighbors>: unsupported option" << Endl;
      }
      result += cellval        * (0.5 + mindist);
      result += mindistcellval * (0.5 - mindist);
   }
   return result/GetTotDim();
}

//_______________________________________________________________________________________________
Double_t TMVA::PDEFoam::WeightGaus( PDEFoamCell* cell, std::vector<Float_t> txvec, UInt_t dim )
{ 
   // Returns the gauss weight between the 'cell' and a given coordinate 'txvec'.
   //
   // Parameters:
   // - cell - the cell
   // - txvec - the transformed event variables (in [0,1]) (coordinates <0 are set to 0, >1 are set to 1)
   // - dim - number of dimensions for the calculation of the euclidean distance.
   //   If dim=0, all dimensions of the foam are taken.  Else only the first 'dim'
   //   coordinates of 'txvec' are used for the calculation of the euclidean distance.
   //
   // Returns:
   // exp(-(d/sigma)^2/2), where 
   //  - d - is the euclidean distance between 'txvec' and the point of the 'cell'
   //    which is most close to 'txvec' (in order to avoid artefacts because of the 
   //    form of the cells).
   //  - sigma = 1/VolFrac

   // get cell coordinates
   PDEFoamVect cellSize(GetTotDim());
   PDEFoamVect cellPosi(GetTotDim());
   cell->GetHcub(cellPosi, cellSize);

   // calc normalized distance
   UInt_t dims;            // number of dimensions for gaus weighting
   if (dim == 0) 
      dims = GetTotDim();  // use all dimensions of cell txvec for weighting
   else if (dim <= UInt_t(GetTotDim()))
      dims = dim;          // use only 'dim' dimensions of cell txvec for weighting
   else {
      Log() << kFATAL << "ERROR: too many given dimensions for Gaus weight!" << Endl;
      return 0.;
   }

   // calc position of nearest edge of cell
   std::vector<Float_t> cell_center;
   for (UInt_t i=0; i<dims; i++){
     if (txvec[i]<0.) txvec[i]=0.;
     if (txvec[i]>1.) txvec[i]=1.;
      //cell_center.push_back(cellPosi[i] + (0.5*cellSize[i]));
      if (cellPosi[i] > txvec.at(i))
         cell_center.push_back(cellPosi[i]);
      else if (cellPosi[i]+cellSize[i] < txvec.at(i))
         cell_center.push_back(cellPosi[i]+cellSize[i]);
      else
         cell_center.push_back(txvec.at(i));
   }

   Double_t distance = 0.; // distance for weighting
   for (UInt_t i=0; i<dims; i++)
      distance += TMath::Power(txvec.at(i)-cell_center.at(i), 2);
   distance = TMath::Sqrt(distance);

   Double_t width = 1./GetPDEFoamVolumeFraction();
   if (width < 1.0e-10) 
      Log() << kWARNING << "Warning: wrong volume fraction: " << GetPDEFoamVolumeFraction() << Endl;
   
   // weight with Gaus with sigma = 1/VolFrac
   return TMath::Gaus(distance, 0, width, kFALSE);
}

//_______________________________________________________________________________________________
TMVA::PDEFoamCell* TMVA::PDEFoam::FindCell( std::vector<Float_t> xvec )
{
   // Find cell that contains xvec
   //
   // loop to find cell that contains xvec
   // start from root Cell, uses binary tree to
   // find cell quickly
   //
   // DD 10.12.2007 
   // cleaned up a bit and improved performance

   PDEFoamVect  cellPosi0(GetTotDim()), cellSize0(GetTotDim());
   PDEFoamCell *cell, *cell0;

   cell=fCells[0]; // start with root cell
   Int_t idim=0;
   while (cell->GetStat()!=1) { //go down binary tree until cell is found
      idim=cell->GetBest();  // dimension that changed
      cell0=cell->GetDau0(); 
      cell0->GetHcub(cellPosi0,cellSize0);

      if (xvec.at(idim)<=cellPosi0[idim]+cellSize0[idim])
         cell=cell0;
      else
         cell=(cell->GetDau1());
   } 
   return cell;
}

//_________________________________________________________________________________________
TH1D* TMVA::PDEFoam::Draw1Dim( const char *opt, Int_t nbin ) 
{
   // Draws 1-dimensional foam (= histogram)
   //
   // Parameters:
   // - opt - "mean", "nev", "rms", "rms_ov_mean", "discr" , "discr_error",
   //         "MonoTargetRegression", "MultiTargetRegression"
   // - nbin - number of bins of result histogram
   //
   // Warning: This function is not well tested!

   if (!(strcmp(opt,"mean")==0 || 
	 strcmp(opt,"nev")==0 || 
	 strcmp(opt,"rms")==0 || 
	 strcmp(opt,"rms_ov_mean")==0 || 
	 strcmp(opt,"discr")==0 || 
	 strcmp(opt,"discr_error")==0 ||
         strcmp(opt,"MonoTargetRegression")==0 || 
         strcmp(opt,"MultiTargetRegression")==0) || 
       GetTotDim()!=1 )
      return 0;

   char hname[100]; char htit[100];
   sprintf(htit,"1-dimensional Foam: %s", opt);
   sprintf(hname,"h%s",opt);

   TH1D* h1=(TH1D*)gDirectory->Get(hname);
   if (h1) delete h1;
   h1= new TH1D(hname, htit, nbin, fXmin[0], fXmax[0]);

   if (!h1) Log() << kFATAL << "ERROR: Can not create histo" << hname << Endl;

   std::vector<Float_t> xvec(GetTotDim(), 0.);

   // loop over all bins
   for (Int_t ibinx=1; ibinx<=nbin; ibinx++) { //loop over  x-bins
      xvec.at(0) = h1->GetBinCenter(ibinx);

      // transform xvec
      std::vector<Float_t> txvec = VarTransform(xvec);

      // loop over all active cells
      for (Long_t iCell=0; iCell<=fLastCe; iCell++) { 
         if (!(fCells[iCell]->GetStat())) continue; // cell not active -> continue
	    
         // get cell position and dimesions
         PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
         fCells[iCell]->GetHcub(cellPosi,cellSize);

         // compare them with txvec
         const Double_t xsmall = 1.e-10; 
         if (!( (txvec.at(0)>cellPosi[0]-xsmall) && 
		(txvec.at(0)<=cellPosi[0]+cellSize[0]+xsmall) ) )
            continue; 

         Double_t vol = fCells[iCell]->GetVolume();
         if (vol<1e-10) {
            Log() << kWARNING << "Project: ERROR: Volume too small!" << Endl; 
            continue;
         }

         // get cell value (depending on the option)
         Double_t var = 0.;

         if (strcmp(opt,"nev")==0 || strcmp(opt,"MultiTargetRegression")==0)
            var = GetCellEntries(fCells[iCell]); 
         else if (strcmp(opt,"mean")==0)
            var = GetCellMean(fCells[iCell]); 
         else if (strcmp(opt,"rms")==0) 
            var = GetCellRMS(fCells[iCell]);
         else if (strcmp(opt,"rms_ov_mean")==0) 
            var = GetCellRMSovMean(fCells[iCell]);
         else if (strcmp(opt,"discr")==0)
            var = GetCellDiscr(fCells[iCell]);
         else if (strcmp(opt,"discr_error")==0)
            var = GetCellDiscrError(fCells[iCell]);
	 else if (strcmp(opt,"MonoTargetRegression")==0)
	    var = GetCellRegValue0(fCells[iCell]); 
         else
            Log() << kFATAL << "Project: ERROR: unknown option: " << opt << Endl;

         // filling value to histogram
         h1->SetBinContent(ibinx, var + h1->GetBinContent(ibinx));
      }
   }
   return h1;
}

//_________________________________________________________________________________________
TH2D* TMVA::PDEFoam::Project2( Int_t idim1, Int_t idim2, const char *opt, const char *ker, UInt_t maxbins )
{ 
   // Project foam variable idim1 and variable idim2 to histogram.
   //
   // Parameters:
   // - idim1, idim2 - variables to project to
   // - opt - rms, rms_ov_mean, nev, discr, MonoTargetRegression, MultiTargetRegression
   // - ker - kGaus, kNone (warning: Gaus may be very slow!)
   // - maxbins - maximal number of bins in result histogram.
   //   Set maxbins to 0 if no maximum bin number should be used.
   //
   // Returns:
   // a 2-dimensional histogram

   if (!(// strcmp(opt,"mean")==0  || 
         strcmp(opt,"nev")==0   || 
         strcmp(opt,"rms")==0   || 
         strcmp(opt,"rms_ov_mean")==0   || 
         strcmp(opt,"discr")==0 || 
         //strcmp(opt,"discr_error")==0 || 
         strcmp(opt,"MonoTargetRegression")==0 || 
         strcmp(opt,"MultiTargetRegression")==0) || 
       (idim1>=GetTotDim()) || (idim1<0) || 
       (idim2>=GetTotDim()) || (idim2<0) ||
       (idim1==idim2) )
      return 0;

   EKernel kernel = kNone;
   if (!strcmp(ker, "kNone"))
      kernel = kNone;
   else if (!strcmp(ker, "kGaus"))
      kernel = kGaus;
   else
      Log() << kWARNING << "Warning: wrong kernel! using kNone instead" << Endl;

   char name[]="PDEFoam::Project: ";
   const Bool_t debug = kFALSE;
   char hname[100]; char htit[100];
   const Double_t foam_area = (fXmax[idim1]-fXmin[idim1])*(fXmax[idim2]-fXmin[idim2]);
   Double_t bin_width = 1.; // minimal size of cell
 
   // loop over all cells to determine minimal cell size -> use for bin width
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) { // loop over all active cells
      if (!(fCells[iCell]->GetStat())) continue;   // cell not active -> continue
      // get cell position and dimesions
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      fCells[iCell]->GetHcub(cellPosi,cellSize);
      // loop over all dimensions and determine minimal cell size
      for (Int_t d1=0; d1<GetTotDim(); d1++)
         if (cellSize[d1]<bin_width && cellSize[d1]>0)
            bin_width = cellSize[d1];
   }
   UInt_t nbin = UInt_t(1./bin_width);  // calculate number of bins

   if (maxbins>0 && nbin>maxbins) // limit maximum number of bins
      nbin = maxbins;

   // root can not handle too many bins in one histogram --> catch this
   // Furthermore, to have more than 1000 bins in the histogram doesn't make 
   // sense.
   if (nbin>1000){
      Log() << kWARNING << "Warning: number of bins too big: " << nbin 
              << "! Using 1000 bins for each dimension instead." << Endl;
      nbin = 1000;
   }
   
   // create result histogram
   sprintf(htit,"%s var%d vs var%d",opt,idim1,idim2);
   sprintf(hname,"h%s_%d_vs_%d",opt,idim1,idim2);
   if (debug) {
      Log() << kInfo <<name<<" Project var"<<idim1<<" and var"<<idim2<<" to histo: "<< Endl; 
      Log() << kVERBOSE <<name<<hname<<" nbin="<<nbin
              <<" fXmin["<<idim1<<"]="<<fXmin[idim1]<<" fXmax["<<idim1<<"]="<<fXmax[idim1]
              <<" fXmin["<<idim2<<"]="<<fXmin[idim2]<<" fXmax["<<idim2<<"]="<<fXmax[idim2]
              << Endl;
   }

   TH2D* h1=(TH2D*)gDirectory->Get(hname);
   if (h1) delete h1;
   h1= new TH2D(hname, htit, nbin, fXmin[idim1], fXmax[idim1], nbin, fXmin[idim2], fXmax[idim2]);

   if (!h1) Log() << kFATAL <<name<<"ERROR: Can not create histo"<<hname<< Endl;
   else if (debug) h1->Print();

   if (debug) Log() << kVERBOSE << "histogram created/ loaded" << Endl;

   // init cell counter (debug)
   UInt_t cell_count = 0;

   // start projection algorithm:
   // loop over all cells
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) { // loop over all active cells
      if (!(fCells[iCell]->GetStat())) continue;   // cell not active -> continue
      cell_count++;
    
      // get cell position and dimesions
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      fCells[iCell]->GetHcub(cellPosi,cellSize);
 
      // init cell value (depending on the option)
      // this value will later be filled into the histogram
      Double_t var = 0.;

      // calculate value (depending on the given option 'opt')
      // =========================================================
      if (strcmp(opt,"mean")==0 || strcmp(opt,"nev")==0 || strcmp(opt,"MultiTargetRegression")==0){
         // calculate projected area of cell
         Double_t area = cellSize[idim1] * cellSize[idim2];
         if (area<1e-20){
            Log() << kWARNING << "PDEFoam::ProjectEvents: Warning, cell volume too small --> skiping cell!" 
                    << Endl;
            continue;
         }

         // calc cell entries per projected cell area
         if (strcmp(opt,"mean")==0)
            var = GetCellMean(fCells[iCell])/(area*foam_area); 
         else
            var = GetCellEntries(fCells[iCell])/(area*foam_area); 

         if (debug)
            Log() << kVERBOSE << "Project2: Cell=" << iCell << " projected density="  << var 
                    << " area=" << area*foam_area 
                    << " entries="<< GetCellEntries(fCells[iCell]) << Endl;
      }
      // =========================================================
      else if (strcmp(opt,"rms")==0){
         var = GetCellRMS(fCells[iCell]);
      }
      // =========================================================
      else if (strcmp(opt,"rms_ov_mean")==0){
         var = GetCellRMSovMean(fCells[iCell]);
      }
      // =========================================================
      else if (strcmp(opt,"discr")==0){
         // calculate cell volume in other dimensions (not including idim1 and idim2)
         Double_t area_cell = 1.;	       
         for (Int_t d1=0; d1<GetTotDim(); d1++){
            if ((d1!=idim1) && (d1!=idim2))
               area_cell *= cellSize[d1];
         }
         if (area_cell<1e-20){
            Log() << kWARNING << "ProjectEvents: Warning, cell volume too small --> skiping cell!" << Endl;
            continue;
         }
	 
         // calc discriminator * (cell area times foam area)
         var = GetCellDiscr(fCells[iCell])*area_cell;  // foam is normalized ->  length of foam = 1.0
      }
      // =========================================================
      else if (strcmp(opt,"discr_error")==0){
         var = GetCellDiscrError(fCells[iCell]);  // not testet jet!
      }
      // =========================================================
      else if (strcmp(opt,"MonoTargetRegression")==0){ // plot mean over all underlying cells?
         var = GetCellRegValue0(fCells[iCell]); 
      }
      else
         Log() << kFATAL << "Project: ERROR: unknown option: " << opt << Endl;

      const Double_t xsmall = (1.e-20)*cellSize[idim1];
      const Double_t ysmall = (1.e-20)*cellSize[idim2];

      Double_t x1 = VarTransformInvers( idim1, cellPosi[idim1]+xsmall );
      Double_t y1 = VarTransformInvers( idim2, cellPosi[idim2]+ysmall );
      
      Double_t x2 = VarTransformInvers( idim1, cellPosi[idim1]+cellSize[idim1]-xsmall );
      Double_t y2 = VarTransformInvers( idim2, cellPosi[idim2]+cellSize[idim2]-ysmall );

      Int_t xbin_start = h1->GetXaxis()->FindBin(x1);
      Int_t xbin_stop  = h1->GetXaxis()->FindBin(x2);

      Int_t ybin_start = h1->GetYaxis()->FindBin(y1);
      Int_t ybin_stop  = h1->GetYaxis()->FindBin(y2);

      // loop over all bins
      for (Int_t ibinx=xbin_start; ibinx<xbin_stop; ibinx++) {    //loop over x-bins
         for (Int_t ibiny=ybin_start; ibiny<ybin_stop; ibiny++) { //loop over y-bins

            ////////////////////// weight with kernel ///////////////////////
            if (kernel != kNone){
               Double_t result = 0.;
               Double_t norm   = 0.;

               // calc current position (depending on ibinx, ibiny)
               Double_t x_curr = VarTransform( idim1, ((x2-x1)*ibinx - x2*xbin_start + x1*xbin_stop)/(xbin_stop-xbin_start) );
               Double_t y_curr = VarTransform( idim2, ((y2-y1)*ibiny - y2*ybin_start + y1*ybin_stop)/(ybin_stop-ybin_start) );

               // loop over all cells
               for (Long_t ice=0; ice<=fLastCe; ice++) {
                  if (!(fCells[ice]->GetStat())) continue;

                  // get cell value (depending on option)
                  Double_t cell_var = 0.;
                  if (strcmp(opt,"mean")==0 || strcmp(opt,"nev")==0 || strcmp(opt,"MultiTargetRegression")==0){
                     PDEFoamVect  _cellPosi(GetTotDim()), _cellSize(GetTotDim());
                     fCells[ice]->GetHcub(_cellPosi, _cellSize);
                     Double_t cellarea = _cellSize[idim1] * _cellSize[idim2];
                     cell_var = GetCellEntries(fCells[ice])/(cellarea*foam_area); 
                  }
                  else if (strcmp(opt,"MonoTargetRegression")==0){
                     // PDEFoamVect  _cellPosi(GetTotDim()), _cellSize(GetTotDim());
                     // fCells[ice]->GetHcub(_cellPosi, _cellSize);
                     cell_var = GetCellRegValue0(fCells[ice]); 
                  }
                  else if (strcmp(opt,"rms")==0) 
                     cell_var = GetCellRMS(fCells[ice]);
                  else if (strcmp(opt,"rms_ov_mean")==0) 
                     cell_var = GetCellRMSovMean(fCells[ice]);
                  else if (strcmp(opt,"discr")==0){
                     PDEFoamVect  _cellPosi(GetTotDim()), _cellSize(GetTotDim());
                     fCells[ice]->GetHcub(_cellPosi, _cellSize);
                     Double_t cellarea = 1.;	       
                     for (Int_t d1=0; d1<GetTotDim(); d1++){
                        if ((d1!=idim1) && (d1!=idim2))
                           cellarea *= _cellSize[d1];
                     }	 
                     cell_var = GetCellDiscr(fCells[ice])*cellarea;
                  }
                  else if (strcmp(opt,"discr_error")==0)
                     cell_var = GetCellDiscrError(fCells[ice]);  // not testet jet!
                  else {
                     Log() << kFATAL << "ERROR: unsupported option for kernel plot!" << Endl;
                     return 0;
                  }
		  
                  // fill ndim coordinate of current cell
                  //Double_t coor[GetTotDim()];
                  std::vector<Float_t> coor;
                  for (Int_t i=0; i<GetTotDim(); i++) {
                     if (i == idim1)
                        coor.push_back(x_curr);
                     else if (i == idim2)
                        coor.push_back(y_curr);
                     else
                        coor.push_back(cellPosi[i] + 0.5*cellSize[i]); // approximation
                  }
		  
                  // calc weighted value
                  Double_t weight_ = 0;
                  if (kernel == kGaus)
                     weight_ = WeightGaus(fCells[ice], coor);
                  else 
                     weight_ = 1.;

                  result += weight_ * cell_var;
                  norm   += weight_;
               }
               var = result/norm;
            }
            ////////////////////// END weight with kernel ///////////////////////

            // filling value to histogram
            h1->SetBinContent(ibinx, ibiny, var + h1->GetBinContent(ibinx, ibiny));

            if (debug) {
               Log() << kVERBOSE << " ibinx=" << ibinx << " ibiny=" << ibiny 
                       << " cellID=" << iCell
                       << " [x1,y1]=[" << x1 << ", " << y1 <<"]"
                       << " [x2,y2]=[" << x2 << ", " << y2 <<"] value=" << var << Endl; 
            }
         } // y-loop
      } // x-loop
   } // cell loop

   Log() << kVERBOSE << "Info: Foam has " << cell_count << " active cells" << Endl;

   return h1;
}

//_________________________________________________________________________________________
TH2D* TMVA::PDEFoam::ProjectMC( Int_t idim1, Int_t idim2, Int_t nevents, Int_t nbin )
{  
   // Project variable idim1 and variable idim2 to histogram by using 
   // foams build in MC generation.
   //
   // Parameters:
   // - nevents - number of events to generate
   // - nbin - number of bins of returned histogram
   // - idim1, idim2 - foam variables (dimensions) to project to

   Bool_t debug = kFALSE;

   char hname[100]; char htit[100];
   sprintf(htit,"MC projection var%d vs var%d", idim1, idim2);
   sprintf(hname,"hMCProjections_%d_vs_%d",     idim1, idim2);
   
   TH2D* h1 = (TH2D*)gDirectory->Get(hname);
   if (h1) delete h1;
   h1= new TH2D(hname, htit, nbin, fXmin[idim1], fXmax[idim1], nbin, fXmin[idim2], fXmax[idim2]);

   Double_t *MCvect = new Double_t[GetTotDim()];
   Double_t wt=0.;
   
   // debug output:
   if (debug)
      Log() << kVERBOSE << "frho=" << fRho << " fPseRan=" << fPseRan << Endl;

   // generate MC events according to foam
   for (Long_t loop=0; loop<nevents; loop++){
      MakeEvent();
      GetMCvect(MCvect);
      GetMCwt(wt);
      h1->Fill(MCvect[idim1], MCvect[idim2], wt);
   }

   return h1;
}

//_________________________________________________________________________________________
TVectorD* TMVA::PDEFoam::GetCellElements( std::vector<Float_t> xvec )
{
   // Returns pointer to cell elements.

   if (unsigned(GetTotDim()) != xvec.size()){
      Log() << kFATAL << "ERROR: dimension of foam not equal to cell vector!" << Endl;
      return 0;
   }

   // find cell
   PDEFoamCell *cell= FindCell(VarTransform(xvec));

   if (!cell){ 
      Log() << kFATAL << "ERROR: cell not found!" << Endl;
      return 0; 
   }

   return dynamic_cast<TVectorD*>(cell->GetElement());
}

//_________________________________________________________________________________________
Double_t TMVA::PDEFoam::GetCellElement( PDEFoamCell *cell, UInt_t i )
{
   // Returns cell element i of cell 'cell'.

//    if (!(cell->GetStat()))
//       Log() << kWARNING << "PDEFoam::GetCellElement(): ERROR: Cell is inactive!" << Endl;

   if (i >= GetNElements()){
      Log() << kFATAL << "PDEFoam: ERROR: Index out of range" << Endl;
      return 0.;
   }      
   
   TVectorD *vec = dynamic_cast<TVectorD*>(cell->GetElement());

   if (!vec){
      Log() << kFATAL << "<GetCellElement> ERROR: cell element is not a TVectorD*" << Endl;
      return 0.;
   }

   return (*vec)(i);
}

//_________________________________________________________________________________________
void TMVA::PDEFoam::SetCellElement( PDEFoamCell *cell, UInt_t i, Double_t value )
{
   // Set cell element i of cell to value.

   if (i >= GetNElements()){
      Log() << kFATAL << "ERROR: Index out of range" << Endl;
      return;
   }      
   
   TVectorD *vec = dynamic_cast<TVectorD*>(cell->GetElement());

   if (!vec){
      Log() << kFATAL << "<SetCellElement> ERROR: cell element is not a TVectorD*" << Endl;
      return;
   }

   (*vec)(i) = value;
}

//_________________________________________________________________________________________
void TMVA::PDEFoam::OutputGrow( Bool_t finished )
{
   // Overridden function of PDEFoam to avoid native foam output.
   // Draw TMVA-process bar instead.

   if (finished) {
      Log() << kINFO << "Elapsed time: " + fTimer->GetElapsedTime() +  "                                 " << Endl;
      return;
   }

   Int_t modulo = 1;

   if (fNCells        >= 100) modulo = Int_t(fNCells/100);
   if (fLastCe%modulo == 0)   fTimer->DrawProgressBar( fLastCe );
}

//_________________________________________________________________________________________
void TMVA::PDEFoam::RootPlot2dim( const TString& filename, std::string what, 
                                  Bool_t CreateCanvas, Bool_t colors, Bool_t log_colors )
{
   // Debugging tool which plots 2-dimensional cells as rectangles
   // in C++ format readable for root.
   //
   // Parameters: 
   // - filename - filename of ouput root macro
   // - CreateCanvas - whether to create a new canvas or not

   Bool_t plotmean = kFALSE;
   Bool_t plotnevents = kFALSE;
   Bool_t plotdensity = kFALSE;   // plot event density
   Bool_t plotrms = kFALSE;
   Bool_t plotrmsovmean = kFALSE;
   Bool_t plotcellnumber = kFALSE;
   Bool_t plotdiscr = kFALSE;     // plot discriminator (Tancredi Carli's Method)
   Bool_t plotdiscrerr = kFALSE;  // plot discriminator error (Tancredi Carli's Method)
   Bool_t plotmonotarget = kFALSE; // target value for monotarget regression
   Bool_t fillcells = kTRUE;

   if (what == "mean")
      plotmean = kTRUE;
   else if (what == "nevents")
      plotnevents = kTRUE;
   else if (what == "density")
      plotdensity = kTRUE;
   else if (what == "rms")
      plotrms = kTRUE;
   else if (what == "rms_ov_mean")
      plotrmsovmean = kTRUE;
   else if (what == "cellnumber")
      plotcellnumber = kTRUE;
   else if (what == "discr")
      plotdiscr = kTRUE;
   else if (what == "discrerr")
      plotdiscrerr = kTRUE;
   else if (what == "monotarget")
      plotmonotarget = kTRUE;
   else if (what == "nofill") {
      plotcellnumber = kTRUE;
      fillcells = kFALSE;
   }
   else {
      plotmean = kTRUE;
      Log() << kWARNING << "Unknown option, plotting mean!" << Endl;
   }

   //    std::cerr << "fLastCe=" << fLastCe << std::endl;

   std::ofstream outfile(filename, std::ios::out);
   Double_t x1,y1,x2,y2,x,y;
   Long_t   iCell;
   Double_t offs =0.01;
   //   Double_t offs =0.1;
   Double_t lpag   =1-2*offs;
   outfile<<"{" << std::endl;
   //outfile<<"cMap = new TCanvas(\"Map1\",\" Cell Map \",600,600);"<<std::endl;
   if (!colors) {   // define grayscale colors from light to dark, starting from color index 1000
      outfile << "TColor *graycolors[100];" << std::endl;
      outfile << "for (Int_t i=0.; i<100; i++)" << std::endl;
      outfile << "  graycolors[i]=new TColor(1000+i, 1-(Float_t)i/100.,1-(Float_t)i/100.,1-(Float_t)i/100.);"<< std::endl;
   }
   if (CreateCanvas) 
      outfile<<"cMap = new TCanvas(\""<< fName <<"\",\"Cell Map for "<< fName <<"\",600,600);"<<std::endl;
   //
   outfile<<"TBox*a=new TBox();"<<std::endl;
   outfile<<"a->SetFillStyle(0);"<<std::endl;  // big frame
   outfile<<"a->SetLineWidth(4);"<<std::endl;
   //   outfile<<"a->SetLineColor(2);"<<std::endl;
   //   outfile<<"a->DrawBox("<<offs<<","<<offs<<","<<(offs+lpag)<<","<<(offs+lpag)<<");"<<std::endl;
   //
   outfile<<"TBox *b1=new TBox();"<<std::endl;  // single cell
   if (fillcells) {
      outfile << (colors ? "gStyle->SetPalette(1, 0);" : "gStyle->SetPalette(0);") <<std::endl;
      outfile <<"b1->SetFillStyle(1001);"<<std::endl;
      outfile<<"TBox *b2=new TBox();"<<std::endl;  // single cell
      outfile <<"b2->SetFillStyle(0);"<<std::endl;     
   }
   else {
      outfile <<"b1->SetFillStyle(0);"<<std::endl;
   }
   //

   Int_t lastcell = fLastCe;
   //   if (lastcell > 10000)
   //     lastcell = 10000;

   if (fillcells) 
      (colors ? gStyle->SetPalette(1, 0) : gStyle->SetPalette(0) );
   
   Double_t zmin = 1E8;  // minimal value (for color calculation)
   Double_t zmax = -1E8; // maximal value (for color calculation)

   Double_t value=0.;
   for (iCell=1; iCell<=lastcell; iCell++) {
      if ( fCells[iCell]->GetStat() == 1) {
         if (plotmean)
            value = GetCellMean(fCells[iCell]);
         else if (plotnevents)
            value = GetCellEntries(fCells[iCell]);
         else if (plotdensity)
            value = GetCellDensity(fCells[iCell]);
         else if (plotrms)
            value = GetCellRMS(fCells[iCell]);
         else if (plotrmsovmean)
            value = GetCellRMSovMean(fCells[iCell]);
         else if (plotdiscr) 
            value = GetCellDiscr(fCells[iCell]);
         else if (plotdiscrerr) 
            value = GetCellDiscrError(fCells[iCell]);
         else if (plotmonotarget) 
            value = GetCellRegValue0(fCells[iCell]); 
         else if (plotcellnumber) 
            value = iCell;
         if (value<zmin)
            zmin=value;
         if (value>zmax)
            zmax=value;
      }
   }
   outfile << "// observed minimum and maximum of distribution: " << std::endl;
   outfile << "// Double_t zmin = "<< zmin << ";" << std::endl;
   outfile << "// Double_t zmax = "<< zmax << ";" << std::endl;

   if (log_colors) {
      if (zmin<1)
         zmin=1;
      zmin=TMath::Log(zmin);
      zmax=TMath::Log(zmax);
      outfile << "// logarthmic color scale used " << std::endl;
   } else
      outfile << "// linear color scale used " << std::endl;

   outfile << "// used minimum and maximum of distribution (taking into account log scale if applicable): " << std::endl;
   outfile << "Double_t zmin = "<< zmin << ";" << std::endl;
   outfile << "Double_t zmax = "<< zmax << ";" << std::endl;

   // Next lines from THistPainter.cxx

   //   Int_t ndivz = colors ? abs(gStyle->GetNumberContours()) : 10;
   Int_t ncolors  = colors ? gStyle->GetNumberOfColors() : 100;
   Double_t dz = zmax - zmin;
   //   Double_t scale = ndivz/dz;
   Double_t scale = (ncolors-1)/dz;

   if (GetTotDim()==2) {
      PDEFoamVect  cellPosi(GetTotDim()); PDEFoamVect  cellSize(GetTotDim());
      outfile << "// =========== Rectangular cells  ==========="<< std::endl;
      for (iCell=1; iCell<=lastcell; iCell++) {
         if ( fCells[iCell]->GetStat() == 1) {
            fCells[iCell]->GetHcub(cellPosi,cellSize);
            x1 = offs+lpag*(cellPosi[0]);             y1 = offs+lpag*(cellPosi[1]);
            x2 = offs+lpag*(cellPosi[0]+cellSize[0]); y2 = offs+lpag*(cellPosi[1]+cellSize[1]);

            value = 0;
            if (fillcells) {
               if      (plotmean)       value = GetCellMean(fCells[iCell]);
               else if (plotnevents)    value = GetCellEntries(fCells[iCell]);
               else if (plotdensity)    value = GetCellDensity(fCells[iCell]);
               else if (plotrms)        value = GetCellRMS(fCells[iCell]);
               else if (plotrmsovmean)  value = GetCellRMSovMean(fCells[iCell]);
               else if (plotdiscr)      value = GetCellDiscr(fCells[iCell]);
               else if (plotdiscrerr)   value = GetCellDiscrError(fCells[iCell]);
               else if (plotmonotarget) value = GetCellRegValue0(fCells[iCell]); 
               else if (plotcellnumber) value = iCell;

               if (log_colors) {
                  if (value<1.) value=1;
                  value = TMath::Log(value);
               }

               Int_t color;
               if (colors)
                  color = gStyle->GetColorPalette(Int_t((value-zmin)*scale));
               else 
                  color = 1000+(Int_t((value-zmin)*scale));

               outfile << "b1->SetFillColor(" << color << ");" << std::endl;
            }

            //     cell rectangle
            outfile<<"b1->DrawBox("<<x1<<","<<y1<<","<<x2<<","<<y2<<");"<<std::endl;
            if (fillcells)
               outfile<<"b2->DrawBox("<<x1<<","<<y1<<","<<x2<<","<<y2<<");"<<std::endl;

            //     cell number
            if (lastcell<=250) {
               x = offs+lpag*(cellPosi[0]+0.5*cellSize[0]); y = offs+lpag*(cellPosi[1]+0.5*cellSize[1]);
            }
         }
      }
      outfile<<"// ============== End Rectangles ==========="<< std::endl;
   }//fDim=2
   //
   //
   outfile << "}" << std::endl;
   outfile.flush();
   outfile.close();
}

//________________________________________________________________________________________________
void TMVA::PDEFoam::SetVolumeFraction( Double_t vfr )
{ 
   // set VolFrac to internal foam density TFDISTR
   fDistr->SetVolumeFraction(vfr);
   SetPDEFoamVolumeFraction(vfr);
}

//________________________________________________________________________________________________
void TMVA::PDEFoam::FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights )
{
   // Insert event to internal foam density TFDISTR.
   fDistr->FillBinarySearchTree(ev, ft, NoNegWeights);
}

//________________________________________________________________________________________________
void TMVA::PDEFoam::Create(Bool_t CreateCellElements)
{ 
   // Function that builds up the foam.
   SetRho(fDistr);      // give filled distribution (binary search tree) to Foam
   Initialize(CreateCellElements); // create foam cells from fDistr
}; 

//________________________________________________________________________________________________
void TMVA::PDEFoam::Init()
{ 
   // Initialize internal foam density TFDISTR
   fDistr->Initialize(GetTotDim());
}

//________________________________________________________________________________________________
void TMVA::PDEFoam::SetFoamType( EFoamType ft )
{
   // Set the foam type.  This determinates the method of the
   // calculation of the density during the foam build-up.
   if (ft==kDiscr)
      fDistr->SetDensityCalc(kDISCRIMINATOR);
   else if (ft==kMonoTarget)
      fDistr->SetDensityCalc(kTARGET);
   else
      fDistr->SetDensityCalc(kEVENT_DENSITY);
}

//________________________________________________________________________________________________
void TMVA::PDEFoam::PrintDensity(){
   // Information output
   fDistr->PrintDensity();
}
 
//________________________________________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::PDEFoam& pdefoam )
{ 
   // Write PDEFoam variables to stream 'os'.
   pdefoam.PrintStream(os);
   return os; // Return the output stream.
}

//________________________________________________________________________________________________
istream& TMVA::operator>> ( istream& istr, TMVA::PDEFoam& pdefoam )
{ 
   // Read PDEFoam variables from stream 'istr'.
   pdefoam.ReadStream(istr);
   return istr;
}

//_______________________________________________________________________
void TMVA::PDEFoam::ReadStream( istream & istr )
{
   // Read PDEFoam variables from stream 'istr'.

   // inherited class variables: fLastCe, fNCells, fDim[GetTotDim()]
   istr >> fLastCe;
   istr >> fNCells;
   istr >> fDim;


   Double_t vfr = -1.;
   istr >> vfr;
   SetPDEFoamVolumeFraction(vfr);

   Log() << kVERBOSE << "Foam dimension: " << GetTotDim() << Endl;

   // read Class Variables: fXmin, fXmax
   if (fXmin) delete [] fXmin;
   if (fXmax) delete [] fXmax;
   fXmin = new Double_t[GetTotDim()];
   fXmax = new Double_t[GetTotDim()];
   for (Int_t i=0; i<GetTotDim(); i++) 
      istr >> fXmin[i];
   for (Int_t i=0; i<GetTotDim(); i++) 
      istr >> fXmax[i];
}

//_______________________________________________________________________
void TMVA::PDEFoam::PrintStream( ostream & ostr ) const
{
   // Write PDEFoam variables to stream 'os'.

   // inherited class variables: fLastCe, fNCells, fDim[GetTotDim()]
   ostr << fLastCe << std::endl;
   ostr << fNCells << std::endl;
   ostr << fDim    << std::endl;
   ostr << GetPDEFoamVolumeFraction() << std::endl;

   // write class variables: fXmin, fXmax
   for (Int_t i=0; i<GetTotDim(); i++) 
      ostr << fXmin[i] << std::endl;
   for (Int_t i=0; i<GetTotDim(); i++) 
      ostr << fXmax[i] << std::endl;
}

//_______________________________________________________________________
void TMVA::PDEFoam::AddXMLTo( void* parent ){
   // write foam variables to xml

   void *variables = gTools().xmlengine().NewChild( parent, 0, "Variables" );
   gTools().AddAttr( variables, "LastCe",           fLastCe );
   gTools().AddAttr( variables, "nCells",           fNCells );
   gTools().AddAttr( variables, "Dim",              fDim );
   gTools().AddAttr( variables, "VolumeFraction",   GetPDEFoamVolumeFraction() );

   void *xmin_wrap;
   for (Int_t i=0; i<GetTotDim(); i++){
      xmin_wrap = gTools().xmlengine().NewChild( variables, 0, "Xmin" );
      gTools().AddAttr( xmin_wrap, "Index", i );
      gTools().AddAttr( xmin_wrap, "Value", fXmin[i] );
   }

   void *xmax_wrap;
   for (Int_t i=0; i<GetTotDim(); i++){
      xmax_wrap = gTools().xmlengine().NewChild( variables, 0, "Xmax" );
      gTools().AddAttr( xmax_wrap, "Index", i );
      gTools().AddAttr( xmax_wrap, "Value", fXmax[i] );
   }
}

//_______________________________________________________________________
void TMVA::PDEFoam::ReadXML( void* parent ){
   void *variables = gTools().xmlengine().GetChild( parent );
   gTools().ReadAttr( variables, "LastCe",         fLastCe );
   gTools().ReadAttr( variables, "nCells",         fNCells );
   gTools().ReadAttr( variables, "Dim",            fDim );
   Float_t volfr;
   gTools().ReadAttr( variables, "VolumeFraction", volfr );
   SetPDEFoamVolumeFraction( volfr );

   if (fXmin) delete [] fXmin;
   if (fXmax) delete [] fXmax;
   fXmin = new Double_t[GetTotDim()];
   fXmax = new Double_t[GetTotDim()];

   void *xmin_wrap = gTools().xmlengine().GetChild( variables );
   for (Int_t counter=0; counter<fDim; counter++) {
      Int_t i=0;
      gTools().ReadAttr( xmin_wrap , "Index", i );
      if (i >= GetTotDim() || i<0)
	 Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmin_wrap , "Value", fXmin[i] );
      xmin_wrap = gTools().xmlengine().GetNext( xmin_wrap );
   }

   void *xmax_wrap = xmin_wrap; //gTools().xmlengine().GetChild( variables );
   for (Int_t counter=0; counter<fDim; counter++) {
      Int_t i=0;
      gTools().ReadAttr( xmax_wrap , "Index", i );
      if (i >= GetTotDim() || i<0)
	 Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmax_wrap , "Value", fXmax[i] );
      xmax_wrap = gTools().xmlengine().GetNext( xmax_wrap );
   }
}

////////////////////////////////////////////////////////////////////////////////
//       End of Class PDEFoam                                                   //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//       Begin of Class TFDISTR                                               //
////////////////////////////////////////////////////////////////////////////////

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DTFISTR                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Helper class used by the PDEFoam object to evaluate the event density       *
 *      at a given point in the Foam.  The main Function is TFDISTR::Density().   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli                                                            *
 *      Dominik Dannheim                                                          *
 *      Alexander Voigt                                                           *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of TFDISTR
//
// The TFDSITR class provides an interface between the Binary search tree
// and the PDEFoam object.  In order to build-up the foam one needs to 
// calculate the density of events at a given point (sampling during
// Foam build-up).  The function TFDISTR::Density() does this job.  It
// Uses a binary search tree, filled with training events, in order to
// provide this density.
//_______________________________________________________________________




//________________________________________________________________________________________________
TMVA::TFDISTR::TFDISTR() :
   fDim(-1),
   fXmin(0),
   fXmax(0),
   fVolFrac(-1.),
   fBst(NULL),
   fDensityCalc(kEVENT_DENSITY), // default: fill event density to BinarySearchTree
   fSignalClass(1),
   fBackgroundClass(0),
   fLogger( new MsgLogger("TFDISTR"))
{}

//________________________________________________________________________________________________
TMVA::TFDISTR::~TFDISTR() {
   if (fBst)  delete fBst;
   if (fXmin) delete [] fXmin;  fXmin=0;
   if (fXmax) delete [] fXmax;  fXmax=0;
   delete fLogger;
}


//________________________________________________________________________________________________
void TMVA::TFDISTR::PrintDensity()
{
   // Information output about foam density

   char name[]={"EventfromDensity::PrintDensity"};

   Log() << kINFO << name<<" Volume fraction to search for events: "<<fVolFrac<< Endl;
   Log() << kINFO << name<<" Binary Tree: "<<fBst<< Endl;

   if (!fBst) Log() << kINFO <<name<<" Binary tree not found ! "<< Endl;

   Log() << kINFO <<name<<" Volfraction= "<<fVolFrac<< Endl;

   for (Int_t idim=0; idim<fDim; idim++){
      Log() << kINFO <<name<<idim<<" fXmin["<<idim<<"]= "<<fXmin[idim]
              <<" fXmax["<<idim<<"]= "<<fXmax[idim]<< Endl;
   }
}

//________________________________________________________________________________________________
void TMVA::TFDISTR::Initialize( Int_t ndim )
{
   // Initialisation procedure of internal foam density.
   // Set dimension to 'ndim' and create BinarySearchTree.

   std::string name = "EventfromDensity::PreInitialize:";
   this->SetDim(ndim);

   if (fBst) delete fBst;
   fBst = new TMVA::BinarySearchTree();
  
   if (!fBst){
      Log() << kFATAL << name << "ERROR: an not create binary tree !" << Endl;
      return;
   }
  
   fBst->SetPeriode(fDim);
}

//________________________________________________________________________________________________
void TMVA::TFDISTR::FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights )
{
   // This method creates an TMVA::Event and inserts it into the
   // binary search tree.
   //
   // If 'NoNegWeights' is true, an event with negative weight will
   // not be filled into the foam.  (Default value: false)

   if(NoNegWeights && ev->GetWeight()<=0)
      return;

   TMVA::Event *event = new TMVA::Event(*ev);
   event->SetSignalClass( fSignalClass );

   // set event class and normalization
   if (ft==kSeparate || ft==kDiscr){
      event->SetClass(ev->IsSignal() ? fSignalClass : fBackgroundClass);
   } else if (ft==kMultiTarget){
      // since in multi target regression targets are handled like
      // variables, remove targets and add them to the event variabels
      std::vector<Float_t> targets = ev->GetTargets();
      for (UInt_t i = 0; i < targets.size(); i++)
	 event->SetVal(i+ev->GetValues().size(), targets.at(i));
      event->GetTargets().clear();
      event->SetClass(fSignalClass);
   }
   fBst->Insert(event);
}

//________________________________________________________________________________________________
void TMVA::TFDISTR::FillEdgeHist( Int_t nDim, TObjArray *myfHistEdg, 
                                  Double_t *cellPosi, Double_t *cellSize, Double_t *ceSum, Double_t ceVol ) 
{
   // New method to fill edge histograms directly, without MC sampling.
   // DD 10.Jan.2008

   Double_t wt;
   std::vector<Double_t> lb(nDim+1);
   std::vector<Double_t> ub(nDim+1);
   Double_t *valuesarr = new Double_t[Int_t(nDim)];
  
   for (Int_t idim=0; idim<nDim; idim++){
      lb[idim]=fXmin[idim]+(fXmax[idim]-fXmin[idim])*cellPosi[idim];
      ub[idim]=fXmin[idim]+(fXmax[idim]-fXmin[idim])*(cellPosi[idim]+cellSize[idim]);
   }
   lb[nDim] = -1.e19;
   ub[nDim] = +1.e19;

   TMVA::Volume volume(&lb, &ub);
   std::vector<const TMVA::BinarySearchTreeNode*> nodes;
   if (!fBst) {
      Log() << kFATAL <<"Binary tree not found ! "<< Endl; 
      exit (1);
   }
   fBst->SearchVolume(&volume, &nodes);
   for (std::vector<const TMVA::BinarySearchTreeNode*>::iterator inode = nodes.begin(); inode != nodes.end(); inode++) {
      const std::vector<Float_t>& values = (*inode)->GetEventV();
      for (Int_t idim=0; idim<nDim; idim++) {
         valuesarr[idim]=(values[idim]-fXmin[idim])/(fXmax[idim]-fXmin[idim]);
      }
      Double_t event_density = 0;
      wt = (TFDISTR::Density(nDim, valuesarr, event_density)) * ceVol;
      for (Int_t idim=0; idim<nDim; idim++) 
         ((TH1D *)(*myfHistEdg)[idim])->Fill((values[idim]-fXmin[idim])/(fXmax[idim]-fXmin[idim]), wt);
      ceSum[0] += wt;    // sum of weights
      ceSum[1] += wt*wt; // sum of weights squared
      ceSum[2]++;        // sum of 1
      if (ceSum[3]>wt) ceSum[3]=wt;  // minimum weight;
      if (ceSum[4]<wt) ceSum[4]=wt;  // maximum weight
   }

   delete[] valuesarr;
}

//________________________________________________________________________________________________
UInt_t TMVA::TFDISTR::GetNEvents( PDEFoamCell* cell )
{
   // Return number of events (from binary search tree) within a certain volume given 
   // by the cell coordinates and size.

   PDEFoamVect  cellSize(fDim);
   PDEFoamVect  cellPosi(fDim);

   cell->GetHcub(cellPosi,cellSize);

   std::vector<Float_t> lb;
   std::vector<Float_t> ub;

   for (Int_t idim=0; idim<fDim; idim++) {
      lb.push_back(VarTransformInvers(idim, cellPosi[idim]));
      ub.push_back(VarTransformInvers(idim, cellPosi[idim] + cellSize[idim]));
   }

   TMVA::Volume volume(&lb, &ub);
   std::vector<const TMVA::BinarySearchTreeNode*> nodes;

   if (!fBst) {
      Log() << kFATAL << "<TFDISTR::GetNEvents> Binary tree not found!" << Endl; 
      exit (1);
   }

   fBst->SearchVolume(&volume, &nodes);
   return nodes.size();
}

//________________________________________________________________________________________________
Bool_t TMVA::TFDISTR::CellRMSCut( PDEFoamCell* cell, Float_t rms_cut, Int_t nbin )
{
   // Checks whether the RMS within the cell is greater than 'rms_cut'.
   // Furthermore control whether RMSError < RMS.
   //
   // 1) fill histograms for each dimension
   // 2) check whether RMS/mean > RMS_error/mean in each dimension
   // 3) if 2) is kTRUE -> return kTRUE only if RMS/mean > rms_cut in one dimension
   //    --> cell may be splitten

   // get cell coordinates
   PDEFoamVect  cellSize(fDim);
   PDEFoamVect  cellPosi(fDim);

   cell->GetHcub(cellPosi, cellSize);

   std::vector<Float_t> lb;
   std::vector<Float_t> ub;

   for (Int_t idim=0; idim<fDim; idim++) {
      lb.push_back(VarTransformInvers(idim, cellPosi[idim]));
      ub.push_back(VarTransformInvers(idim, cellPosi[idim] + cellSize[idim]));
   }

   TMVA::Volume volume(&lb, &ub);
   std::vector<const TMVA::BinarySearchTreeNode*> nodes;

   if (!fBst) {
      Log() << kFATAL << "<TFDISTR::CellRMSCut> Binary tree not found!" << Endl; 
      return 0.;
   }

   fBst->SearchVolume(&volume, &nodes);

   // if no events found -> do not split cell
   if (nodes.size() <= 1)
      return kFALSE;

   // create histogram for each dimension
   std::vector<TH1F*> histo;
   for (Int_t i=0; i<fDim; i++){
      std::stringstream title (std::stringstream::in | std::stringstream::out);
      title << "proj" << i;
      histo.push_back(new TH1F(title.str().c_str(), title.str().c_str(), nbin, lb.at(i), ub.at(i)));
   }

   // 1) fill histos with all events found
   for (UInt_t i=0; i<nodes.size(); i++){
      for (Int_t d=0; d<fDim; d++)
         (histo.at(d))->Fill((nodes.at(i)->GetEventV()).at(d), nodes.at(i)->GetWeight());
   }

   // calc RMS/mean and RMSError/mean
   std::vector<Float_t> rms_mean, rms_err_mean;
   for (Int_t i=0; i<fDim; i++){ // loop over all histograms

      // loop over all bins in histo
      std::vector<Float_t> bin_val;
      for (Int_t k=1; k<=nbin; k++){
         bin_val.push_back( histo.at(i)->GetBinContent(k) );
      }

      // calc mean
      Float_t mean = 0;
      for (UInt_t k=0; k<bin_val.size(); k++)
         mean += bin_val.at(k);
      mean /= bin_val.size();

      // calc rms
      Float_t rms  = 0;
      for (UInt_t k=0; k<bin_val.size(); k++)
         rms += TMath::Power( bin_val.at(k) - mean, 2 );	 
      rms /= (bin_val.size()-1);  // variance
      rms = TMath::Sqrt(rms)/mean; // significan value

      // calc error on rms
      Float_t rms_err = 0;
      rms_err = (1+rms*TMath::Sqrt(2))*rms/TMath::Sqrt(2*bin_val.size());

      // save values
      if (TMath::Abs(mean) > 1.0e-15){
         rms_mean.push_back( rms );
         rms_err_mean.push_back( rms_err );
      }
      else {
         rms_mean.push_back(0.);     // set values to avoid cut-check
         rms_err_mean.push_back(1.);
      }
   }

   // 2) and 3) apply cut on 'rms_cur'
   Bool_t result = kFALSE;
   for (Int_t i=0; i<fDim; i++){
      //if (rms_mean.at(i) > rms_err_mean.at(i))  // if enough statistics
      result = result || (rms_mean.at(i) > rms_cut);
   }

   // delete histograms
   for (Int_t i=0; i<fDim; i++)
      delete histo.at(i);

   return result;
}

//________________________________________________________________________________________________
Double_t TMVA::TFDISTR::Density( Int_t nDim, Double_t *Xarg, Double_t &event_density )
{
   // This function is needed during the foam buildup.
   // It return a certain density depending on the selected classification
   // or regression options:
   //
   // In case of separated foams (classification) or multi target regression:
   //  - returns event density within volume (specified by VolFrac)
   // In case of unified foams: (classification)
   //  - returns discriminator (N_sig)/(N_sig + N_bg) divided by volume (specified by VolFrac)
   // In case of mono target regression:
   //  - returns average target value within volume divided by volume (specified by VolFrac)

   assert(nDim == fDim);

   Bool_t simplesearch=true; // constant volume for all searches
   Bool_t usegauss=true;     // kFALSE: use Teepee function, kTRUE: use Gauss with sigma=0.3
   Int_t targetevents=0;     // if >0: density is defined as inverse of volume that includes targetevents
   //                           (this option should probably not be used together simplesearch=false) 
   char name[]={" DensityFromEvents: "};
   Bool_t debug0 = false;
   Bool_t debug  = false;

   Double_t volscale=1.;
   Double_t lasthighvolscale=0.;
   Double_t lastlowvolscale=0.;
   Int_t nit=0;

   // probevolume relative to hypercube with edge length 1:
   Double_t probevolume_inv;

   static Int_t nev=0;  nev++;

   if (debug) for (Int_t idim=0; idim<nDim; idim++) 
                 Log() << kVERBOSE << nev << name << idim <<" unscaled in: Xarg= "<< Xarg[idim] << Endl;   

   //make a variable transform, since Foam only knows about x=[0,1]
   Double_t wbin;
   for (Int_t idim=0; idim<nDim; idim++){
      if (debug)
         std::cerr << "fXmin[" << idim << "] = " << fXmin[idim]
                   << ", fXmax[" << idim << "] = " << fXmax[idim] << std::endl;
      wbin=fXmax[idim]-fXmin[idim];
      Xarg[idim]=fXmin[idim]+wbin*Xarg[idim];
   }

   if (debug0) for (Int_t idim=0; idim<nDim; idim++) 
                  Log() << kVERBOSE <<nev<<name<<idim<<" scaled in: Xarg= "<<Xarg[idim]<< Endl;

   //create volume around point to be found
   std::vector<Double_t> lb(nDim);
   std::vector<Double_t> ub(nDim);

 volumesearchstart:
   probevolume_inv = pow((fVolFrac/2/volscale), nDim); 

   for (Int_t idim = 0; idim < fDim; idim++) {
      lb[idim] = Xarg[idim];
      ub[idim] = Xarg[idim];
   }

   for (Int_t idim = 0; idim < nDim; idim++) {
      Double_t volsize=(fXmax[idim] - fXmin[idim]) / fVolFrac * volscale;    
      lb[idim] -= volsize;
      ub[idim] += volsize;
      if (debug) {
         std::cerr << "lb[" << idim << "]=" << lb[idim] << std::endl;
         std::cerr << "ub[" << idim << "]=" << ub[idim] << std::endl;
      }
   }

   TMVA::Volume volume(&lb, &ub);
      
   std::vector<const TMVA::BinarySearchTreeNode*> nodes;

   if (!fBst) {
      Log() << kFATAL <<name<<" Binary tree not found ! "<< Endl; 
   }

   // do range searching
   fBst->SearchVolume(&volume, &nodes);

   // normalized density: (number of counted events) / volume / (total number of events)
   // should be ~1 on average
   Double_t count=(Double_t) (nodes.size()); // number of events found

   event_density = count * probevolume_inv;  // store density based on total number of events   

   // add a small number, to stabilize cell-division algorithm

   if (targetevents>0 && nit<25) {
      nit++;
      if (count<targetevents) {   // repeat search with increased volume if not enough events found
         lastlowvolscale=volscale;
         if (lasthighvolscale>0.)
            volscale+=(lasthighvolscale-volscale)/2;
         else
            volscale*=1.5;
         goto volumesearchstart;
      }
      //      else if (count>targetevents) { // repeat search with decreased volume if too many events found
      //         lasthighvolscale=volscale;
      //         if (lastlowvolscale>0.)
      //            volscale-=(volscale-lastlowvolscale)/2;
      //         else
      //            volscale*=0.5;
      //         goto volumesearchstart;
      //      }
   }

   if (targetevents>0 && count<targetevents)
      Log() << kWARNING << "WARNING: Number of target events not reached within 25 iterations. count==" 
              << count << Endl;

   Double_t N_sig = 0;
   Double_t N_tar = 0;
   Double_t weighted_count = 0.; // number of events found (sum of weights!)
   for (UInt_t j=0; j<nodes.size(); j++)
      weighted_count += (nodes.at(j))->GetWeight();

   if (FillDiscriminator()){ // calc number of signal events in nodes
      N_sig = 0;
      // sum over all nodes->IsSignal;
      for (Int_t j=0; j<count; j++){
         N_sig += ((nodes.at(j))->IsSignal()?1.:0.) * (nodes.at(j))->GetWeight();
      }
   }
   else if (FillTarget0()){
      N_tar = 0;
      // sum over all nodes->GetTarget(0);
      for (Int_t j=0; j<count; j++) {
         if (((nodes.at(j))->GetTargets()).size() < 1) 
            Log() << kFATAL << "ERROR: No targets for node " << j << Endl;
         N_tar += ((nodes.at(j))->GetTargets()).at(0) * ((nodes.at(j))->GetWeight());
      }
   }

   if (simplesearch){
      if (FillDiscriminator())
         return (N_sig/(weighted_count+0.1))*probevolume_inv; // fill:  (N_sig/N_total) / (cell_volume)
      else if (FillTarget0())
         return (N_tar/(weighted_count+0.1))*probevolume_inv; // fill:  (N_tar/N_total) / (cell_volume)
      else
         return ((weighted_count+0.1)*probevolume_inv); // fill:  N_total(weighted) / cell_volume
   }
  
   Double_t density = 0.;
   Double_t normalised_distance = 0;

   if (count==0)
      return (0.1*probevolume_inv);

   unsigned usednodes;

   for (usednodes = 0; usednodes < nodes.size(); usednodes++) { // loop over all nodes
      const std::vector<Float_t>& values = nodes[usednodes]->GetEventV();
      for (Int_t idim = 0; idim < nDim; idim++) 
         normalised_distance += pow( ( values[idim]-Xarg[idim] )/ (ub[idim]-lb[idim]) * 2 , 2);
      normalised_distance = sqrt(normalised_distance/nDim);
      if (usegauss)
         density+=TMath::Gaus(normalised_distance, 0, 0.3, kFALSE);    // Gaus Kernel with sigma=0.3
      else  
         density+=(1-normalised_distance);  // Teepee kernel
   }
   density /= usednodes;

   if (FillDiscriminator())
      density = (N_sig/density)*probevolume_inv; // fill:  (N_sig/N_total) / (cell_volume)
   else if (FillTarget0())
      density = (N_tar/density)*probevolume_inv; // fill:  (N_tar/N_total) / (cell_volume)
   else 
      density *= probevolume_inv; // fill:  N_total / cell_volume

   return density;
}

//________________________________________________________________________________________________
TH2D* TMVA::TFDISTR::MakeHistogram( Int_t nbinsx, Int_t nbinsy ) 
{
   // Unused function.
   // Warning: only works in 2D for now.

   //  const Bool_t debug = kTRUE;
   const Bool_t debug = kFALSE;

   TH2D *foamhist = new TH2D("foamhist", "Distributon",
                             nbinsx, fXmin[0], fXmax[0],
                             nbinsy, fXmin[1], fXmax[1]);
   if (debug)
      std::cerr << "TFDIST::MakeHistogram, nbinsx=" << nbinsx
                << ", nbinsy=" << nbinsy
                << ", fXmin[0]=" << fXmin[0]
                << ", fXmax[0]=" << fXmax[0]
                << ", fXmin[1]=" << fXmin[1]
                << ", fXmax[1]=" << fXmax[1]
                << std::endl;

   Double_t xvec[2];

   for (Int_t i = 1; i < nbinsx+1; i++) { // ROOT's weird bin numbering
      for (Int_t j = 1; j < nbinsy+1; j++) {
         xvec[0] = foamhist->GetXaxis()->GetBinCenter(i);
         xvec[1] = foamhist->GetYaxis()->GetBinCenter(j);

         // Transform to foam-internal coordinate space, that this->Density expects
         xvec[0] = (xvec[0] - fXmin[0]) / (fXmax[0] - fXmin[0]);
         xvec[1] = (xvec[1] - fXmin[1]) / (fXmax[1] - fXmin[1]);

         // Note: this->Density changes xvec, need to print them before!
         if (debug)
            std::cerr << "xvec[0]=" << xvec[0]
                      << ", xvec[1]=" << xvec[1]
                      << std::endl;

         Int_t ibin = foamhist->GetBin(i, j);
         Double_t var = this->Density(2, xvec);

         foamhist->SetBinContent(ibin, var);

         if (debug)
            std::cerr << "i=" << i
                      << ", j=" << j
                      << ", var=" << var
                      << ", ibin=" << ibin
                      << std::endl;
      }
   }

   return foamhist;
}

ClassImp(TMVA::TFDISTR)

////////////////////////////////////////////////////////////////////////////////
//       End of Class TFDISTR                                                 //
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//                                                          //
// Class PDEFoamIntegrand is an Abstract class representing   //
// n-dimensional real positive integrand function           //
//                                                          //
//////////////////////////////////////////////////////////////

ClassImp(TMVA::PDEFoamIntegrand);

TMVA::PDEFoamIntegrand::PDEFoamIntegrand(){};


////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
// Class PDEFoamCell  used in PDEFoam                                             //
// ==================================                                             //
// Objects of this class are hyper-rectangular cells organized in the binary tree.//
// Special algorithm for encoding relative positioning of the cells               //
// allow to save total memory allocation needed for the system of cells.          //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////

ClassImp(TMVA::PDEFoamCell)

//________________________________________________________________________________
TMVA::PDEFoamCell::PDEFoamCell()
{
   // Default constructor for streamer

   fParent  = 0;
   fDaught0 = 0;
   fDaught1 = 0;
   fElement = 0;
}

//_________________________________________________________________________________
TMVA::PDEFoamCell::PDEFoamCell(Int_t kDim)
{
   // User constructor allocating single empty Cell
   if (  kDim >0) {
      //---------=========----------
      fDim     = kDim;
      fStatus   = 1;
      fParent   = 0;
      fDaught0  = 0;
      fDaught1  = 0;
      fXdiv     = 0.0;
      fBest     = 0;
      fVolume   = 0.0;
      fIntegral = 0.0;
      fDrive    = 0.0;
      fPrimary  = 0.0;
      fElement  = 0;
   } else
      Error( "PDEFoamCell", "Dimension has to be >0" );
}

//_________________________________________________________________________________
TMVA::PDEFoamCell::PDEFoamCell(PDEFoamCell &From): TObject(From)
{
   // Copy constructor (not tested!)

   Error( "PDEFoamCell", "+++++ NEVER USE Copy constructor for PDEFoamCell");
   fStatus      = From.fStatus;
   fParent      = From.fParent;
   fDaught0     = From.fDaught0;
   fDaught1     = From.fDaught1;
   fXdiv        = From.fXdiv;
   fBest        = From.fBest;
   fVolume      = From.fVolume;
   fIntegral    = From.fIntegral;
   fDrive       = From.fDrive;
   fPrimary     = From.fPrimary;
   fElement     = From.fElement;
}

//___________________________________________________________________________________
TMVA::PDEFoamCell::~PDEFoamCell()
{
   // Destructor
}

//___________________________________________________________________________________
TMVA::PDEFoamCell& TMVA::PDEFoamCell::operator=(const PDEFoamCell &From)
{
   // Substitution operator = (never used)

   Info("PDEFoamCell", "operator=\n ");
   if (&From == this) return *this;
   fStatus      = From.fStatus;
   fParent      = From.fParent;
   fDaught0     = From.fDaught0;
   fDaught1     = From.fDaught1;
   fXdiv        = From.fXdiv;
   fBest        = From.fBest;
   fVolume      = From.fVolume;
   fIntegral    = From.fIntegral;
   fDrive       = From.fDrive;
   fPrimary     = From.fPrimary;
   fElement     = From.fElement;
   return *this;
}


//___________________________________________________________________________________
void TMVA::PDEFoamCell::Fill(Int_t Status, PDEFoamCell *Parent, PDEFoamCell *Daugh1, PDEFoamCell *Daugh2)
{
   // Fills in certain data into newly allocated cell

   fStatus  = Status;
   fParent  = Parent;
   //   Log() << "D1" << Daugh1 << Endl;
   fDaught0 = Daugh1;
   fDaught1 = Daugh2;
}

////////////////////////////////////////////////////////////////////////////////
//              GETTERS/SETTERS
////////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________________
void    TMVA::PDEFoamCell::GetHcub( PDEFoamVect &cellPosi, PDEFoamVect &cellSize)  const
{
   // Provides size and position of the cell
   // These parameter are calculated by analyzing information in all parents
   // cells up to the root cell. It takes time but saves memory.
   if(fDim<1) return;
   const PDEFoamCell *pCell,*dCell;
   cellPosi = 0.0; cellSize=1.0; // load all components
   dCell = this;
   while(dCell != 0) {
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
      if(dCell == pCell->GetDau0()  ) {
         cellSize[kDiv] *=xDivi;
         cellPosi[kDiv] *=xDivi;
      } else if(   dCell == pCell->GetDau1()  ) {
         cellSize[kDiv] *=(1.0-xDivi);
         cellPosi[kDiv]  =cellPosi[kDiv]*(1.0-xDivi)+xDivi;
      } else {
         Error( "GetHcub ","Something wrong with linked tree \n");
      }
      dCell=pCell;
   }//while
}//GetHcub

//______________________________________________________________________________________
void    TMVA::PDEFoamCell::GetHSize( PDEFoamVect &cellSize)  const
{
   // Provides size of the cell
   // Size parameters are calculated by analyzing information in all parents
   // cells up to the root cell. It takes time but saves memory.
   if(fDim<1) return;
   const PDEFoamCell *pCell,*dCell;
   cellSize=1.0; // load all components
   dCell = this;
   while(dCell != 0) {
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
      if(dCell == pCell->GetDau0() ) {
         cellSize[kDiv]=cellSize[kDiv]*xDivi;
      } else if(dCell == pCell->GetDau1()  ) {
         cellSize[kDiv]=cellSize[kDiv]*(1.0-xDivi);
      } else {
         Error( "GetHSize ","Something wrong with linked tree \n");
      }
      dCell=pCell;
   }//while
}//GetHSize

//_________________________________________________________________________________________
void TMVA::PDEFoamCell::CalcVolume(void)
{
   // Calculates volume of the cell using size params which are calculated

   Int_t k;
   Double_t volu=1.0;
   if(fDim>0) {         // h-cubical subspace
      PDEFoamVect cellSize(fDim);
      GetHSize(cellSize);
      for(k=0; k<fDim; k++) volu *= cellSize[k];
   }
   fVolume =volu;
}

//__________________________________________________________________________________________
void TMVA::PDEFoamCell::Print(Option_t *option) const
{
   // Printout of the cell geometry parameters for the debug purpose

   if (!option) Error( "Print", "No option set\n");

   cout <<  " Status= "<<     fStatus   <<",";
   cout <<  " Volume= "<<     fVolume   <<",";
   cout <<  " TrueInteg= " << fIntegral <<",";
   cout <<  " DriveInteg= "<< fDrive    <<",";
   cout <<  " PrimInteg= " << fPrimary  <<",";
   cout << endl;
   cout <<  " Xdiv= "<<fXdiv<<",";
   cout <<  " Best= "<<fBest<<",";
   cout <<  " Parent=  {"<< (GetPare() ? GetPare()->GetSerial() : -1) <<"} "; // extra DEBUG
   cout <<  " Daught0= {"<< (GetDau0() ? GetDau0()->GetSerial() : -1 )<<"} "; // extra DEBUG
   cout <<  " Daught1= {"<< (GetDau1() ? GetDau1()->GetSerial()  : -1 )<<"} "; // extra DEBUG
   cout << endl;
   //
   //
   if (fDim>0 ) {
      PDEFoamVect cellPosi(fDim); PDEFoamVect cellSize(fDim);
      GetHcub(cellPosi,cellSize);
      cout <<"   Posi= "; cellPosi.Print("1"); cout<<","<< endl;
      cout <<"   Size= "; cellSize.Print("1"); cout<<","<< endl;
   }
}
///////////////////////////////////////////////////////////////////
//        End of  class  PDEFoamCell                             //
///////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////
//                                                                               //
// Class  PDEFoamMaxwt                                                           //
// ===================                                                           //
// Small auxiliary class for controlling MC weight.                              //
// It provides certain measure of the "maximum weight"                           //
// depending on small user-parameter "epsilon".                                  //
// It creates and uses 2 histograms of the TH1D class.                           //
// User defines no. of bins nBin,  nBin=1000 is  recommended                     //
// wmax defines weight range (1,wmax), it is adjusted "manually"                 //
//                                                                               //
///////////////////////////////////////////////////////////////////////////////////

ClassImp(TMVA::PDEFoamMaxwt);

//____________________________________________________________________________
TMVA::PDEFoamMaxwt::PDEFoamMaxwt():
   fLogger( new MsgLogger("PDEFoamMaxwt") )
{
   // Constructor for streamer
   fNent = 0;
   fnBin = 0;
   fWtHst1 = 0;
   fWtHst2 = 0;
}

//____________________________________________________________________________
TMVA::PDEFoamMaxwt::PDEFoamMaxwt(Double_t wmax, Int_t nBin):
   fLogger( new MsgLogger("PDEFoamMaxwt") )
{
   // Principal user constructor
   fNent = 0;
   fnBin = nBin;
   fwmax = wmax;
   fWtHst1 = new TH1D("PDEFoamMaxwt_hst_Wt1","Histo of weight   ",nBin,0.0,wmax);
   fWtHst2 = new TH1D("PDEFoamMaxwt_hst_Wt2","Histo of weight**2",nBin,0.0,wmax);
   fWtHst1->SetDirectory(0);// exclude from diskfile
   fWtHst2->SetDirectory(0);// and enable deleting
}

//______________________________________________________________________________
TMVA::PDEFoamMaxwt::PDEFoamMaxwt(PDEFoamMaxwt &From): 
   TObject(From),
   fLogger( new MsgLogger("PDEFoamMaxwt") )
{
   // Explicit COPY CONSTRUCTOR (unused, so far)
   fnBin   = From.fnBin;
   fwmax   = From.fwmax;
   fWtHst1 = From.fWtHst1;
   fWtHst2 = From.fWtHst2;
   Error( "PDEFoamMaxwt","COPY CONSTRUCTOR NOT TESTED!");
}

//_______________________________________________________________________________
TMVA::PDEFoamMaxwt::~PDEFoamMaxwt()
{
   // Destructor
   delete fWtHst1; // For this SetDirectory(0) is needed!
   delete fWtHst2; //
   fWtHst1=0;
   fWtHst2=0;
   delete fLogger;
}
//_______________________________________________________________________________
void TMVA::PDEFoamMaxwt::Reset()
{
   // Reseting weight analysis
   fNent = 0;
   fWtHst1->Reset();
   fWtHst2->Reset();
}

//_______________________________________________________________________________
TMVA::PDEFoamMaxwt& TMVA::PDEFoamMaxwt::operator=(const PDEFoamMaxwt &From)
{
   // substitution =
   if (&From == this) return *this;
   fnBin = From.fnBin;
   fwmax = From.fwmax;
   fWtHst1 = From.fWtHst1;
   fWtHst2 = From.fWtHst2;
   return *this;
}

//________________________________________________________________________________
void TMVA::PDEFoamMaxwt::Fill(Double_t wt)
{
   // Filling analyzed weight
   fNent =  fNent+1.0;
   fWtHst1->Fill(wt,1.0);
   fWtHst2->Fill(wt,wt);
}

//________________________________________________________________________________
void TMVA::PDEFoamMaxwt::Make(Double_t eps, Double_t &MCeff)
{
   // Calculates Efficiency= aveWt/wtLim for a given tolerance level epsilon<<1
   // To be called at the end of the MC run.

   Double_t wtLim,aveWt;
   GetMCeff(eps, MCeff, wtLim);
   aveWt = MCeff*wtLim;
   Log()<< "00000000000000000000000000000000000000000000000000000000000000000000000"<<Endl;
   Log()<< "00 -->wtLim: No_evt ="<<fNent<<"   <Wt> = "<<aveWt<<"  wtLim=  "<<wtLim<<Endl;
   Log()<< "00 -->wtLim: For eps = "<<eps  <<"    EFFICIENCY <Wt>/wtLim= "<<MCeff<<Endl;
   Log()<< "00000000000000000000000000000000000000000000000000000000000000000000000"<<Endl;
}

//_________________________________________________________________________________
void TMVA::PDEFoamMaxwt::GetMCeff(Double_t eps, Double_t &MCeff, Double_t &wtLim)
{
   // Calculates Efficiency= aveWt/wtLim for a given tolerance level epsilon<<1
   // using information stored in two histograms.
   // To be called at the end of the MC run.

   Int_t ib,ibX;
   Double_t lowEdge,bin,bin1;
   Double_t aveWt, aveWt1;

   fWtHst1->Print();
   fWtHst2->Print();

   // Convention on bin-numbering: nb=1 for 1-st bin, underflow nb=0, overflow nb=Nb+1
   Double_t sum   = 0.0;
   Double_t sumWt = 0.0;
   for(ib=0;ib<=fnBin+1;ib++) {
      sum   += fWtHst1->GetBinContent(ib);
      sumWt += fWtHst2->GetBinContent(ib);
   }
   if( (sum == 0.0) || (sumWt == 0.0) ) {
      Log()<<"PDEFoamMaxwt::Make: zero content of histogram !!!,sum,sumWt ="<<sum<<sumWt<<Endl;
   }
   aveWt = sumWt/sum;
   //--------------------------------------
   for( ibX=fnBin+1; ibX>0; ibX--) {
      lowEdge = (ibX-1.0)*fwmax/fnBin;
      sum   = 0.0;
      sumWt = 0.0;
      for( ib=0; ib<=fnBin+1; ib++) {
         bin  = fWtHst1->GetBinContent(ib);
         bin1 = fWtHst2->GetBinContent(ib);
         if(ib >= ibX) bin1=lowEdge*bin;
         sum   += bin;
         sumWt += bin1;
      }
      aveWt1 = sumWt/sum;
      if( TMath::Abs(1.0-aveWt1/aveWt) > eps ) break;
   }
   //---------------------------
   if(ibX == (fnBin+1) ) {
      wtLim = 1.0e200;
      MCeff   = 0.0;
      Log()<< "+++++ wtLim undefined. Higher uper limit in histogram"<<Endl;
   } else if( ibX == 1) {
      wtLim = 0.0;
      MCeff   =-1.0;
      Log()<< "+++++ wtLim undefined. Lower uper limit or more bins "<<Endl;
   } else {
      wtLim= (ibX)*fwmax/fnBin; // We over-estimate wtLim, under-estimate MCeff
      MCeff  = aveWt/wtLim;
   }
}
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//      End of    Class  PDEFoamMaxwt                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Auxiliary class PDEFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the PDEFoam  cells                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

ClassImp(TMVA::PDEFoamVect);

//_____________________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect():
   fLogger( new MsgLogger("PDEFoamVect") )
{
   // Default constructor for streamer

   fDim    =0;
   fCoords =0;
   fNext   =0;
   fPrev   =0;
}

//______________________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect(Int_t n):
   fLogger( new MsgLogger("PDEFoamVect") )
{
   // User constructor creating n-dimensional vector
   // and allocating dynamically array of components

   Int_t i;
   fNext=0;
   fPrev=0;
   fDim=n;
   fCoords = 0;
   if (n>0) {
      fCoords = new Double_t[fDim];
      if(gDebug) {
         if(fCoords == 0)
            Error( "PDEFoamVect", "Constructor failed to allocate\n");
      }
      for (i=0; i<n; i++) *(fCoords+i)=0.0;
   }
   if(gDebug) Info("PDEFoamVect", "USER CONSTRUCTOR PDEFoamVect(const Int_t)\n ");
}

//___________________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect(const PDEFoamVect &Vect):
   TObject(Vect),
   fLogger( new MsgLogger("PDEFoamVect") )
{
   // Copy constructor

   fNext=0;
   fPrev=0;
   fDim=Vect.fDim;
   fCoords = 0;
   if(fDim>0)  fCoords = new Double_t[fDim];
   if(gDebug) {
      if(fCoords == 0) {
         Error( "PDEFoamVect", "Constructor failed to allocate fCoords\n");
      }
   }
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = Vect.fCoords[i];
   Error( "PDEFoamVect","+++++ NEVER USE Copy constructor !!!!!\n ");
}

//___________________________________________________________________________
TMVA::PDEFoamVect::~PDEFoamVect()
{
   // Destructor
   if(gDebug) Info("PDEFoamVect"," DESTRUCTOR PDEFoamVect~ \n");
   delete [] fCoords; //  free(fCoords)
   fCoords=0;
   delete fLogger;
}


//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////

//____________________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(const PDEFoamVect& Vect)
{
   // substitution operator

   Int_t i;
   if (&Vect == this) return *this;
   if( fDim != Vect.fDim )
      Error( "PDEFoamVect","operator=Dims. are different: %d and %d \n ",fDim,Vect.fDim);
   if( fDim != Vect.fDim ) {  // cleanup
      delete [] fCoords;
      fCoords = new Double_t[fDim];
   }
   fDim=Vect.fDim;
   for(i=0; i<fDim; i++)
      fCoords[i] = Vect.fCoords[i];
   fNext=Vect.fNext;
   fPrev=Vect.fPrev;
   if(gDebug)  Info("PDEFoamVect", "SUBSITUTE operator =\n ");
   return *this;
}

//______________________________________________________________________
Double_t &TMVA::PDEFoamVect::operator[](Int_t n)
{
   // [] is for access to elements as in ordinary matrix like a[j]=b[j]
   // (Perhaps against some strict rules but rather practical.)
   // Range protection is built in, consequently for substitution
   // one should use rather use a=b than explicit loop!

   if ((n<0) || (n>=fDim)) {
      Error(  "PDEFoamVect","operator[], out of range \n");
   }
   return fCoords[n];
}

//______________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator*=(const Double_t &x)
{
   // unary multiplication operator *=

   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]*x;
   return *this;
}

//_______________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator+=(const PDEFoamVect& Shift)
{
   // unary addition operator +=; adding vector c*=x,
   if( fDim != Shift.fDim){
      Error(  "PDEFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]+Shift.fCoords[i];
   return *this;
}

//________________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator-=(const PDEFoamVect& Shift)
{
   // unary subtraction operator -=
   if( fDim != Shift.fDim) {
      Error(  "PDEFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]-Shift.fCoords[i];
   return *this;
}

//_________________________________________________________________________
TMVA::PDEFoamVect TMVA::PDEFoamVect::operator+(const PDEFoamVect &p2)
{
   // addition operator +; sum of 2 vectors: c=a+b, a=a+b,
   // NEVER USE IT, VERY SLOW!!!
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp += p2;
   return temp;
}

//__________________________________________________________________________
TMVA::PDEFoamVect TMVA::PDEFoamVect::operator-(const PDEFoamVect &p2)
{
   // subtraction operator -; difference of 2 vectors; c=a-b, a=a-b,
   // NEVER USE IT, VERY SLOW!!!
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp -= p2;
   return temp;
}

//___________________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t Vect[])
{
   // Loading in ordinary double prec. vector, sometimes can be useful
   Int_t i;
   for(i=0; i<fDim; i++)
      fCoords[i] = Vect[i];
   return *this;
}

//____________________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t x)
{
   // Loading in double prec. number, sometimes can be useful
   if(fCoords != 0) {
      for(Int_t i=0; i<fDim; i++)
         fCoords[i] = x;
   }
   return *this;
}
//////////////////////////////////////////////////////////////////////////////
//                          OTHER METHODS                                   //
//////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
void TMVA::PDEFoamVect::Print(Option_t *option) const
{
   // Printout of all vector components on "Log()"
   if(!option) Error( "Print ", "No option set \n");
   Int_t i;
   cout << "(";
   for(i=0; i<fDim-1; i++) cout  << SW2 << *(fCoords+i) << ",";
   cout  << SW2 << *(fCoords+fDim-1);
   cout << ")";
}
//______________________________________________________________________________
void TMVA::PDEFoamVect::PrintList(void)
{
   // Printout of all member vectors in the list starting from "this"
   Long_t i=0;
   if(this == 0) return;
   PDEFoamVect *current=this;
   while(current != 0) {
      Log() << "vec["<<i<<"]=";
      current->Print("1");
      Log() << Endl;
      current = current->fNext;
      i++;
   }
}

///////////////////////////////////////////////////////////////////////////////
//                End of Class PDEFoamVect                                     //
///////////////////////////////////////////////////////////////////////////////
