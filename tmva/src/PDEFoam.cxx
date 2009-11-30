
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoam                                                               *
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

//_____________________________________________________________________
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
//_____________________________________________________________________


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>

#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include "TMVA/PDEFoam.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"

#ifndef ROOT_TStyle
#include "TStyle.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TH1D
#include "TH1D.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TVectorT
#include "TVectorT.h"
#endif
#ifndef ROOT_TRandom3
#include "TRandom3.h"
#endif
#ifndef ROOT_TColor
#include "TColor.h"
#endif

ClassImp(TMVA::PDEFoam)

static const Float_t gHigh= FLT_MAX;
static const Float_t gVlow=-FLT_MAX;

using namespace std;

//_____________________________________________________________________
TMVA::PDEFoam::PDEFoam() :
   fLogger(new MsgLogger("PDEFoam"))
{
   // Default constructor for streamer, user should not use it.
   fDim      = 0;
   fNoAct    = 1;
   fNCells   = 0;
   fMaskDiv  = 0;
   fInhiDiv  = 0;
   fCells    = 0;
   fAlpha    = 0;
   fHistEdg  = 0;
   fRvec     = 0;
   fPseRan   = new TRandom3(4356);  // generator of pseudorandom numbers

   fFoamType  = kSeparate;
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

   fDistr = new PDEFoamDistr();
   fDistr->SetSignalClass( fSignalClass );
   fDistr->SetBackgroundClass( fBackgroundClass );

   fTimer = new Timer(fNCells, "PDEFoam", kTRUE);
   fVariableNames = new TObjArray();
}

//_____________________________________________________________________
TMVA::PDEFoam::PDEFoam(const TString& Name) :
   fLogger(new MsgLogger("PDEFoam"))
{
   // User constructor, to be employed by the user

   if(strlen(Name)  >129) {
      Log() << kFATAL << "Name too long " << Name.Data() << Endl;
   }
   fName     = Name;          // Class name
   fMaskDiv  = 0;             // Dynamic Mask for  cell division, h-cubic
   fInhiDiv  = 0;             // Flag allowing to inhibit cell division in certain projection/edge
   fCells    = 0;
   fAlpha    = 0;
   fHistEdg  = 0;
   fDim      = 0;                // dimension of hyp-cubical space
   fNCells   = 1000;             // Maximum number of Cells,    is usually re-set
   fNSampl   = 200;              // No of sampling when dividing cell
   //------------------------------------------------------
   fNBin     = 8;                // binning of edge-histogram in cell exploration
   fEvPerBin =25;                // maximum no. of EFFECTIVE event per bin, =0 option is inactive
   //------------------------------------------------------
   fLastCe   =-1;                // Index of the last cell
   fNoAct    = 1;                // No of active cells (used in MC generation)
   fPseRan   = new TRandom3(4356);   // Initialize private copy of random number generator

   fFoamType  = kSeparate;
   fXmin      = 0;
   fXmax      = 0;
   fCutNmin   = kFALSE;
   fCutRMSmin = kFALSE;
   SetPDEFoamVolumeFraction(-1.);

   fSignalClass     = -1;
   fBackgroundClass = -1;

   fDistr = new PDEFoamDistr();
   fDistr->SetSignalClass( fSignalClass );
   fDistr->SetBackgroundClass( fBackgroundClass );

   fTimer = new Timer(fNCells, "PDEFoam", kTRUE);
   fVariableNames = new TObjArray();

   Log().SetSource( "PDEFoam" );
}

//_____________________________________________________________________
TMVA::PDEFoam::~PDEFoam()
{
   // Default destructor

   delete fVariableNames;
   delete fTimer;
   delete fDistr;
   delete fPseRan;
   if (fXmin) delete [] fXmin;  fXmin=0;
   if (fXmax) delete [] fXmax;  fXmax=0;

   if(fCells!= 0) {
      for(Int_t i=0; i<fNCells; i++) delete fCells[i]; // PDEFoamCell*[]
      delete [] fCells;
   }
   delete [] fRvec;    //double[]
   delete [] fAlpha;   //double[]
   delete [] fMaskDiv; //int[]
   delete [] fInhiDiv; //int[]

   delete fLogger;
}

//_____________________________________________________________________
TMVA::PDEFoam::PDEFoam(const PDEFoam &From):
   TObject(From),
   fLogger( new MsgLogger("PDEFoam"))
{
   // Copy Constructor  NOT IMPLEMENTED (NEVER USED)
   Log() << kFATAL << "COPY CONSTRUCTOR NOT IMPLEMENTED" << Endl;
}

//_____________________________________________________________________
void TMVA::PDEFoam::SetkDim(Int_t kDim) 
{ 
   // Sets dimension of cubical space
   if (kDim < 1)
      Log() << kFATAL << "<SetkDim>: Dimension is zero or negative!" << Endl;

   fDim = kDim;
   if (fXmin) delete [] fXmin;
   if (fXmax) delete [] fXmax;
   fXmin = new Double_t[GetTotDim()];
   fXmax = new Double_t[GetTotDim()];
}

//_____________________________________________________________________
void TMVA::PDEFoam::SetXmin(Int_t idim, Double_t wmin)
{
   // set lower foam bound in dimension idim
   if (idim<0 || idim>=GetTotDim())
      Log() << kFATAL << "<SetXmin>: Dimension out of bounds!" << Endl;

   fXmin[idim]=wmin;
   fDistr->SetXmin(idim, wmin);
}

//_____________________________________________________________________
void TMVA::PDEFoam::SetXmax(Int_t idim, Double_t wmax)
{
   // set upper foam bound in dimension idim
   if (idim<0 || idim>=GetTotDim())
      Log() << kFATAL << "<SetXmax>: Dimension out of bounds!" << Endl;

   fXmax[idim]=wmax;
   fDistr->SetXmax(idim, wmax);
}

//_____________________________________________________________________
void TMVA::PDEFoam::Create(Bool_t CreateCellElements)
{
   // Basic initialization of FOAM invoked by the user.
   // IMPORTANT: Random number generator and the distribution object has to be
   // provided using SetPseRan and SetRho prior to invoking this initializator!

   Bool_t addStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);

   if(fPseRan==0) Log() << kFATAL << "Random number generator not set" << Endl;
   if(fDistr==0)  Log() << kFATAL << "Distribution function not set" << Endl;
   if(fDim==0)    Log() << kFATAL << "Zero dimension not allowed" << Endl;

   /////////////////////////////////////////////////////////////////////////
   //                   ALLOCATE SMALL LISTS                              //
   //  it is done globally, not for each cell, to save on allocation time //
   /////////////////////////////////////////////////////////////////////////
   fRvec = new Double_t[fDim];   // Vector of random numbers
   if(fRvec==0)  Log() << kFATAL << "Cannot initialize buffer fRvec" << Endl;

   if(fDim>0){
      fAlpha = new Double_t[fDim];    // sum<1 for internal parametrization of the simplex
      if(fAlpha==0)  Log() << kFATAL << "Cannot initialize buffer fAlpha" << Endl;
   }

   //====== List of directions inhibited for division
   if(fInhiDiv == 0){
      fInhiDiv = new Int_t[fDim];
      for(Int_t i=0; i<fDim; i++) fInhiDiv[i]=0;
   }
   //====== Dynamic mask used in Explore for edge determination
   if(fMaskDiv == 0){
      fMaskDiv = new Int_t[fDim];
      for(Int_t i=0; i<fDim; i++) fMaskDiv[i]=1;
   }
   //====== Initialize list of histograms
   fHistEdg = new TObjArray(fDim);           // Initialize list of histograms
   for(Int_t i=0; i<fDim; i++){
      TString hname, htitle;
      hname   = fName+TString("_HistEdge_");
      hname  += i;
      htitle  = TString("Edge Histogram No. ");
      htitle += i;
      (*fHistEdg)[i] = new TH1D(hname.Data(),htitle.Data(),fNBin,0.0, 1.0); // Initialize histogram for each edge
      ((TH1D*)(*fHistEdg)[i])->Sumw2();
   }

   // ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| //
   //                     BUILD-UP of the FOAM                            //
   // ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| //

   // Define and explore root cell(s)
   InitCells(CreateCellElements);
   Grow();

   TH1::AddDirectory(addStatus);
} // Create

//_____________________________________________________________________
void TMVA::PDEFoam::InitCells(Bool_t CreateCellElements)
{
   // Internal subprogram used by Create.
   // It initializes "root part" of the FOAM of the tree of cells.

   fLastCe =-1;                             // Index of the last cell
   if(fCells!= 0) {
      for(Int_t i=0; i<fNCells; i++) delete fCells[i];
      delete [] fCells;
   }
   //
   fCells = new PDEFoamCell*[fNCells];
   for(Int_t i=0; i<fNCells; i++){
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

//_____________________________________________________________________
Int_t TMVA::PDEFoam::CellFill(Int_t Status, PDEFoamCell *parent)
{
   // Internal subprogram used by Create.
   // It initializes content of the newly allocated active cell.

   PDEFoamCell *cell;
   if (fLastCe==fNCells){
      Log() << kFATAL << "Too many cells" << Endl;
   }
   fLastCe++;   // 0-th cell is the first

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

//_____________________________________________________________________
void TMVA::PDEFoam::Explore(PDEFoamCell *cell)
{
   // Internal subprogram used by Create.
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
      toteventsOld = GetBuildUpCellEvents(cell);

   /////////////////////////////////////////////////////
      //    Special Short MC sampling to probe cell      //
      /////////////////////////////////////////////////////
      ceSum[0]=0;
      ceSum[1]=0;
      ceSum[2]=0;
      ceSum[3]=gHigh;  //wtmin
      ceSum[4]=gVlow;  //wtmax

      for (i=0;i<fDim;i++) ((TH1D *)(*fHistEdg)[i])->Reset(); // Reset histograms

      Double_t nevEff=0.;
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
      kBest=-1;

      //------------------------------------------------------------------
      nevMC            = ceSum[2];
      Double_t intTrue = ceSum[0]/(nevMC+0.000001);
      Double_t intDriv=0.;

      if (kBest == -1) Varedu(ceSum,kBest,xBest,yBest); // determine the best edge,
      if (CutRMSmin())
         intDriv =sqrt( ceSum[1]/nevMC -intTrue*intTrue ); // Older ansatz, numerically not bad
      else
         intDriv =sqrt(ceSum[1]/nevMC) -intTrue; // Foam build-up, sqrt(<w**2>) -<w>

      //=================================================================================
      cell->SetBest(kBest);
      cell->SetXdiv(xBest);
      cell->SetIntg(intTrue);
      cell->SetDriv(intDriv);
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
            SetCellElement( parent, 0, GetBuildUpCellEvents(parent) + totevents - toteventsOld);
      }
      delete [] volPart;
      delete [] xRand;
}

//_____________________________________________________________________
void TMVA::PDEFoam::Varedu(Double_t ceSum[5], Int_t &kBest, Double_t &xBest, Double_t &yBest)
{
   // Internal subrogram used by Create.
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

   if( (kBest >= fDim) || (kBest<0) )
      Log() << kFATAL << "Something wrong with kBest" << Endl;
}          //PDEFoam::Varedu

//_____________________________________________________________________
void TMVA::PDEFoam::MakeAlpha()
{
   // Internal subrogram used by Create.
   // Provides random vector Alpha  0< Alpha(i) < 1

   // simply generate and load kDim uniform random numbers
   fPseRan->RndmArray(fDim,fRvec);   // kDim random numbers needed
   for(Int_t k=0; k<fDim; k++) fAlpha[k] = fRvec[k];
} //MakeAlpha

//_____________________________________________________________________
Long_t TMVA::PDEFoam::PeekMax()
{
   // Internal subprogram used by Create.
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
            bCutNmin = GetBuildUpCellEvents(fCells[i]) > GetNmin();

         // choose cell
         if(driv > drivMax && bCutNmin && bCutRMS) {
            drivMax = driv;
            iCell = i;
         }
      }
   }

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

//_____________________________________________________________________
Int_t TMVA::PDEFoam::Divide(PDEFoamCell *cell)
{
   // Internal subrogram used by Create.
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
   fNoAct++;

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

//_____________________________________________________________________
Double_t TMVA::PDEFoam::Eval(Double_t *xRand, Double_t &event_density)
{
   // Internal subprogram.
   // Evaluates distribution to be generated.
   return fDistr->Density(xRand, event_density);
}

//_____________________________________________________________________
void TMVA::PDEFoam::Grow()
{
   // Internal subrogram used by Create.
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

//_____________________________________________________________________
void  TMVA::PDEFoam::SetInhiDiv(Int_t iDim, Int_t InhiDiv)
{
   // This can be called before Create, after setting kDim
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

//_____________________________________________________________________
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

//_____________________________________________________________________
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

//_____________________________________________________________________
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

//_____________________________________________________________________
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

//_____________________________________________________________________
void TMVA::PDEFoam::PrintCellElements()
{
   // This debug function prints the cell elements of all active
   // cells.

   for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
      if (!fCells[iCell]->GetStat()) continue;

      Log() << "cell[" << iCell << "] elements: [";
      for (UInt_t i=0; i<GetNElements(); i++){
         if (i>0) Log() << " ; ";
         Log() << GetCellElement(fCells[iCell], i);
      }
      Log() << "]" << Endl;
   }
}

//_____________________________________________________________________
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

//_____________________________________________________________________
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

      if (N_ev > 1e-20){
         SetCellElement(fCells[iCell], 0, tar/N_ev);  // set average target
         SetCellElement(fCells[iCell], 1, tar/TMath::Sqrt(N_ev)); // set error on average target
      }
      else {
         SetCellElement(fCells[iCell], 0, 0.0 ); // set mean target
         SetCellElement(fCells[iCell], 1, -1  ); // set mean target error
      }
   }
}

//_____________________________________________________________________
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

//_____________________________________________________________________
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

   if (kernel == kNone) result = GetCellValue(cell, kDiscriminator);
   else if (kernel == kGaus) {
      Double_t norm = 0.;

      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_discr = GetCellValue(fCells[iCell], kDiscriminator);
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

//_____________________________________________________________________
void TMVA::PDEFoam::FillFoamCells(const Event* ev, Bool_t NoNegWeights)
{
   // This function fills an event into the foam.
   //
   // In case of Mono-Target regression this function prepares the
   // calculation of the average target value in every cell.  Note,
   // that only target 0 is saved in the cell!
   //
   // In case of a unified foam this function prepares the calculation of
   // the cell discriminator in every cell.
   //
   // If 'NoNegWeights' is true, an event with negative weight will
   // not be filled into the foam.  (Default value: false)

   std::vector<Float_t> values  = ev->GetValues();
   std::vector<Float_t> targets = ev->GetTargets();
   Float_t weight               = ev->GetOriginalWeight();
   EFoamType ft                 = GetFoamType();

   if((NoNegWeights && weight<=0) || weight==0)
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

//_____________________________________________________________________
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
      if (GetCellValue(cell, kTarget0Error)!=-1)
         // cell is not empty
         result = GetCellValue(cell, kTarget0);
      else
         // cell is empty -> calc average target of neighbor cells
         result = GetAverageNeighborsValue(txvec, kTarget0);
   } // kernel == kNone
   else if (kernel == kGaus){
      // return gaus weighted cell density

      Double_t norm = 0.;
      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_val = 0;
         if (GetCellValue(fCells[iCell], kTarget0Error) != -1)
            // cell is not empty
            cell_val = GetCellValue(fCells[iCell], kTarget0);
         else
            // cell is empty -> calc average target of neighbor cells
            cell_val = GetAverageNeighborsValue(txvec, kTarget0);
         Double_t gau = WeightGaus(fCells[iCell], txvec);

         result += gau * cell_val;
         norm   += gau;
      }
      result /= norm;
   }
   else if (kernel == kLinN) {
      if (GetCellValue(cell, kTarget0Error) != -1)
         // cell is not empty -> weight with non-empty neighbors
         result = WeightLinNeighbors(txvec, kTarget0, -1, -1, kTRUE);
      else
         // cell is empty -> calc average target of non-empty neighbor
         // cells
         result = GetAverageNeighborsValue(txvec, kTarget0);
   }
   else {
      Log() << kFATAL << "ERROR: unknown kernel!" << Endl;
      return -999.;
   }

   return result;
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetAverageNeighborsValue( std::vector<Float_t> txvec,
                                                  ECellValue cv )
{
   // This function returns the average value 'cv' of only nearest
   // neighbor cells.  It is used in cases, where empty cells shall
   // not be evaluated.
   //
   // Parameters:
   // - txvec - event vector, transformed into foam [0, 1]
   // - cv - cell value, see definition of ECellValue

   const Double_t xoffset = 1.e-6;
   Double_t norm   = 0; // normalisation
   Double_t result = 0; // return value

   PDEFoamCell *cell = FindCell(txvec); // find cooresponding cell
   PDEFoamVect cellSize(GetTotDim());
   PDEFoamVect cellPosi(GetTotDim());
   cell->GetHcub(cellPosi, cellSize); // get cell coordinates

   // loop over all dimensions and find neighbor cells
   for (Int_t dim=0; dim<GetTotDim(); dim++) {
      std::vector<Float_t> ntxvec = txvec;
      PDEFoamCell* left_cell  = 0; // left cell
      PDEFoamCell* right_cell = 0; // right cell

      // get left cell
      ntxvec[dim] = cellPosi[dim]-xoffset;
      left_cell = FindCell(ntxvec);
      if (!CellValueIsUndefined(left_cell)){
         // if left cell is not empty, take its value
         result += GetCellValue(left_cell, cv);
         norm++;
      }
      // get right cell
      ntxvec[dim] = cellPosi[dim]+cellSize[dim]+xoffset;
      right_cell = FindCell(ntxvec);
      if (!CellValueIsUndefined(right_cell)){
         // if right cell is not empty, take its value
         result += GetCellValue(right_cell, cv);
         norm++;
      }
   }
   if (norm>0)  result /= norm; // calc average target
   else         result = 0;     // return null if all neighbors are empty

   return result;
}

//_____________________________________________________________________
Bool_t TMVA::PDEFoam::CellValueIsUndefined( PDEFoamCell* cell )
{
   // Returns true, if the value of the given cell is undefined.

   EFoamType ft = GetFoamType();
   switch(ft){
   case kSeparate:
      return kFALSE;
      break;
   case kDiscr:
      return kFALSE;
      break;
   case kMonoTarget:
      return GetCellValue(cell, kTarget0Error) == -1;
      break;
   case kMultiTarget:
      return kFALSE;
      break;
   default:
      return kFALSE;
   }
}

//_____________________________________________________________________
std::vector<Float_t> TMVA::PDEFoam::GetCellTargets( std::vector<Float_t> tvals, ETargetSelection ts )
{
   // This function is used when the MultiTargetRegression==True
   // option is set.  It calculates the mean target or most probable
   // target values if 'tvals' variables are given ('tvals' does not
   // contain targets)
   //
   // Parameters:
   // - tvals - transformed event variables (element of [0,1]) (no targets!)
   // - ts - method of target selection (kMean, kMpv)
   //
   // Result:
   // vetor of mean targets or most probable targets over all cells
   // which first coordinates enclose 'tvals'

   std::vector<Float_t> target(GetTotDim()-tvals.size(), 0); // returned vector
   std::vector<Float_t> norm(target); // normalisation
   Double_t max_dens = 0.;            // maximum cell density

   // find cells, which fit tvals (no targets)
   std::vector<PDEFoamCell*> cells = FindCells(tvals);
   if (cells.size()<1) return target;

   // loop over all cells found
   std::vector<PDEFoamCell*>::iterator cell_it(cells.begin());
   for (cell_it=cells.begin(); cell_it!=cells.end(); cell_it++){

      // get density of cell
      Double_t cell_density = GetCellValue(*cell_it, kDensity);

      // get cell position and size
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      (*cell_it)->GetHcub(cellPosi, cellSize);

      // loop over all target dimensions, in order to calculate target
      // value
      if (ts==kMean){
         // sum cell density times cell center
         for (UInt_t itar=0; itar<target.size(); itar++){
            UInt_t idim = itar+tvals.size();
            target.at(itar) += cell_density *
               VarTransformInvers(idim, cellPosi[idim]+0.5*cellSize[idim]);
            norm.at(itar) += cell_density;
         } // loop over targets
      } else {
         // get cell center with maximum event density
         if (cell_density > max_dens){
            max_dens = cell_density; // save new max density
            // fill target values
            for (UInt_t itar=0; itar<target.size(); itar++){
               UInt_t idim = itar+tvals.size();
               target.at(itar) =
                  VarTransformInvers(idim, cellPosi[idim]+0.5*cellSize[idim]);
            } // loop over targets
         }
      }
   } // loop over cells

   // normalise mean cell density
   if (ts==kMean){
      for (UInt_t itar=0; itar<target.size(); itar++){
         if (norm.at(itar)>1.0e-15)
            target.at(itar) /= norm.at(itar);
         else
            // normalisation factor is too small -> return approximate
            // target value
            target.at(itar) = (fXmax[itar+tvals.size()]-fXmin[itar+tvals.size()])/2.;
      }
   }

   return target;
}

//_____________________________________________________________________
std::vector<Float_t> TMVA::PDEFoam::GetProjectedRegValue( std::vector<Float_t> vals, EKernel kernel, ETargetSelection ts )
{
   // This function is used when the MultiTargetRegression==True option is set.
   // Returns regression value i, given the event variables 'vals'.
   // Note: number of foam dimensions = number of variables + number of targets
   //
   // Parameters:
   // - vals - event variables (no targets)
   // - kernel - used kernel (None or Gaus)
   // - ts - method of target selection (Mean or Mpv)

   // checkt whether vals are within foam borders.
   // if not -> push it into foam
   const Float_t xsmall = 1.e-7;
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
   std::vector<Float_t> target(GetTotDim()-txvec.size(), 0); // returned vector

   // choose kernel
   /////////////////// no kernel //////////////////
      if (kernel == kNone)
         return GetCellTargets(txvec, ts);
      ////////////////////// Gaus kernel //////////////////////
      else if (kernel == kGaus){

         std::vector<Float_t> norm(target); // normalisation

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

            std::vector<Float_t> val = GetCellTargets(new_vec, ts);
            for (UInt_t itar=0; itar<target.size(); itar++){
               target.at(itar) += gau * val.at(itar);
               norm.at(itar)   += gau;
            }
         }

         // normalisation
         for (UInt_t itar=0; itar<target.size(); itar++){
            if (norm.at(itar)<1.0e-20){
               Log() << kWARNING << "Warning: norm too small!" << Endl;
               target.at(itar) = 0.;
            } else
               target.at(itar) /= norm.at(itar);
         }

      } // end gaus kernel
      else {
         Log() << kFATAL << "<GetProjectedRegValue> ERROR: unsupported kernel!" << Endl;
         return std::vector<Float_t>(GetTotDim()-txvec.size(), 0);
      }

      return target;
}

//_____________________________________________________________________
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
      return GetCellValue(cell, kDensity);
   }
   else if (kernel == kGaus){
      // return gaus weighted cell density

      Double_t norm = 0.;

      for (Long_t iCell=0; iCell<=fLastCe; iCell++) {
         if (!(fCells[iCell]->GetStat())) continue;

         // calc cell density
         Double_t cell_dens = GetCellValue(fCells[iCell], kDensity);
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

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetCellValue( PDEFoamCell* cell, ECellValue cv )
{
   // This function returns a value, which was saved in the foam cell,
   // depending on the foam type.  The value to return is specified
   // with the 'cv' parameter.

   switch(cv){
      
   case kTarget0:
      if (GetFoamType() == kMonoTarget) return GetCellElement(cell, 0);
      break;

   case kTarget0Error:
      if (GetFoamType() == kMonoTarget) return GetCellElement(cell, 1);
      break;

   case kDiscriminator:
      if (GetFoamType() == kDiscr) return GetCellElement(cell, 0);
      break;

   case kDiscriminatorError:
      if (GetFoamType() == kDiscr) return GetCellElement(cell, 1);
      break;

   case kMeanValue:
      return cell->GetIntg();
      break;

   case kRms:
      return cell->GetDriv();
      break;

   case kRmsOvMean:
      if (cell->GetIntg() != 0) return cell->GetDriv()/cell->GetIntg();
      break;

   case kNev:
      if (GetFoamType() == kSeparate || GetFoamType() == kMultiTarget) 
         return GetCellElement(cell, 0);
      break;

   case kDensity: {
      
      Double_t volume  = cell->GetVolume();
      if ( volume > 1.0e-10 ){
         return GetCellValue(cell, kNev)/volume;
      } else {
         if (volume<=0){
            cell->Print("1"); // debug output
            Log() << kWARNING << "<GetCellDensity(cell)>: ERROR: cell volume"
                  << " negative or zero!"
                  << " ==> return cell density 0!"
                  << " cell volume=" << volume
                  << " cell entries=" << GetCellValue(cell, kNev) << Endl;
            return 0;
         } else
            Log() << kWARNING << "<GetCellDensity(cell)>: WARNING: cell volume"
                  << " close to zero!"
                  << " cell volume: " << volume << Endl;
      }
   } // kDensity
      
   default:
      return 0;
   }

   return 0;
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetCellValue(std::vector<Float_t> xvec, ECellValue cv)
{
   // This function finds the cell, which corresponds to the given
   // event vector 'xvec' and return its value, which is given by the
   // parameter 'cv'.
   
   return GetCellValue(FindCell(VarTransform(xvec)), cv);
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetBuildUpCellEvents( PDEFoamCell* cell )
{
   // Returns the number of events, saved in the 'cell' during foam build-up.
   // Only used during foam build-up!
   return GetCellElement(cell, 0);
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::WeightLinNeighbors( std::vector<Float_t> txvec, ECellValue cv, Int_t dim1, Int_t dim2, Bool_t TreatEmptyCells )
{
   // results the cell value, corresponding to txvec, weighted by the
   // neighor cells via a linear function
   //
   // Parameters
   //  - txvec - event vector, transformed to interval [0,1]
   //  - cv - cell value to be weighted
   //  - dim1, dim2 - dimensions for two-dimensional projection.
   //    Default values: dim1 = dim2 = -1
   //    If dim1 and dim2 are set to values >=0 and <fDim, than
   //    the function GetProjectionCellValue() is used to get cell
   //    value.  This is used for projection to two dimensions within
   //    Project2().
   //  - TreatEmptyCells - if this option is set false (default),
   //    it is not checked, wether the cell or its neighbors are empty
   //    or not.  If this option is set true, than only non-empty
   //    neighbor cells are taken into account for weighting.  If the
   //    cell, which contains txvec is empty, than its value is
   //    replaced by the average value of the non-empty neighbor cells

   Double_t result = 0.;
   UInt_t norm     = 0;
   const Double_t xoffset = 1.e-6;

   if (txvec.size() != UInt_t(GetTotDim()))
      Log() << kFATAL << "Wrong dimension of event variable!" << Endl;

   // find cell, which contains txvec
   PDEFoamCell *cell= FindCell(txvec);
   PDEFoamVect cellSize(GetTotDim());
   PDEFoamVect cellPosi(GetTotDim());
   cell->GetHcub(cellPosi, cellSize);
   // calc value of cell, which contains txvec
   Double_t cellval = 0;
   if (!(TreatEmptyCells && CellValueIsUndefined(cell)))
      // cell is not empty -> get cell value
      cellval = GetCellValue(cell, cv);
   else
      // cell is empty -> get average value of non-empty neighbor
      // cells
      cellval = GetAverageNeighborsValue(txvec, cv);

   // loop over all dimensions to find neighbor cells
   for (Int_t dim=0; dim<GetTotDim(); dim++) {
      std::vector<Float_t> ntxvec = txvec;
      Double_t mindist;
      PDEFoamCell *mindistcell = 0; // cell with minimal distance to txvec
      // calc minimal distance to neighbor cell
      mindist = (txvec[dim]-cellPosi[dim])/cellSize[dim];
      if (mindist<0.5) { // left neighbour
         ntxvec[dim] = cellPosi[dim]-xoffset;
         mindistcell = FindCell(ntxvec); // left neighbor cell
      } else { // right neighbour
         mindist=1-mindist;
         ntxvec[dim] = cellPosi[dim]+cellSize[dim]+xoffset;
         mindistcell = FindCell(ntxvec); // right neighbor cell
      }
      Double_t mindistcellval = 0; // value of cell, which contains ntxvec
      if (dim1>=0 && dim1<GetTotDim() &&
          dim2>=0 && dim2<GetTotDim() &&
          dim1!=dim2){
         cellval        = GetProjectionCellValue(cell, dim1, dim2, cv);
         mindistcellval = GetProjectionCellValue(mindistcell, dim1, dim2, cv);
      } else {
         mindistcellval = GetCellValue(mindistcell, cv);
      }
      // if treatment of empty neighbor cells is deactivated, do
      // normal weighting
      if (!(TreatEmptyCells && CellValueIsUndefined(mindistcell))){
         result += cellval        * (0.5 + mindist);
         result += mindistcellval * (0.5 - mindist);
         norm++;
      }
   }
   if (norm==0) return cellval;     // all nearest neighbors were empty
   else         return result/norm; // normalisation
}

//_____________________________________________________________________
Float_t TMVA::PDEFoam::WeightGaus( PDEFoamCell* cell, std::vector<Float_t> txvec,
                                   UInt_t dim )
{
   // Returns the gauss weight between the 'cell' and a given coordinate 'txvec'.
   //
   // Parameters:
   // - cell - the cell
   // - txvec - the transformed event variables (in [0,1]) (coordinates <0 are
   //   set to 0, >1 are set to 1)
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

   Float_t distance = 0.; // distance for weighting
   for (UInt_t i=0; i<dims; i++)
      distance += TMath::Power(txvec.at(i)-cell_center.at(i), 2);
   distance = TMath::Sqrt(distance);

   Float_t width = 1./GetPDEFoamVolumeFraction();
   if (width < 1.0e-10)
      Log() << kWARNING << "Warning: wrong volume fraction: " << GetPDEFoamVolumeFraction() << Endl;

   // weight with Gaus with sigma = 1/VolFrac
   return TMath::Gaus(distance, 0, width, kFALSE);
}

//_____________________________________________________________________
TMVA::PDEFoamCell* TMVA::PDEFoam::FindCell( std::vector<Float_t> xvec )
{
   // Find cell that contains xvec
   //
   // loop to find cell that contains xvec start from root Cell, uses
   // binary tree to find cell quickly

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

//_____________________________________________________________________
void TMVA::PDEFoam::FindCellsRecursive(std::vector<Float_t> txvec, PDEFoamCell* cell, std::vector<PDEFoamCell*> &cells)
{
   // This is a helper function for FindCells().  It saves in 'cells'
   // all cells, which contain txvec.  It works analogous to
   // FindCell().
   //
   // Parameters:
   //
   // - txvec - vector of variables (no targets!) (transformed into
   //   foam)
   //
   // - cell - cell to start searching with (usually root cell
   //   fCells[0])
   //
   // - cells - list of cells found

   PDEFoamVect  cellPosi0(GetTotDim()), cellSize0(GetTotDim());
   PDEFoamCell *cell0;
   Int_t idim=0;

   while (cell->GetStat()!=1) { //go down binary tree until cell is found
      idim=cell->GetBest();  // dimension that changed

      if (idim < Int_t(txvec.size())){
         // case 1: cell is splitten in dimension of a variable
         cell0=cell->GetDau0();
         cell0->GetHcub(cellPosi0,cellSize0);
         // check, whether left daughter cell contains txvec
         if (txvec.at(idim)<=cellPosi0[idim]+cellSize0[idim])
            cell=cell0;
         else
            cell=cell->GetDau1();
      } else {
         // case 2: cell is splitten in target dimension
         FindCellsRecursive(txvec, cell->GetDau0(), cells);
         FindCellsRecursive(txvec, cell->GetDau1(), cells);
         return;
      }
   }
   cells.push_back(cell);
}

//_____________________________________________________________________
std::vector<TMVA::PDEFoamCell*> TMVA::PDEFoam::FindCells(std::vector<Float_t> txvec)
{
   // Find all cells, that contain txvec.  This function can be used,
   // when the dimension of the foam is greater than the dimension of
   // txvec.  E.G this is the case for multi-target regression
   //
   // Parameters:
   //
   // - txvec - vector of variables (no targets!) (transformed into
   //   foam)
   //
   // Return value:
   //
   // - vector of cells, that fit txvec

   std::vector<PDEFoamCell*> cells(0);

   // loop over all target dimensions
   FindCellsRecursive(txvec, fCells[0], cells);

   return cells;
}

//_____________________________________________________________________
TH1D* TMVA::PDEFoam::Draw1Dim( const char *opt, Int_t nbin )
{
   // Draws 1-dimensional foam (= histogram)
   //
   // Parameters:
   //
   // - opt - cell_value, rms, rms_ov_mean
   //   if cell_value is set, the following values will be filled into
   //   the result histogram:
   //    - number of events - in case of classification with 2 separate
   //                         foams or multi-target regression
   //    - discriminator    - in case of classification with one
   //                         unified foam
   //    - target           - in case of mono-target regression
   //
   // - nbin - number of bins of result histogram
   //
   // Warning: This function is not well tested!

   // avoid plotting of wrong dimensions
   if ( GetTotDim()!=1 ) return 0;

   // select value to plot
   ECellValue cell_value = kNev;
   EFoamType  foam_type  = GetFoamType();
   if (strcmp(opt,"cell_value")==0){
      if (foam_type == kSeparate || foam_type == kMultiTarget){
         cell_value = kNev;
      } else if (foam_type == kDiscr){
         cell_value = kDiscriminator;
      } else if (foam_type == kMonoTarget){
         cell_value = kTarget0;
      } else {
         Log() << kFATAL << "unknown foam type" << Endl;
         return 0;
      }
   } else if (strcmp(opt,"rms")==0){
      cell_value = kRms;
   } else if (strcmp(opt,"rms_ov_mean")==0){
      cell_value = kRmsOvMean;
   } else {
      Log() << kFATAL << "<Draw1Dim>: unknown option:" << opt << Endl;
      return 0;
   }

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

         // filling value to histogram
         h1->SetBinContent(ibinx, 
                           GetCellValue(fCells[iCell], cell_value) + h1->GetBinContent(ibinx));
      }
   }
   return h1;
}

//_____________________________________________________________________
TH2D* TMVA::PDEFoam::Project2( Int_t idim1, Int_t idim2, const char *opt, const char *ker, UInt_t maxbins )
{
   // Project foam variable idim1 and variable idim2 to histogram.
   //
   // Parameters:
   //
   // - idim1, idim2 - dimensions to project to
   //
   // - opt - cell_value, rms, rms_ov_mean
   //   if cell_value is set, the following values will be filled into
   //   the result histogram:
   //    - number of events - in case of classification with 2 separate
   //                         foams or multi-target regression
   //    - discriminator    - in case of classification with one
   //                         unified foam
   //    - target           - in case of mono-target regression
   //
   // - ker - kGaus, kNone (warning: Gaus may be very slow!)
   //
   // - maxbins - maximal number of bins in result histogram.
   //   Set maxbins to 0 if no maximum bin number should be used.
   //
   // Returns:
   // a 2-dimensional histogram

   // avoid plotting of wrong dimensions
   if ((idim1>=GetTotDim()) || (idim1<0) ||
       (idim2>=GetTotDim()) || (idim2<0) ||
       (idim1==idim2) )
      return 0;

   // select value to plot
   ECellValue cell_value = kNev;
   EFoamType  foam_type  = GetFoamType();
   if (strcmp(opt,"cell_value")==0){
      if (foam_type == kSeparate || foam_type == kMultiTarget){
         cell_value = kNev;
      } else if (foam_type == kDiscr){
         cell_value = kDiscriminator;
      } else if (foam_type == kMonoTarget){
         cell_value = kTarget0;
      } else {
         Log() << kFATAL << "unknown foam type" << Endl;
         return 0;
      }
   } else if (strcmp(opt,"rms")==0){
      cell_value = kRms;
   } else if (strcmp(opt,"rms_ov_mean")==0){
      cell_value = kRmsOvMean;
   } else {
      Log() << kFATAL << "unknown option given" << Endl;
      return 0;
   }

   // select kernel to use
   EKernel kernel = kNone;
   if (!strcmp(ker, "kNone"))
      kernel = kNone;
   else if (!strcmp(ker, "kGaus"))
      kernel = kGaus;
   else if (!strcmp(ker, "kLinN"))
      kernel = kLinN;
   else
      Log() << kWARNING << "Warning: wrong kernel! using kNone instead" << Endl;

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
            << " Using 1000 bins for each dimension instead." << Endl;
      nbin = 1000;
   }

   // create result histogram
   char hname[100], htit[100];
   sprintf(htit,"%s var%d vs var%d",opt,idim1,idim2);
   sprintf(hname,"h%s_%d_vs_%d",opt,idim1,idim2);

   // if histogram with this name already exists, delete it
   TH2D* h1=(TH2D*)gDirectory->Get(hname);
   if (h1) delete h1;
   h1= new TH2D(hname, htit, nbin, fXmin[idim1], fXmax[idim1], nbin, fXmin[idim2], fXmax[idim2]);

   if (!h1) Log() << kFATAL << "ERROR: Can not create histo" << hname << Endl;

   // ============== start projection algorithm ================
   // loop over all active cells
   for (Long_t iCell=0; iCell<=fLastCe; iCell++) { // loop over all active cells
      if (!(fCells[iCell]->GetStat())) continue;   // cell not active -> continue

      // get cell position and dimesions
      PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
      fCells[iCell]->GetHcub(cellPosi,cellSize);

      // get cell value (depending on the option)
      // this value will later be filled into the histogram
      Double_t var = GetProjectionCellValue(fCells[iCell], idim1, idim2, cell_value);

      const Double_t xsmall = (1.e-20)*cellSize[idim1];
      const Double_t ysmall = (1.e-20)*cellSize[idim2];

      // coordinates of upper left corner of cell
      Double_t x1 = VarTransformInvers( idim1, cellPosi[idim1]+xsmall );
      Double_t y1 = VarTransformInvers( idim2, cellPosi[idim2]+ysmall );

      // coordinates of lower right corner of cell
      Double_t x2 = VarTransformInvers( idim1, cellPosi[idim1]+cellSize[idim1]-xsmall );
      Double_t y2 = VarTransformInvers( idim2, cellPosi[idim2]+cellSize[idim2]-ysmall );

      // most left and most right bins, which correspond to cell
      // borders
      Int_t xbin_start = h1->GetXaxis()->FindBin(x1);
      Int_t xbin_stop  = h1->GetXaxis()->FindBin(x2);

      // upper and lower bins, which correspond to cell borders
      Int_t ybin_start = h1->GetYaxis()->FindBin(y1);
      Int_t ybin_stop  = h1->GetYaxis()->FindBin(y2);

      // loop over all bins, which the cell occupies
      for (Int_t ibinx=xbin_start; ibinx<xbin_stop; ibinx++) {    //loop over x-bins
         for (Int_t ibiny=ybin_start; ibiny<ybin_stop; ibiny++) { //loop over y-bins

            ////////////////////// weight with kernel ///////////////////////
            if (kernel == kGaus){
               Double_t result = 0.;
               Double_t norm   = 0.;

               // calc current position (depending on ibinx, ibiny)
               Double_t x_curr =
                  VarTransform( idim1, ((x2-x1)*ibinx - x2*xbin_start + x1*xbin_stop)/(xbin_stop-xbin_start) );
               Double_t y_curr =
                  VarTransform( idim2, ((y2-y1)*ibiny - y2*ybin_start + y1*ybin_stop)/(ybin_stop-ybin_start) );

               // loop over all active cells
               for (Long_t ice=0; ice<=fLastCe; ice++) {
                  if (!(fCells[ice]->GetStat())) continue;

                  // get cell value (depending on option)
                  Double_t cell_var = GetProjectionCellValue(fCells[ice], idim1, idim2, cell_value);

                  // fill ndim coordinate of current cell
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
                  Double_t weight_ = WeightGaus(fCells[ice], coor);

                  result += weight_ * cell_var;
                  norm   += weight_;
               }
               var = result/norm;
            }
            else if (kernel == kLinN){
               // calc current position (depending on ibinx, ibiny)
               Double_t x_curr =
                  VarTransform( idim1, ((x2-x1)*ibinx - x2*xbin_start + x1*xbin_stop)/(xbin_stop-xbin_start) );
               Double_t y_curr =
                  VarTransform( idim2, ((y2-y1)*ibiny - y2*ybin_start + y1*ybin_stop)/(ybin_stop-ybin_start) );

               // fill ndim coordinate of current cell
               std::vector<Float_t> coor;
               for (Int_t i=0; i<GetTotDim(); i++) {
                  if (i == idim1)
                     coor.push_back(x_curr);
                  else if (i == idim2)
                     coor.push_back(y_curr);
                  else
                     coor.push_back(cellPosi[i] + 0.5*cellSize[i]); // approximation
               }

               var = WeightLinNeighbors(coor, cell_value, idim1, idim2);
            }
            ////////////////////// END weight with kernel ///////////////////////

            // filling value to histogram
            h1->SetBinContent(ibinx, ibiny, var + h1->GetBinContent(ibinx, ibiny));
         } // y-loop
      } // x-loop
   } // cell loop

   return h1;
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetProjectionCellValue( PDEFoamCell* cell,
                                                Int_t idim1,
                                                Int_t idim2,
                                                ECellValue cv )
{
   // Helper function for projection function Project2().  It returns
   // the cell value of 'cell' corresponding to the given option 'cv'.
   // The two dimensions are needed for weighting the return value,
   // because Project2() projects the foam to two dimensions.

   Double_t val = 0.; // return variable

   // get cell position and dimesions
   PDEFoamVect  cellPosi(GetTotDim()), cellSize(GetTotDim());
   cell->GetHcub(cellPosi,cellSize);
   const Double_t foam_area = (fXmax[idim1]-fXmin[idim1])*(fXmax[idim2]-fXmin[idim2]);

   // calculate cell value (depending on the given option 'cv')
   if (cv == kNev){
      // calculate projected area of cell
      Double_t area = cellSize[idim1] * cellSize[idim2];
      if (area<1e-20){
         Log() << kWARNING << "PDEFoam::Project2: Warning, cell volume too small --> skiping cell!" << Endl;
         return 0;
      }

      // calc cell entries per projected cell area
      val = GetCellValue(cell, kNev)/(area*foam_area);
   }
   // =========================================================
   else if (cv == kRms){
      val = GetCellValue(cell, kRms);
   }
   // =========================================================
   else if (cv == kRmsOvMean){
      val = GetCellValue(cell, kRmsOvMean);
   }
   // =========================================================
   else if (cv == kDiscriminator){
      // calculate cell volume in other dimensions (not including idim1 and idim2)
      Double_t area_cell = 1.;
      for (Int_t d1=0; d1<GetTotDim(); d1++){
         if ((d1!=idim1) && (d1!=idim2))
            area_cell *= cellSize[d1];
      }
      if (area_cell<1e-20){
         Log() << kWARNING << "PDEFoam::Project2: Warning, cell volume too small --> skiping cell!" << Endl;
         return 0;
      }

      // calc discriminator * (cell area times foam area)
      // foam is normalized -> length of foam = 1.0
      val = GetCellValue(cell, kDiscriminator)*area_cell;
   }
   // =========================================================
   else if (cv == kDiscriminatorError){
      // not testet jet!
      val = GetCellValue(cell, kDiscriminator);
   }
   // =========================================================
   else if (cv == kTarget0){
      // plot mean over all underlying cells?
      val = GetCellValue(cell, kTarget0);
   }
   else {
      Log() << kFATAL << "Project2: unknown option" << Endl;
      return 0;
   }

   return val;
}

//_____________________________________________________________________
TVectorD* TMVA::PDEFoam::GetCellElements( std::vector<Float_t> xvec )
{
   // Returns pointer to cell elements.  The given event vector 'xvec'
   // must be untransformed (i.e. [xmin, xmax]).

   assert(unsigned(GetTotDim()) == xvec.size());

   return dynamic_cast<TVectorD*>(FindCell(VarTransform(xvec))->GetElement());
}

//_____________________________________________________________________
Double_t TMVA::PDEFoam::GetCellElement( PDEFoamCell *cell, UInt_t i )
{
   // Returns cell element i of cell 'cell'.

   assert(i < GetNElements());

   return (*dynamic_cast<TVectorD*>(cell->GetElement()))(i);
}

//_____________________________________________________________________
void TMVA::PDEFoam::SetCellElement( PDEFoamCell *cell, UInt_t i, Double_t value )
{
   // Set cell element i of cell to value.

   if (i >= GetNElements()){
      Log() << kFATAL << "ERROR: Index out of range" << Endl;
      return;
   }

   TVectorD *vec = dynamic_cast<TVectorD*>(cell->GetElement());

   if (!vec)
      Log() << kFATAL << "<SetCellElement> ERROR: cell element is not a TVectorD*" << Endl;

   (*vec)(i) = value;
}

//_____________________________________________________________________
void TMVA::PDEFoam::OutputGrow( Bool_t finished )
{
   // Overridden function of PDEFoam to avoid native foam output.
   // Draw TMVA-process bar instead.

   if (finished) {
      Log() << kINFO << "Elapsed time: " + fTimer->GetElapsedTime()
            << "                                 " << Endl;
      return;
   }
   
   Int_t modulo = 1;

   if (fNCells        >= 100) modulo = Int_t(fNCells/100);
   if (fLastCe%modulo == 0)   fTimer->DrawProgressBar( fLastCe );
}

//_____________________________________________________________________
void TMVA::PDEFoam::RootPlot2dim( const TString& filename, std::string what,
                                  Bool_t CreateCanvas, Bool_t colors, Bool_t log_colors )
{
   // Debugging tool which plots 2-dimensional cells as rectangles
   // in C++ format readable for root.
   //
   // Parameters:
   // - filename - filename of ouput root macro
   // - CreateCanvas - whether to create a new canvas or not

   if (GetTotDim() != 2)
      Log() << kFATAL << "RootPlot2dim() can only be used with "
            << "two-dimensional foams!" << Endl;

   // select value to plot
   ECellValue cell_value = kNev;
   Bool_t plotcellnumber = kFALSE;
   Bool_t fillcells      = kTRUE;

   if (what == "mean")
      cell_value = kMeanValue;
   else if (what == "nevents")
      cell_value = kNev;
   else if (what == "density")
      cell_value = kDensity;
   else if (what == "rms")
      cell_value = kRms;
   else if (what == "rms_ov_mean")
      cell_value = kRmsOvMean;
   else if (what == "discr")
      cell_value = kDiscriminator;
   else if (what == "discrerr")
      cell_value = kDiscriminatorError;
   else if (what == "monotarget")
      cell_value = kTarget0;
   else if (what == "cellnumber")
      plotcellnumber = kTRUE;
   else if (what == "nofill") {
      plotcellnumber = kTRUE;
      fillcells = kFALSE;
   } else {
      cell_value = kMeanValue;
      Log() << kWARNING << "Unknown option, plotting mean!" << Endl;
   }

   std::ofstream outfile(filename, std::ios::out);
   Double_t x1,y1,x2,y2,x,y;
   Long_t   iCell;
   Double_t offs =0.01;
   Double_t lpag   =1-2*offs;

   outfile<<"{" << std::endl;

   if (!colors) { // define grayscale colors from light to dark,
      // starting from color index 1000
      outfile << "TColor *graycolors[100];" << std::endl;
      outfile << "for (Int_t i=0.; i<100; i++)" << std::endl;
      outfile << "  graycolors[i]=new TColor(1000+i, 1-(Float_t)i/100.,1-(Float_t)i/100.,1-(Float_t)i/100.);"<< std::endl;
   }
   if (CreateCanvas)
      outfile << "cMap = new TCanvas(\"" << fName << "\",\"Cell Map for "
              << fName << "\",600,600);" << std::endl;

   outfile<<"TBox*a=new TBox();"<<std::endl;
   outfile<<"a->SetFillStyle(0);"<<std::endl;  // big frame
   outfile<<"a->SetLineWidth(4);"<<std::endl;
   outfile<<"TBox *b1=new TBox();"<<std::endl;  // single cell
   if (fillcells) {
      outfile << (colors ? "gStyle->SetPalette(1, 0);" : "gStyle->SetPalette(0);") 
              << std::endl;
      outfile <<"b1->SetFillStyle(1001);"<<std::endl;
      outfile<<"TBox *b2=new TBox();"<<std::endl;  // single cell
      outfile <<"b2->SetFillStyle(0);"<<std::endl;
   }
   else {
      outfile <<"b1->SetFillStyle(0);"<<std::endl;
   }

   Int_t lastcell = fLastCe;

   if (fillcells)
      (colors ? gStyle->SetPalette(1, 0) : gStyle->SetPalette(0) );

   Double_t zmin = 1E8;  // minimal value (for color calculation)
   Double_t zmax = -1E8; // maximal value (for color calculation)

   Double_t value=0.;
   for (iCell=1; iCell<=lastcell; iCell++) {
      if ( fCells[iCell]->GetStat() == 1) {
         if (plotcellnumber)
            value = iCell;
         else
            value = GetCellValue(fCells[iCell], cell_value);
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

   Int_t ncolors  = colors ? gStyle->GetNumberOfColors() : 100;
   Double_t dz    = zmax - zmin;
   Double_t scale = (ncolors-1)/dz;

   PDEFoamVect  cellPosi(GetTotDim()); PDEFoamVect  cellSize(GetTotDim());
   outfile << "// =========== Rectangular cells  ==========="<< std::endl;
   for (iCell=1; iCell<=lastcell; iCell++) {
      if ( fCells[iCell]->GetStat() == 1) {
         fCells[iCell]->GetHcub(cellPosi,cellSize);
         x1 = offs+lpag*(cellPosi[0]);             
         y1 = offs+lpag*(cellPosi[1]);
         x2 = offs+lpag*(cellPosi[0]+cellSize[0]); 
         y2 = offs+lpag*(cellPosi[1]+cellSize[1]);
         
         value = 0;
         if (fillcells) {
            if (plotcellnumber) 
               value = iCell;
            else 
               value = GetCellValue(fCells[iCell], cell_value);

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
            x = offs+lpag*(cellPosi[0]+0.5*cellSize[0]); 
            y = offs+lpag*(cellPosi[1]+0.5*cellSize[1]);
         }
      }
   }
   outfile<<"// ============== End Rectangles ==========="<< std::endl;

   outfile << "}" << std::endl;
   outfile.flush();
   outfile.close();
}

//_____________________________________________________________________
void TMVA::PDEFoam::SetVolumeFraction( Double_t vfr )
{
   // set VolFrac to internal foam density PDEFoamDistr
   fDistr->SetVolumeFraction(vfr);
   SetPDEFoamVolumeFraction(vfr);
}

//_____________________________________________________________________
void TMVA::PDEFoam::FillBinarySearchTree( const Event* ev, Bool_t NoNegWeights )
{
   // Insert event to internal foam density PDEFoamDistr.
   fDistr->FillBinarySearchTree(ev, GetFoamType(), NoNegWeights);
}

//_____________________________________________________________________
void TMVA::PDEFoam::Init()
{
   // Initialize internal foam density PDEFoamDistr
   fDistr->Initialize(GetTotDim());
}

//_____________________________________________________________________
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

   fFoamType = ft; // set foam type class variable
}

//_____________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::PDEFoam& pdefoam )
{
   // Write PDEFoam variables to stream 'os'.
   pdefoam.PrintStream(os);
   return os; // Return the output stream.
}

//_____________________________________________________________________
istream& TMVA::operator>> ( istream& istr, TMVA::PDEFoam& pdefoam )
{
   // Read PDEFoam variables from stream 'istr'.
   pdefoam.ReadStream(istr);
   return istr;
}

//_____________________________________________________________________
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

//_____________________________________________________________________
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

//_____________________________________________________________________
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

//_____________________________________________________________________
void TMVA::PDEFoam::ReadXML( void* parent ) {
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
