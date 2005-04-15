// @(#)root/foam:$Name:  $:$Id: TFoam.cxx,v 1.8 2005/04/13 13:15:27 brun Exp $
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

//______________________________________________________________________________
//
// FOAM  Version 1.02M
// ===================
// Authors:
//   S. Jadach and P.Sawicki
//   Institute of Nuclear Physics, Cracow, Poland
//   Stanislaw. Jadach@ifj.edu.pl, Pawel.Sawicki@ifj.edu.pl
//
// What is FOAM for?
// =================
// * Suppose you want to generate randomly points (vectors) according to
//   an arbitrary probability distribution  in n dimensions,
//   for which you supply your own subprogram. FOAM can do it for you!
//   Even if your distributions has quite strong peaks and is discontinuous!
// * FOAM generates random points with weight one or with variable weight.
// * FOAM is capable to integrate using efficient "adaptive" MC method.
//   (The distribution does not need to be normalized to one.)
// How does it work?
// =================
// FOAM is the simplified version of the multi-dimensional general purpose
// Monte Carlo event generator (integrator) FOAM.
// It creates hyper-rectangular "foam of cells", which is more dense around its peaks.
// See the following 2-dim. example of the map of 1000 cells for doubly peaked distribution:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_MapCamel1000.gif">
<!--*/
// -->END_HTML
// FOAM is now fully integrated with the ROOT package.
// The important bonus of the ROOT use is persistency of the FOAM objects!
//
// For more sophisticated problems full version of FOAM may be more appropriate:
//BEGIN_HTML <!--
/* -->
  See <A HREF="http://jadach.home.cern.ch/jadach/Foam/Index.html"> full version of FOAM</A>
<!--*/
// -->END_HTML
// Simple example of the use of FOAM:
// ==================================
// Int_t kanwa(){
//   gSystem->Load("libFoam.so");
//   TH2D  *hst_xy = new TH2D("hst_xy" ,  "x-y plot", 50,0,1.0, 50,0,1.0);
//   Double_t *MCvect =new Double_t[2]; // 2-dim vector generated in the MC run
//   TRandom3  *PseRan   = new TRandom3();  // Create random number generator
//   PseRan->SetSeed(4357);                // Set seed
//   TFoam   *FoamX    = new TFoam("FoamX");   // Create Simulator
//   FoamX->SetkDim(2);          // No. of dimensions, obligatory!
//   FoamX->SetnCells(500);      // No. of cells, can be omitted, default=2000
//   FoamX->SetRhoInt(Camel2);   // Set 2-dim distribution, included below
//   FoamX->SetPseRan(PseRan);   // Set random number generator
//   FoamX->Initialize();        // Initialize simulator, takes a few seconds...
//   // From now on FoamX is ready to generate events according to Camel2(x,y)
//   for(Long_t loop=0; loop<100000; loop++){
//     FoamX->MakeEvent();          // generate MC event
//     FoamX->GetMCvect( MCvect);   // get generated vector (x,y)
//     Double_t x=MCvect[0];
//     Double_t y=MCvect[1];
//     if(loop<10) cout<<"(x,y) =  ( "<< x <<", "<< y <<" )"<<endl;
//     hst_xy->Fill(x,y);           // fill scattergram
//   }// loop
//   Double_t MCresult, MCerror;
//   FoamX->GetIntegMC( MCresult, MCerror);  // get MC integral, should be one
//   cout << " MCresult= " << MCresult << " +- " << MCerror <<endl;
//   // now hst_xy will be plotted visualizing generated distribution
//   TCanvas *cKanwa = new TCanvas("cKanwa","Canvas for plotting",600,600);
//   cKanwa->cd();
//   hst_xy->Draw("lego2");
// }//kanwa
// Double_t sqr(Double_t x){return x*x;};
// Double_t Camel2(Int_t nDim, Double_t *Xarg){
// // 2-dimensional distribution for FOAM, normalized to one (within 1e-5)
//   Double_t x=Xarg[0];
//   Double_t y=Xarg[1];
//   Double_t GamSq= sqr(0.100e0);
//   Double_t Dist=exp(-(sqr(x-1./3) +sqr(y-1./3))/GamSq)/GamSq/TMath::Pi();
//   Dist        +=exp(-(sqr(x-2./3) +sqr(y-2./3))/GamSq)/GamSq/TMath::Pi();
//   return 0.5*Dist;
// }// Camel2
// Two-dim. histogram of the MC points generated with the above program looks as follows:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_cKanwa.gif">
<!--*/
// -->END_HTML
// Canonical nine steering parameters of FOAM
// ===========================================
//------------------------------------------------------------------------------
//  Name     | default  | Description
//------------------------------------------------------------------------------
//  Dim      | 0        | Dimension of the integration space. Must be redefined!
//  nCells   | 1000     | No of allocated number of cells,
//  nSampl   | 200      | No. of MC events in the cell MC exploration
//  nBin     | 8        | No. of bins in edge-histogram in cell exploration
//  OptRej   | 1        | OptRej = 0, weighted; OptRej=1, wt=1 MC events
//  OptDrive | 2        | Maximum weight reduction, =1 for variance reduction
//  EvPerBin | 25       | Maximum number of the effective wt=1 events/bin,
//           |          | EvPerBin=0 deactivates this option
//  Chat     | 1        | =0,1,2 is the ``chat level'' in the standard output
//  MaxWtRej | 1.1      | Maximum weight used to get w=1 MC events
//------------------------------------------------------------------------------
// The above can be redefined before calling 'Initialize()' method,
// for instance FoamObject->SetkDim(15) sets dimension of the distribution to 15.
// Only kDim HAS TO BE redefined, the other parameters may be left at their defaults.
// nCell may be increased up to about million cells for wildly peaked distributions.
// Increasing nSampl sometimes helps, but it may cost CPU time.
// MaxWtRej may need to be increased for wild a distribution, while using OptRej=0.
//
// --------------------------------------------------------------------
// Past versions of FOAM: August 2003, v.1.00; September 2003 v.1.01
// Adopted starting from FOAM-2.06 by P. Sawicki
// --------------------------------------------------------------------
// Users of FOAM are kindly requested to cite the following work:
// S. Jadach, Computer Physics Communications 152 (2003) 55.
//
//______________________________________________________________________________

#include "TFoam.h"
#include "TFoamIntegrand.h"
#include "TFoamMaxwt.h"
#include "TFoamVect.h"
#include "TFoamCell.h"
#include "Riostream.h"
#include "TH1.h"
#include "TRefArray.h"
#include "TMethodCall.h"
#include "TRandom.h"
#include "TMath.h"

ClassImp(TFoam);

//FFFFFF  BoX-FORMATs for nice and flexible outputs
#define BXOPE cout<<\
"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"<<endl<<\
"F                                                                              F"<<endl
#define BXTXT(text) cout<<\
"F                   "<<setw(40)<<         text           <<"                   F"<<endl
#define BX1I(name,numb,text) cout<<\
"F "<<setw(10)<<name<<" = "<<setw(10)<<numb<<" = "          <<setw(50)<<text<<" F"<<endl
#define BX1F(name,numb,text)     cout<<"F "<<setw(10)<<name<<\
          " = "<<setw(15)<<setprecision(8)<<numb<<"   =    "<<setw(40)<<text<<" F"<<endl
#define BX2F(name,numb,err,text) cout<<"F "<<setw(10)<<name<<\
" = "<<setw(15)<<setprecision(8)<<numb<<" +- "<<setw(15)<<setprecision(8)<<err<<\
                                                      "  = "<<setw(25)<<text<<" F"<<endl
#define BXCLO cout<<\
"F                                                                              F"<<endl<<\
"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"<<endl
  //FFFFFF  BoX-FORMATs ends here

static const Double_t kHigh= 1.0e150;
static const Double_t kVlow=-1.0e150;

#define SW2 setprecision(7) << setw(12)

//________________________________________________________________________________________________
TFoam::TFoam()
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

}
//_________________________________________________________________________________________________
TFoam::TFoam(const Char_t* Name)
{
// User constructor, to be employed by the user

  if(strlen(Name)  >129) {
    Error("TFoam","Name too long %s \n",Name);
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
  fWtMin = kHigh;               // Minimal weight
  fWtMax = kVlow;               // Maximal weight
  fMaxWtRej =1.10;              // Maximum weight in rejection for getting wt=1 events
  fPseRan   = 0;                // Initialize private copy of random number generator
  fMCMonit  = 0;                // MC efficiency monitoring
  fRho = 0;                     // pointer to abstract class providing function to integrate
  fMethodCall=0;                // ROOT's pointer to global distribution function
}

//_______________________________________________________________________________________________
TFoam::~TFoam()
{
// Default destructor
//  cout<<" DESTRUCTOR entered "<<endl;
  Int_t i;

  if(fCells!= 0) {
    for(i=0; i<fNCells; i++) delete fCells[i]; // TFoamCell*[]
    delete [] fCells;
  }
  delete [] fRvec;    //double[]
  delete [] fAlpha;   //double[]
  delete [] fMCvect;  //double[]
  delete [] fPrimAcu; //double[]
  delete [] fMaskDiv; //int[]
  delete [] fInhiDiv; //int[]
  //
  //
  if( fXdivPRD!= 0) {
    for(i=0; i<fDim; i++) delete fXdivPRD[i]; // TFoamVect*[]
    delete [] fXdivPRD;
  }
  delete fMCMonit;
  delete fHistWt;
  //
}


//_____________________________________________________________________________________________
TFoam::TFoam(const TFoam &From): TObject(From)
{
// Copy Constructor  NOT IMPLEMENTED (NEVER USED)
  Error("TFoam", "COPY CONSTRUCTOR NOT IMPLEMENTED \n");
}

//_____________________________________________________________________________________________
void TFoam::Initialize(TRandom *PseRan, TFoamIntegrand *fun )
{
// Basic initialization of FOAM invoked by the user. Mandatory!
// ============================================================
// This method starts the process of the cell build-up.
// User must invoke Initialize with two arguments or Initialize without arguments.
// This is done BEFORE generating first MC event and AFTER allocating FOAM object
// and reseting (optionally) its internal parameters/switches.
// The overall oprerational scheme of the FOAM is the following:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_schema2.gif">
<!--*/
// -->END_HTML
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
// The above sequence of the procedure calls is depicted in the following figure:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_Initialize_schema.gif">
<!--*/
// -->END_HTML

 SetPseRan(PseRan);
 SetRho(fun);
 Initialize();

}

//_______________________________________________________________________________________
void TFoam::Initialize()
{
// Basic initialization of FOAM invoked by the user.
// IMPORTANT: Random number generator and the distribution object has to be
// provided using SetPseRan and SetRho prior to invoking this initializator!

  Int_t i;

  if(fChat>0){
    BXOPE;
    BXTXT("****************************************");
    BXTXT("******      TFoam::Initialize    ******");
    BXTXT("****************************************");
    BXTXT(fName);
    BX1F("  Version",fVersion,  fDate);
    BX1I("     kDim",fDim,     " Dimension of the hyper-cubical space             ");
    BX1I("   nCells",fNCells,   " Requested number of Cells (half of them active)  ");
    BX1I("   nSampl",fNSampl,   " No of MC events in exploration of a cell         ");
    BX1I("     nBin",fNBin,     " No of bins in histograms, MC exploration of cell ");
    BX1I(" EvPerBin",fEvPerBin, " Maximum No effective_events/bin, MC exploration  ");
    BX1I(" OptDrive",fOptDrive, " Type of Driver   =1,2 for Sigma,WtMax            ");
    BX1I("   OptRej",fOptRej,   " MC rejection on/off for OptRej=0,1               ");
    BX1F(" MaxWtRej",fMaxWtRej, " Maximum wt in rejection for wt=1 evts");
    BXCLO;
  }

  if(fPseRan==0) Error("Initialize", "Random number generator not set \n");
  if(fRho==0 && fMethodCall==0 ) Error("Initialize", "Distribution function not set \n");
  if(fDim==0) Error("Initialize", "Zero dimension not alowed \n");

  /////////////////////////////////////////////////////////////////////////
  //                   ALLOCATE SMALL LISTS                              //
  //  it is done globally, not for each cell, to save on allocation time //
  /////////////////////////////////////////////////////////////////////////
  fRNmax= fDim+1;
  fRvec = new Double_t[fRNmax];   // Vector of random numbers
  if(fRvec==0)  Error("Initialize", "Cannot initialize buffer fRvec \n");

  if(fDim>0){
    fAlpha = new Double_t[fDim];    // sum<1 for internal parametrization of the simplex
    if(fAlpha==0)  Error("Initialize", "Cannot initialize buffer fAlpha \n" );
  }
  fMCvect = new Double_t[fDim]; // vector generated in the MC run
  if(fMCvect==0)  Error("Initialize", "Cannot initialize buffer fMCvect  \n" );

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
    fXdivPRD = new TFoamVect*[fDim];
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
    //cout<<"i= "<<i<<"  hname= "<<hname<<"  htitle= "<<htitle<<endl;
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
  InitCells();
  //        PrintCells(); cout<<" ===== after InitCells ====="<<endl;
  Grow();
  //        PrintCells(); cout<<" ===== after Grow      ====="<<endl;

  MakeActiveList(); // Final Preperations for the M.C. generation

  // Preperations for the M.C. generation
  fSumWt  = 0.0;               // M.C. generation sum of Wt
  fSumWt2 = 0.0;               // M.C. generation sum of Wt**2
  fSumOve = 0.0;               // M.C. generation sum of overweighed
  fNevGen = 0.0;               // M.C. generation sum of 1d0
  fWtMax  = kVlow;               // M.C. generation maximum wt
  fWtMin  = kHigh;               // M.C. generation minimum wt
  fMCresult=fCells[0]->GetIntg(); // M.C. Value of INTEGRAL,temporary assignment
  fMCresult=fCells[0]->GetIntg(); // M.C. Value of INTEGRAL,temporary assignment
  fMCerror =fCells[0]->GetIntg(); // M.C. Value of ERROR   ,temporary assignment
  fMCMonit = new TFoamMaxwt(5.0,1000);  // monitoring M.C. efficiency
  //
  if(fChat>0){
    Double_t Driver = fCells[0]->GetDriv();
    BXOPE;
    BXTXT("***  TFoam::Initialize FINISHED!!!  ***");
    BX1I("    nCalls",fNCalls,  "Total number of function calls         ");
    BX1F("    XPrime",fPrime,   "Primary total integral                 ");
    BX1F("    XDiver",Driver,    "Driver  total integral                 ");
    BX1F("  MCresult",fMCresult,"Estimate of the true MC Integral       ");
    BXCLO;
  }
  if(fChat==2) PrintCells();
} // Initialize

//_______________________________________________________________________________________
void TFoam::InitCells()
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
  fCells = new TFoamCell*[fNCells];
  for(i=0;i<fNCells;i++){
    fCells[i]= new TFoamCell(fDim); // Allocate BIG list of cells
    fCells[i]->SetSerial(i);
  }
  if(fCells==0) Error("InitCells", "Cannot initialize CELLS \n"  );

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
Int_t TFoam::CellFill(Int_t Status, TFoamCell *Parent)
{
// Internal subprogram used by Initialize.
// It initializes content of the newly allocated active cell.

  TFoamCell *Cell;
  if (fLastCe==fNCells){
    Error( "CellFill", "Too many cells\n");
  }
  fLastCe++;   // 0-th cell is the first
  if (Status==1) fNoAct++;

  Cell = fCells[fLastCe];

  Cell->Fill(Status, Parent, 0, 0);

  Cell->SetBest( -1);         // pointer for planning division of the cell
  Cell->SetXdiv(0.5);         // factor for division
  Double_t xInt2,xDri2;
  if(Parent!=0){
    xInt2  = 0.5*Parent->GetIntg();
    xDri2  = 0.5*Parent->GetDriv();
    Cell->SetIntg(xInt2);
    Cell->SetDriv(xDri2);
  }else{
    Cell->SetIntg(0.0);
    Cell->SetDriv(0.0);
  }
  return fLastCe;
}

//______________________________________________________________________________________
void TFoam::Explore(TFoamCell *Cell)
{
// Internal subprogram used by Initialize.
// It explores newly defined cell with help of special short MC sampling.
// As a result, estimates of true and drive volume is defined/determined
// Average and dispersion of the weight distribution will is found along
// each edge and the best edge (minimum dispersion, best maximum weight)
// is memorized for future use.
// The optimal division point for eventual future cell division is
// determined/recorded. Recorded are also minimum and maximum weight etc.
// The volume estimate in all (inactive) parent cells is updated.
// Note that links to parents and initial volume = 1/2 parent has to be
// already defined prior to calling this routine.

  Double_t Wt, Dx, xBest, yBest;
  Double_t IntOld, DriOld;

  Long_t iev;
  Double_t NevMC;
  Int_t i, j, k;
  Int_t nProj, kBest;
  Double_t CeSum[5], Xproj;

  TFoamVect  Size(fDim);
  TFoamVect  Posi(fDim);

  Cell->GetHcub(Posi,Size);

  TFoamCell  *Parent;

  Double_t *xRand = new Double_t[fDim];

  Double_t *VolPart=0;

  Cell->CalcVolume();
  Dx = Cell->GetVolume();
  IntOld = Cell->GetIntg(); //memorize old values,
  DriOld = Cell->GetDriv(); //will be needed for correcting parent cells


  /////////////////////////////////////////////////////
  //    Special Short MC sampling to probe cell      //
  /////////////////////////////////////////////////////
  CeSum[0]=0;
  CeSum[1]=0;
  CeSum[2]=0;
  CeSum[3]=kHigh;  //wtmin
  CeSum[4]=kVlow;  //wtmax
  //
  for(i=0;i<fDim;i++) ((TH1D *)(*fHistEdg)[i])->Reset(); // Reset histograms
  fHistWt->Reset();
  //
  // ||||||||||||||||||||||||||BEGIN MC LOOP|||||||||||||||||||||||||||||
  Double_t NevEff=0.;
  for(iev=0;iev<fNSampl;iev++){
    MakeAlpha();               // generate uniformly vector inside hypercube

    if(fDim>0){
      for(j=0; j<fDim; j++)
        xRand[j]= Posi[j] +fAlpha[j]*(Size[j]);
    }

    Wt=Dx*Eval(xRand);

    nProj = 0;
    if(fDim>0){
      for(k=0; k<fDim; k++){
        Xproj =fAlpha[k];
        ((TH1D *)(*fHistEdg)[nProj])->Fill(Xproj,Wt);
        nProj++;
      }
    }
    //
    fNCalls++;
    CeSum[0] += Wt;    // sum of weights
    CeSum[1] += Wt*Wt; // sum of weights squared
    CeSum[2]++;        // sum of 1
    if (CeSum[3]>Wt) CeSum[3]=Wt;  // minimum weight;
    if (CeSum[4]<Wt) CeSum[4]=Wt;  // maximum weight
    // test MC loop exit condition
    NevEff = CeSum[0]*CeSum[0]/CeSum[1];
    if( NevEff >= fNBin*fEvPerBin) break;
  }   // ||||||||||||||||||||||||||END MC LOOP|||||||||||||||||||||||||||||
  //------------------------------------------------------------------
  //---  predefine logics of searching for the best division edge ---
  for(k=0; k<fDim;k++){
    fMaskDiv[k] =1;                       // default is all
    if( fInhiDiv[k]==1) fMaskDiv[k] =0; // inhibit some...
  }
  // Note that predefined division below overrule inhibition above
  kBest=-1;
  Double_t rmin,rmax,rdiv;
  if(fOptPRD){          // quick check
    for(k=0; k<fDim; k++){
      rmin= Posi[k];
      rmax= Posi[k] +Size[k];
      if( fXdivPRD[k] != 0){
        Int_t N= (fXdivPRD[k])->GetDim();
        for(j=0; j<N; j++){
          rdiv=(*fXdivPRD[k])[j];
          // check predefined divisions is available in this cell
          if( (rmin +1e-99 <rdiv) && (rdiv< rmax -1e-99)){
            kBest=k;
            xBest= (rdiv-Posi[k])/Size[k] ;
            goto ee05;
          }
        }
      }
    }//k
  }
  ee05:
  //------------------------------------------------------------------
  fNEffev += (Long_t)NevEff;
  NevMC          = CeSum[2];
  Double_t IntTrue = CeSum[0]/(NevMC+0.000001);
  Double_t IntDriv=0.;
  Double_t IntPrim=0.;

  switch(fOptDrive){
  case 1:                       // VARIANCE REDUCTION
    if(kBest == -1) Varedu(CeSum,kBest,xBest,yBest); // determine the best edge,
    //IntDriv =sqrt( CeSum[1]/NevMC -IntTrue*IntTrue ); // Older ansatz, numericaly not bad
    IntDriv =sqrt(CeSum[1]/NevMC) -IntTrue; // Foam build-up, sqrt(<w**2>) -<w>
    IntPrim =sqrt(CeSum[1]/NevMC);          // MC gen. sqrt(<w**2>) =sqrt(<w>**2 +sigma**2)
    break;
  case 2:                       // WTMAX  REDUCTION
    if(kBest == -1) Carver(kBest,xBest,yBest);  // determine the best edge
    IntDriv =CeSum[4] -IntTrue; // Foam build-up, wtmax-<w>
    IntPrim =CeSum[4];          // MC generation, wtmax!
    break;
  default:
    Error("Explore", "Wrong fOptDrive = \n" );
  }//switch
  //=================================================================================
  //hist_Neff_distrib.Fill( fLastCe/2.0+0.01, NevEff+0.01);  //
  //hist_kBest_distrib.Fill( kBest+0.50, 1.0 ); //  debug
  //hist_xBest_distrib.Fill( xBest+0.01, 1.0 ); //  debug
  //=================================================================================
  Cell->SetBest(kBest);
  Cell->SetXdiv(xBest);
  Cell->SetIntg(IntTrue);
  Cell->SetDriv(IntDriv);
  Cell->SetPrim(IntPrim);
  // correct/update integrals in all parent cells to the top of the tree
  Double_t  ParIntg, ParDriv;
  for(Parent = Cell->GetPare(); Parent!=0; Parent = Parent->GetPare()){
    ParIntg = Parent->GetIntg();
    ParDriv = Parent->GetDriv();
    Parent->SetIntg( ParIntg   +IntTrue -IntOld );
    Parent->SetDriv( ParDriv   +IntDriv -DriOld );
  }
  delete [] VolPart;
  delete [] xRand;
  //Cell->Print();
} // TFoam::Explore

//______________________________________________________________________________________
void TFoam::Varedu(Double_t CeSum[5], Int_t &kBest, Double_t &xBest, Double_t &yBest)
{
// Internal subrogram used by Initialize.
// In determines the best edge candidate and the position of the cell division plane
// in case of the variance reduction for future cell division,
// using results of the MC exploration run stored in fHistEdg

  Double_t Nent   = CeSum[2];
  Double_t swAll  = CeSum[0];
  Double_t sswAll = CeSum[1];
  Double_t SSw    = sqrt(sswAll)/sqrt(Nent);
  //
  Double_t SwIn,SwOut,SSwIn,SSwOut,xLo,xUp;
  kBest =-1;
  xBest =0.5;
  yBest =1.0;
  Double_t MaxGain=0.0;
  // Now go over all projections kProj
  for(Int_t kProj=0; kProj<fDim; kProj++)
    if( fMaskDiv[kProj]){
    // initialize search over bins
    Double_t SigmIn =0.0; Double_t SigmOut =0.0;
    Double_t SSwtBest = kHigh;
    Double_t Gain =0.0;
    Double_t xMin=0.0; Double_t xMax=0.0;
    // Double loop over all pairs jLo<jUp
    for(Int_t jLo=1; jLo<=fNBin; jLo++){
      Double_t swIn=0;  Double_t sswIn=0;
      for(Int_t jUp=jLo; jUp<=fNBin;jUp++){
        swIn  +=     ((TH1D *)(*fHistEdg)[kProj])->GetBinContent(jUp);
        sswIn += Sqr(((TH1D *)(*fHistEdg)[kProj])->GetBinError(  jUp));
        xLo=(jLo-1.0)/fNBin;
        xUp=(jUp*1.0)/fNBin;
        SwIn  =        swIn/Nent;
        SwOut = (swAll-swIn)/Nent;
        SSwIn = sqrt(sswIn)       /sqrt(Nent*(xUp-xLo))     *(xUp-xLo);
        SSwOut= sqrt(sswAll-sswIn)/sqrt(Nent*(1.0-xUp+xLo)) *(1.0-xUp+xLo);
        if( (SSwIn+SSwOut) < SSwtBest){
          SSwtBest = SSwIn+SSwOut;
          Gain     = SSw-SSwtBest;
          SigmIn   = SSwIn -SwIn;  // Debug
          SigmOut  = SSwOut-SwOut; // Debug
          xMin    = xLo;
          xMax    = xUp;
        }
      }//jUp
    }//jLo
    Int_t iLo = (Int_t) (fNBin*xMin);
    Int_t iUp = (Int_t) (fNBin*xMax);
    //----------DEBUG printout
    //cout<<"@@@@@  xMin xMax = "<<xMin   <<" "<<xMax<<"  iLo= "<<iLo<<"  iUp= "<<iUp;
    //cout<<"  SSwtBest/SSw= "<<SSwtBest/SSw<<"  Gain/SSw= "<< Gain/SSw<<endl;
    //----------DEBUG auxilary Plot
      for(Int_t iBin=1;iBin<=fNBin;iBin++)
        if( ((iBin-0.5)/fNBin > xMin) && ((iBin-0.5)/fNBin < xMax) ){
          ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin,SigmIn/(xMax-xMin));
        }else{
          ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin,SigmOut/(1-xMax+xMin));
        }

    if(Gain>=MaxGain){
      MaxGain=Gain;
      kBest=kProj; // <--- !!!!! The best edge
      xBest=xMin;
      yBest=xMax;
      if(iLo == 0   ) xBest=yBest; // The best division point
      if(iUp == fNBin) yBest=xBest; // this is not really used
    }
  }
  //----------DEBUG printout
  //cout<<"@@@@@@@>>>>> kBest= "<<kBest<<"  MaxGain/SSw= "<< MaxGain/SSw<<endl;
  if( (kBest >= fDim) || (kBest<0) ) Error("Varedu", "Something wrong with kBest \n" );
}          //TFoam::Varedu

//________________________________________________________________________________________
void TFoam::Carver(Int_t &kBest, Double_t &xBest, Double_t &yBest)
{
// Internal subrogram used by Initialize.
// Determines the best edge-candidate and the position of the division plane
// for the future cell division, in the case of the optimization of the maximum weight.
// It exploits results of the cell MC exploration run stored in fHistEdg.

  Int_t    kProj,iBin;
  Double_t Carve,CarvTot,CarvMax,CarvOne,BinMax,BinTot,PrimTot,PrimMax;
  Int_t    jLow,jUp,iLow,iUp;
  Double_t TheBin;
  Int_t    jDivi; // TEST

  Double_t *Bins  = new Double_t[fNBin];      // bins of histogram for single  PROJECTION
  if(Bins==0)    Error("Carver", "Cannot initialize buffer Bins \n" );

  kBest =-1;
  xBest =0.5;
  yBest =1.0;
  CarvMax = kVlow;
  PrimMax = kVlow;
  for(kProj=0; kProj<fDim; kProj++)
    if( fMaskDiv[kProj] ){
    //if( kProj==1 ){
    //cout<<"==================== Carver histogram: kProj ="<<kProj<<"==================="<<endl;
    //((TH1D *)(*fHistEdg)[kProj])->Print("all");
    BinMax = kVlow;
    for(iBin=0; iBin<fNBin;iBin++){
      Bins[iBin]= ((TH1D *)(*fHistEdg)[kProj])->GetBinContent(iBin+1);
      BinMax = TMath::Max( BinMax, Bins[iBin]);       // Maximum content/bin
    }
    if(BinMax < 0 ) {       //case of empty cell
      delete [] Bins;
      return;
    }
    CarvTot = 0.0;
    BinTot  = 0.0;
    for(iBin=0;iBin<fNBin;iBin++){
      CarvTot = CarvTot + (BinMax-Bins[iBin]);     // Total Carve (more stable)
      BinTot  +=Bins[iBin];
    }
    PrimTot = BinMax*fNBin;
     //cout <<"Carver:  CarvTot "<<CarvTot<< "    PrimTot "<<PrimTot<<endl;
    jLow =0;
    jUp  =fNBin-1;
    CarvOne = kVlow;
    Double_t Ylevel = kVlow;
    for(iBin=0; iBin<fNBin;iBin++){
      TheBin = Bins[iBin];
      //-----  walk to the left and find first bin > TheBin
      iLow = iBin;
      for(Int_t j=iBin; j>-1; j-- ) {
        if(TheBin< Bins[j]) break;
        iLow = j;
      }
      //iLow = iBin;
      //if(iLow>0)     while( (TheBin >= Bins[iLow-1])&&(iLow >0) ){iLow--;} // horror!!!
      //------ walk to the right and find first bin > TheBin
      iUp  = iBin;
      for(Int_t j=iBin; j<fNBin; j++){
        if(TheBin< Bins[j]) break;
        iUp = j;
      }
      //iUp  = iBin;
      //if(iUp<fNBin-1) while( (TheBin >= Bins[iUp+1])&&( iUp<fNBin-1 ) ){iUp++;} // horror!!!
      //
      Carve = (iUp-iLow+1)*(BinMax-TheBin);
      if( Carve > CarvOne){
        CarvOne = Carve;
        jLow = iLow;
        jUp  = iUp;
        Ylevel = TheBin;
      }
    }//iBin
    if( CarvTot > CarvMax){
      CarvMax   = CarvTot;
      PrimMax   = PrimTot;
      //cout <<"Carver:   PrimMax "<<PrimMax<<endl;
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
    //cout<<"kProj= "<<kProj<<" jLow= "<<jLow<<" jUp= "<<jUp<<endl;
      for(iBin=0;    iBin<fNBin;  iBin++)
        ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin+1,BinMax);
      for(iBin=jLow; iBin<jUp+1;   iBin++)
        ((TH1D *)(*fHistDbg)[kProj])->SetBinContent(iBin+1,Ylevel);
  }//kProj
  if( (kBest >= fDim) || (kBest<0) ) Error("Carver", "Something wrong with kBest \n" );
  delete [] Bins;
}          //TFoam::Carver

//______________________________________________________________________________________________
void TFoam::MakeAlpha()
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
void TFoam::Grow()
{
// Internal subrogram used by Initialize.
// It grow new cells by the binary division process.

  Long_t iCell;
  TFoamCell* newCell;

  while ( (fLastCe+2) < fNCells ){  // this condition also checked inside Divide
      iCell   = PeekMax();        // peek up cell with maximum driver integral
      if( (iCell<0) || (iCell>fLastCe) ) Error("Grow", "Wrong iCell \n");
      newCell = fCells[iCell];

    if(fLastCe !=0){
      Int_t kEcho=10;
      if(fLastCe>=10000) kEcho=100;
      if( (fLastCe%kEcho)==0){
        if(fDim<10)
          cout<<fDim<<flush;
        else
          cout<<"."<<flush;
        if( (fLastCe%(100*kEcho))==0)  cout<<"|"<<fLastCe<<endl<<flush;
      }
    }
    //
    if( Divide( newCell )==0) break;  // and divide it into two
  }
  cout<<endl<<flush;
  CheckAll(0);   // set arg=1 for more info
}// Grow

//_____________________________________________________________________________________________
Long_t  TFoam::PeekMax()
{
// Internal subprogram used by Initialize.
// It finds cell with maximal Driver integral for the purpose of the division.

   Long_t  i;
   Long_t iCell = -1;
   Double_t  DrivMax, Driv;

   DrivMax = kVlow;
   for(i=0; i<=fLastCe; i++) //without root
   {
      if( fCells[i]->GetStat() == 1 )
      {
         Driv =  TMath::Abs( fCells[i]->GetDriv());
         //cout<<"PeekMax: Driv = "<<Driv<<endl;
         if(Driv > DrivMax)
         {
            DrivMax = Driv;
            iCell = i;
         }
      }
   }
//  cout << 'TFoam_PeekMax: iCell=' << iCell << endl;
   if (iCell == -1)
      cout << "STOP in TFoam::PeekMax: not found iCell=" <<  iCell << endl;
   return(iCell);
}                 // TFoam_PeekMax

//_____________________________________________________________________________________________
Int_t TFoam::Divide(TFoamCell *Cell)
{
// Internal subrogram used by Initialize.
// It divides cell iCell into two daughter cells.
// The iCell is retained and tagged as inactive, daughter cells are appended
// at the end of the buffer.
// New vertex is added to list of vertices.
// List of active cells is updated, iCell removed, two daughters added
// and their properties set with help of MC sampling (TFoam_Explore)
// Returns Code RC=-1 of buffer limit is reached,  fLastCe=fnBuf.

  Double_t Xdiv;
  Int_t   kBest;

  if(fLastCe+1 >= fNCells) Error("Divide", "Buffer limit is reached, fLastCe=fnBuf \n");

  Cell->SetStat(0); // reset Cell as inactive
  fNoAct--;

  Xdiv  = Cell->GetXdiv();
  kBest = Cell->GetBest();
  if( kBest<0 || kBest>=fDim ) Error("Divide", "Wrong kBest \n");

  //////////////////////////////////////////////////////////////////
  //           define two daughter cells (active)                 //
  //////////////////////////////////////////////////////////////////

  Int_t d1 = CellFill(1,   Cell);
  Int_t d2 = CellFill(1,   Cell);
  Cell->SetDau0((fCells[d1]));
  Cell->SetDau1((fCells[d2]));
  Explore( (fCells[d1]) );
  Explore( (fCells[d2]) );
  return 1;
} // TFoam_Divide


//_________________________________________________________________________________________
void TFoam::MakeActiveList()
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
  for(iCell=0; iCell<=fLastCe; iCell++){
    if (fCells[iCell]->GetStat()==1){
      fPrime += fCells[iCell]->GetPrim();
      fCellsAct->Add(fCells[iCell]);
      n++;
    }
  }

  if(fNoAct != n)  Error("MakeActiveList", "Wrong fNoAct               \n"  );
  if(fPrime == 0.) Error("MakeActiveList", "Integrand function is zero  \n"  );

  fPrimAcu  = new  Double_t[fNoAct]; // cumulative primary for MC generation
  if( fCellsAct==0 || fPrimAcu==0 ) Error("MakeActiveList", "Cant allocate fCellsAct or fPrimAcu \n");

  sum =0.0;
  for(iCell=0; iCell<fNoAct; iCell++){
    sum = sum + ( (TFoamCell *) (fCellsAct->At(iCell)) )->GetPrim()/fPrime;
    fPrimAcu[iCell]=sum;
  }

} //MakeActiveList

//__________________________________________________________________________________________
void TFoam::ResetPseRan(TRandom *PseRan)
{
// User may optionally reset random number generator using this method
// Usually it is done when FOAM object is restored from the disk.
// IMPORTANT: this method deletes existing  random number generator registered in the FOAM object.
// In particular such an object is created by the streamer diring the disk-read operation.

   if(fPseRan) {
      Info("ResetPseRan", "Resetting random number generator  \n");
      delete fPseRan;
   }
   SetPseRan(PseRan);
}

//__________________________________________________________________________________________
void TFoam::SetRho(TFoamIntegrand *fun)
{
// User may use this method to set (register) random number generator used by
// the given instance of the FOAM event generator. Note that single r.n. generator
// may serve several FOAM objects.

   if (fun)
      fRho=fun;
   else
      Error("SetRho", "Bad function \n" );
}

//__________________________________________________________________________________________
void TFoam::ResetRho(TFoamIntegrand *fun)
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
void TFoam::SetRhoInt(void *fun)
{
// User may use this to set pointer to the global function (not descending
// from TFoamIntegrand) serving as a distribution for FOAM.
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

//__________________________________________________________________________________________
Double_t TFoam::Eval(Double_t *xRand)
{
// Internal subprogram.
// Evaluates distribution to be generated.

  Double_t result;

  if(!fRho)    //interactive mode
    {
    Long_t paramArr[3];
    paramArr[0]=(Long_t)fDim;
    paramArr[1]=(Long_t)xRand;
    fMethodCall->SetParamPtrs(paramArr);
    fMethodCall->Execute(result);
    }
  else {       //compiled mode
    result=fRho->Density(fDim,xRand);
  }

  return result;
}

//___________________________________________________________________________________________
void TFoam::GenerCel2(TFoamCell *&pCell)
{
// Internal subprogram.
// Return randomly chosen active cell with probability equal to its
// contribution into total driver integral using interpolation search.

  Long_t  lo, hi, hit;
  Double_t fhit, flo, fhi;
  Double_t Random;

  Random=fPseRan->Rndm();
  lo  = 0;              hi =fNoAct-1;
  flo = fPrimAcu[lo];  fhi=fPrimAcu[hi];
  while(lo+1<hi){
    hit = lo + (Int_t)( (hi-lo)*(Random-flo)/(fhi-flo)+0.5);
    if (hit<=lo)
      hit = lo+1;
    else if(hit>=hi)
      hit = hi-1;
    fhit=fPrimAcu[hit];
    if (fhit>Random){
      hi = hit;
      fhi = fhit;
    }else{
      lo = hit;
      flo = fhit;
    }
  }
  if (fPrimAcu[lo]>Random)
    pCell = (TFoamCell *) fCellsAct->At(lo);
  else
    pCell = (TFoamCell *) fCellsAct->At(hi);
}       // TFoam::GenerCel2


//___________________________________________________________________________________________
void TFoam::MakeEvent(void)
{
// User subprogram.
// It generates randomly point/vector according to user-defined distribution.
// Prior initialization with help of Initialize() is mandatory.
// Generated MC point/vector is available using GetMCvect and the MC weight with GetMCwt.
// MC point is generated with wt=1 or with variable weight, see OptRej switch.

  Int_t      j;
  Double_t   Wt,Dx,MCwt;
  TFoamCell *rCell;
  //
  //********************** MC LOOP STARS HERE **********************
 ee0:
    GenerCel2(rCell);   // choose randomly one cell

  MakeAlpha();

  TFoamVect  Posi(fDim); TFoamVect  Size(fDim);
  rCell->GetHcub(Posi,Size);
  for(j=0; j<fDim; j++)
    fMCvect[j]= Posi[j] +fAlpha[j]*Size[j];
  Dx = rCell->GetVolume();      // Cartesian volume of the Cell
  //  weight average normalized to PRIMARY integral over the cell

  Wt=Dx*Eval(fMCvect);

  MCwt = Wt / rCell->GetPrim();  // PRIMARY controls normalization
  fNCalls++;
  fMCwt   =  MCwt;
  // accumulation of statistics for the main MC weight
  fSumWt  += MCwt;           // sum of Wt
  fSumWt2 += MCwt*MCwt;      // sum of Wt**2
  fNevGen++;                 // sum of 1d0
  fWtMax  =  TMath::Max(fWtMax, MCwt);   // maximum wt
  fWtMin  =  TMath::Min(fWtMin, MCwt);   // minimum wt
  fMCMonit->Fill(MCwt);
  fHistWt->Fill(MCwt,1.0);          // histogram
  //*******  Optional rejection ******
  if(fOptRej == 1){
    Double_t Random;
    Random=fPseRan->Rndm();
    if( fMaxWtRej*Random > fMCwt) goto ee0;  // Wt=1 events, internal rejection
    if( fMCwt<fMaxWtRej ){
      fMCwt = 1.0;                  // normal Wt=1 event
    }else{
      fMCwt = fMCwt/fMaxWtRej;    // weight for overveighted events! kept for debug
      fSumOve += fMCwt-fMaxWtRej; // contribution of overveighted
    }
  }
  //********************** MC LOOP ENDS HERE **********************
} // MakeEvent

//_________________________________________________________________________________
void TFoam::GetMCvect(Double_t *MCvect)
{
// User may get generated MC point/vector with help of this method

  for ( Int_t k=0 ; k<fDim ; k++) *(MCvect +k) = fMCvect[k];
}//GetMCvect

//___________________________________________________________________________________
Double_t TFoam::GetMCwt(void)
{
// User may get weight MC weight using this method

  return(fMCwt);
}
//___________________________________________________________________________________
void TFoam::GetMCwt(Double_t &MCwt)
{
// User may get weight MC weight using this method

  MCwt=fMCwt;
}

//___________________________________________________________________________________
Double_t TFoam::MCgenerate(Double_t *MCvect)
{
// User subprogram which generates MC event and returns MC weight

   MakeEvent();
   GetMCvect(MCvect);
   return(fMCwt);
}//MCgenerate

//___________________________________________________________________________________
void TFoam::GetIntegMC(Double_t &MCresult, Double_t &MCerror)
{
// User subprogram.
// It provides the value of the integral calculated from the averages of the MC run
// May be called after (or during) the MC run.

  Double_t MCerelat;
  MCresult = 0.0;
  MCerelat = 1.0;
  if (fNevGen>0){
    MCresult = fPrime*fSumWt/fNevGen;
    MCerelat = sqrt( fSumWt2/(fSumWt*fSumWt) - 1/fNevGen);
  }
  MCerror = MCresult *MCerelat;
}//GetIntegMC

//____________________________________________________________________________________
void  TFoam::GetIntNorm(Double_t& IntNorm, Double_t& Errel )
{
// User subprogram.
// It returns NORMALIZATION integral to be combined with the average weights
// and content of the histograms in order to get proper absolute normaizadion
// of the integranl and distributions.
// It can be called after initialization, before or during the MC run.

  if(fOptRej == 1){    // Wt=1 events, internal rejection
    Double_t IntMC,ErrMC;
    GetIntegMC(IntMC,ErrMC);
    IntNorm = IntMC;
    Errel   = ErrMC;
  }else{                // Wted events, NO internal rejection
    IntNorm = fPrime;
    Errel   = 0;
  }
}//GetIntNorm

//______________________________________________________________________________________
void  TFoam::GetWtParams(Double_t eps, Double_t &AveWt, Double_t &WtMax, Double_t &Sigma)
{
// May be called optionally after the MC run.
// Returns various parameters of the MC weight for efficiency evaluation

  Double_t MCeff, WtLim;
  fMCMonit->GetMCeff(eps, MCeff, WtLim);
  WtMax = WtLim;
  AveWt = fSumWt/fNevGen;
  Sigma = sqrt( fSumWt2/fNevGen -AveWt*AveWt );
}//GetMCeff

//_______________________________________________________________________________________
void TFoam::Finalize(Double_t& IntNorm, Double_t& Errel)
{
// May be called optionally by the user after the MC run.
// It provides normalization and also prints some information/statistics on the MC run.

  GetIntNorm(IntNorm,Errel);
  Double_t MCresult,MCerror;
  GetIntegMC(MCresult,MCerror);
  Double_t MCerelat= MCerror/MCresult;
  //
  if(fChat>0){
    Double_t eps = 0.0005;
    Double_t MCeff, MCef2, WtMax, AveWt, Sigma;
    GetWtParams(eps, AveWt, WtMax, Sigma);
    MCeff=0;
    if(WtMax>0.0) MCeff=AveWt/WtMax;
    MCef2 = Sigma/AveWt;
    Double_t Driver = fCells[0]->GetDriv();
    //
    BXOPE;
    BXTXT("****************************************");
    BXTXT("******     TFoam::Finalize       ******");
    BXTXT("****************************************");
    BX1I("    NevGen",fNevGen, "Number of generated events in the MC generation   ");
    BX1I("    nCalls",fNCalls, "Total number of function calls                    ");
    BXTXT("----------------------------------------");
    BX1F("     AveWt",AveWt,    "Average MC weight                      ");
    BX1F("     WtMin",fWtMin,  "Minimum MC weight (absolute)           ");
    BX1F("     WtMax",fWtMax,  "Maximum MC weight (absolute)           ");
    BXTXT("----------------------------------------");
    BX1F("    XPrime",fPrime,  "Primary total integral, R_prime        ");
    BX1F("    XDiver",Driver,   "Driver  total integral, R_loss         ");
    BXTXT("----------------------------------------");
    BX2F("    IntMC", MCresult,  MCerror,      "Result of the MC Integral");
    BX1F(" MCerelat", MCerelat,  "Relative error of the MC intgral       ");
    BX1F(" <w>/WtMax",MCeff,     "MC efficiency, acceptance rate");
    BX1F(" Sigma/<w>",MCef2,     "MC efficiency, variance/ave_wt");
    BX1F("     WtMax",WtMax,     "WtMax(esp= 0.0005)            ");
    BX1F("     Sigma",Sigma,     "variance of MC weight         ");
    if(fOptRej==1){
       Double_t AvOve=fSumOve/fSumWt;
    BX1F("<OveW>/<W>",AvOve,     "Contrib. of events wt>MaxWtRej");
    }
    BXCLO;
  }
}  // Finalize

//_____________________________________________________________________________________
void  TFoam::SetInhiDiv(Int_t iDim, Int_t InhiDiv)
{
// This can be called before Initialize, after setting kDim
// It defines which variables are excluded in the process of the cell division.
// For example 'FoamX->SetInhiDiv(1, 1);' inhibits division of y-variable.
// The resulting map of cells in 2-dim. case will look as follows:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_Map2.gif">
<!--*/
// -->END_HTML

  if(fDim==0) Error("TFoam","SetInhiDiv: fDim=0 \n");
  if(fInhiDiv == 0){
    fInhiDiv = new Int_t[ fDim ];
    for(Int_t i=0; i<fDim; i++) fInhiDiv[i]=0;
  }
  //
  if( ( 0<=iDim) && (iDim<fDim)){
    fInhiDiv[iDim] = InhiDiv;
  }else
    Error("SetInhiDiv:","Wrong iDim \n");
}//SetInhiDiv

//______________________________________________________________________________________
void  TFoam::SetXdivPRD(Int_t iDim, Int_t len, Double_t xDiv[])
{
// This should be called before Initialize, after setting  kDim
// It predefines values of the cell division for certain variable iDim.
// For example setting 3 predefined division lines using:
//     xDiv[0]=0.30; xDiv[1]=0.40; xDiv[2]=0.65;
//     FoamX->SetXdivPRD(0,3,xDiv);
// results in the following 2-dim. pattern of the cells:
//BEGIN_HTML <!--
/* -->
<img src="gif/foam_Map3.gif">
<!--*/
// -->END_HTML

  Int_t i;

  if(fDim<=0)  Error("SetXdivPRD", "fDim=0 \n");
  if(   len<1 )  Error("SetXdivPRD", "len<1 \n");
  // allocate list of pointers, if it was not done before
  if(fXdivPRD == 0){
    fXdivPRD = new TFoamVect*[fDim];
    for(i=0; i<fDim; i++)  fXdivPRD[i]=0; 
  }
  // set division list for direction iDim in H-cubic space!!!
  if( ( 0<=iDim) && (iDim<fDim)){
    fOptPRD =1;      // !!!!
    if( fXdivPRD[iDim] != 0)
      Error("SetXdivPRD", "Second allocation of XdivPRD not allowed \n");
    fXdivPRD[iDim] = new TFoamVect(len); // allocate list of division points
    for(i=0; i<len; i++){
      (*fXdivPRD[iDim])[i]=xDiv[i]; // set list of division points
    }
  }else
    Error("SetXdivPRD", "Wrong iDim  \n");

  // Priting predefined division points 
  cout<<" SetXdivPRD, idim= "<<iDim<<"  len= "<<len<<"   "<<endl;
  for(i=0; i<len; i++){
    cout<< (*fXdivPRD[iDim])[i] <<"  ";
  }
  cout<<endl;
  for(i=0; i<len; i++)  cout<< xDiv[i] <<"   ";
  cout<<endl;
  //
}//SetXdivPRD

//_______________________________________________________________________________________
void TFoam::CheckAll(Int_t level)
{
//  User utility, miscellaneous and debug.
//  Checks all pointers in the tree of cells. This is useful autodiagnostic.
//  level=0, no printout, failures causes STOP
//  level=1, printout, failures lead to WARNINGS only

  Int_t Errors, Warnings;
  TFoamCell *Cell;
  Long_t iCell;

  Errors = 0; Warnings = 0;
  if (level==1) cout << "///////////////////////////// FOAM_Checks /////////////////////////////////" << endl;
  for(iCell=1; iCell<=fLastCe; iCell++){
    Cell = fCells[iCell];
//  checking general rules
    if( ((Cell->GetDau0()==0) && (Cell->GetDau1()!=0) ) ||
        ((Cell->GetDau1()==0) && (Cell->GetDau0()!=0) ) ){
      Errors++;
      if (level==1) Error("CheckAll","ERROR: Cell's no %d has only one daughter \n",iCell);
    }
    if( (Cell->GetDau0()==0) && (Cell->GetDau1()==0) && (Cell->GetStat()==0) ){
      Errors++;
      if (level==1) Error("CheckAll","ERROR: Cell's no %d  has no daughter and is inactive \n",iCell);
    }
    if( (Cell->GetDau0()!=0) && (Cell->GetDau1()!=0) && (Cell->GetStat()==1) ){
      Errors++;
      if (level==1) Error("CheckAll","ERROR: Cell's no %d has two daughters and is active \n",iCell);
    }

// checking parents
    if( (Cell->GetPare())!=fCells[0] ){ // not child of the root
      if ( (Cell != Cell->GetPare()->GetDau0()) && (Cell != Cell->GetPare()->GetDau1()) ){
        Errors++;
        if (level==1) Error("CheckAll","ERROR: Cell's no %d parent not pointing to this cell\n ",iCell);
      }
    }

// checking daughters
    if(Cell->GetDau0()!=0){
      if(Cell != (Cell->GetDau0())->GetPare()){
        Errors++;
        if (level==1)  Error("CheckAll","ERROR: Cell's no %d daughter 0 not pointing to this cell \n",iCell);
      }
    }
    if(Cell->GetDau1()!=0){
      if(Cell != (Cell->GetDau1())->GetPare()){
        Errors++;
        if (level==1) Error("CheckAll","ERROR: Cell's no %d daughter 1 not pointing to this cell \n",iCell);
      }
    }
  }// loop after cells;

// Check for empty cells
  for(iCell=0; iCell<=fLastCe; iCell++){
    Cell = fCells[iCell];
    if( (Cell->GetStat()==1) && (Cell->GetDriv()==0) ){
      Warnings++;
      if(level==1) Warning("CheckAll", "Warning: Cell no. %d is active but empty \n", iCell);
    }
  }
// summary
  if(level==1){
    Info("CheckAll","Check has found %d errors and %d warnings \n",Errors, Warnings);
  }
  if(Errors>0){
    Info("CheckAll","Check - found total %d  errors \n",Errors);
  }
} // Check

//________________________________________________________________________________________
void TFoam::PrintCells(void)
{
// Prints geometry of ALL cells of the FOAM

  Long_t iCell;

  for(iCell=0; iCell<=fLastCe; iCell++){
    cout<<"Cell["<<iCell<<"]={ ";
    //cout<<"  "<< fCells[iCell]<<"  ";  // extra DEBUG
    cout<<endl;
    fCells[iCell]->Print("1");
    cout<<"}"<<endl;
  }
}

//_________________________________________________________________________________________
void TFoam::RootPlot2dim(Char_t *filename)
{
// Debugging tool which plots 2-dimensional cells as rectangles
// in C++ format readable for root

  ofstream outfile(filename, ios::out);
  Double_t   x1,y1,x2,y2,x,y;
  Long_t    iCell;
  Double_t offs =0.1;
  Double_t lpag   =1-2*offs;
  outfile<<"{" << endl;
  outfile<<"cMap = new TCanvas(\"Map1\",\" Cell Map \",600,600);"<<endl;
  //
  outfile<<"TBox*a=new TBox();"<<endl;
  outfile<<"a->SetFillStyle(0);"<<endl;  // big frame
  outfile<<"a->SetLineWidth(4);"<<endl;
  outfile<<"a->SetLineColor(2);"<<endl;
  outfile<<"a->DrawBox("<<offs<<","<<offs<<","<<(offs+lpag)<<","<<(offs+lpag)<<");"<<endl;
  //
  outfile<<"TText*t=new TText();"<<endl;  // text for numbering
  outfile<<"t->SetTextColor(4);"<<endl;
  if(fLastCe<51)
    outfile<<"t->SetTextSize(0.025);"<<endl;  // text for numbering
  else if(fLastCe<251)
    outfile<<"t->SetTextSize(0.015);"<<endl;
  else
    outfile<<"t->SetTextSize(0.008);"<<endl;
  //
  outfile<<"TBox*b=new TBox();"<<endl;  // single cell
  outfile <<"b->SetFillStyle(0);"<<endl;
  //
  if(fDim==2 && fLastCe<=2000){
    TFoamVect  Posi(fDim); TFoamVect  Size(fDim);
    outfile << "// =========== Rectangular cells  ==========="<< endl;
    for(iCell=1; iCell<=fLastCe; iCell++){
      if( fCells[iCell]->GetStat() == 1){
        fCells[iCell]->GetHcub(Posi,Size);
        x1 = offs+lpag*(        Posi[0]); y1 = offs+lpag*(        Posi[1]);
        x2 = offs+lpag*(Posi[0]+Size[0]); y2 = offs+lpag*(Posi[1]+Size[1]);
        //     cell rectangle
        if(fLastCe<=2000)
          outfile<<"b->DrawBox("<<x1<<","<<y1<<","<<x2<<","<<y2<<");"<<endl;
        //     cell number
        if(fLastCe<=250){
          x = offs+lpag*(Posi[0]+0.5*Size[0]); y = offs+lpag*(Posi[1]+0.5*Size[1]);
          outfile<<"t->DrawText("<<x<<","<<y<<","<<"\""<<iCell<<"\""<<");"<<endl;
        }
      }
    }
    outfile<<"// ============== End Rectangles ==========="<< endl;
  }//kDim=2
  //
  //
  outfile << "}" << endl;
  outfile.close();
}

void TFoam::LinkCells()
{
// Void function for backward compatibility

  Info("LinkCells", "VOID function for backward compatibility \n");
  return;
}

////////////////////////////////////////////////////////////////////////////////
//       End of Class TFoam                                                   //
////////////////////////////////////////////////////////////////////////////////

