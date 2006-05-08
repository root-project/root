// @(#)root/tmva $Id: TMVA_MethodCFMlpANN.cxx,v 1.2 2006/05/08 21:33:46 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCFMlpANN                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Interface for Clermond-Ferrand artificial neural network             
//                                                                      
//_______________________________________________________________________

#include "TMatrix.h"
#include "TObjString.h"
#include "Riostream.h"
#include <string>
#include "TMVA_MethodCFMlpANN.h"
#include "TMVA_MethodCFMlpANN_def.h"
#include "TMVA_Tools.h"

#define DEBUG_TMVA_MethodCFMlpANN kFALSE

ClassImp(TMVA_MethodCFMlpANN)

// initialization of statics
static Int_t         TMVA_MethodCFMlpANN_nsel    = 0;
TMVA_MethodCFMlpANN* TMVA_MethodCFMlpANN::fThis = 0;

// references for mlpl3 functions <=======please check
//extern "C" Int_t train_nn__( Double_t *tin2, Double_t *tout2, Int_t *ntrain, 
//			     Int_t *ntest, Int_t *nvar2, Int_t *nlayer, 
//			     Int_t *nodes, Int_t *ncycle );

//_______________________________________________________________________
int TMVA_MethodCFMlpANN_dataInterface( Double_t* /*tout2*/, Double_t*  /*tin2*/, 
				       Int_t* /* icode*/, Int_t*  /*flag*/, 
				       Int_t*  /*nalire*/, Int_t* nvar, 
				       Double_t* xpg, Int_t* iclass, Int_t* ikend )
{
  // Data interface function 
   
  // icode and ikend are dummies needed to match f2c mlpl3 functions
  *ikend = 0; 

  // retrieve pointer to current object (CFMlpANN must be a singleton class!)
  TMVA_MethodCFMlpANN* O = TMVA_MethodCFMlpANN::This();

  // sanity checks
  if (0 == xpg) {
    cout << "*** ERROR in MethodCFMlpANN_DataInterface zero pointer xpg ==> exit(1)"
	 << endl;
    exit(1);
  }
  if (*nvar != O->GetNvar()) {
    cout << "*** ERROR in MethodCFMlpANN_DataInterface mismatch in num of variables: " 
	 << *nvar << " " << O->GetNvar()
	 << " ==> exit(1)"
	 << endl;
    exit(1);
  }

  // fill variables
  *iclass = (int)O->GetClass( TMVA_MethodCFMlpANN_nsel );
  for (Int_t ivar=0; ivar<O->GetNvar(); ivar++) 
    xpg[ivar] = (double)O->GetData( TMVA_MethodCFMlpANN_nsel, ivar );

  ++TMVA_MethodCFMlpANN_nsel;

  return 0;
}

//_______________________________________________________________________
void TMVA_MethodCFMlpANN_writeWeightsToFile( Int_t nva, Int_t lclass, 
					     Double_t* xmaxNN, Double_t* xminNN,
					     Int_t layermNN, Int_t* neuronNN, 
					     Double_t* wNN, Double_t* wwNN, Double_t* tempNN )
{
   // file interface function
   
#define w_ref(a_1,a_2,a_3) wNN [((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]
#define ww_ref(a_1,a_2)    wwNN[(a_2)*max_nLayers_ + a_1 - 7]

  // retrieve pointer to current object (CFMlpANN must be a singleton class!)
  TMVA_MethodCFMlpANN* O = TMVA_MethodCFMlpANN::This();

  TString fname      = O->GetWeightFileName();
  TString ClassName  = "TMVA_MethodCFMlpANN";
  cout << "--- " << ClassName << ": creating weight file: " << fname << endl;  

  Bool_t isOK = kTRUE;

  // open file
  ofstream* fout = new ofstream( fname );

  if (!fout->good( )) { // file not found --> Error
    cout << "--- " << ClassName << ": Error in ::WriteWeightsToFile: "
	 << "unable to open input file: " << fname << endl;
    isOK = kFALSE;
  }
  else {

    // write variable names and min/max 
    // NOTE: the latter values are mandatory for the normalisation 
    // in the reader application !!!
    for (Int_t ivar=0; ivar<O->GetNvar(); ivar++) {
      TString var = (*O->GetInputVars())[ivar];
      *fout << var << "  " << O->GetXminNorm( var ) << "  " << O->GetXmaxNorm( var ) << endl;
    }
      
    // write number of variables and classes
    *fout << nva << "    " << lclass << endl;
      
    // number of output classes must be 2
    if (lclass != 2) { // wrong file
      cout << "--- " << ClassName << ": Error in ::WriteWeightsToFile: "
	   << "mismatch in number of classes" << endl;
    }
    else {

      // check that we are not at the end of the file
      if (fout->eof( )) {
	cout << "--- " << ClassName << ": Error in ::WriteWeightsToFile: "
	     << "EOF while writing output file: " << fname << endl;
      }
      else {
	
	// write extrema of input variables
	for (Int_t ivar=0; ivar<nva; ivar++) 
	  *fout << xmaxNN[ivar] << "   " << xminNN[ivar] << endl;
	
	// write number of layers (sum of: input + output + hidden)
	*fout << layermNN << endl;;
	
	Int_t layer(0);
	for (layer=0; layer<layermNN; layer++) {	      
	  // write number of neurons for each layer
	  *fout << neuronNN[layer] << "     ";
	}
	*fout << endl;
	
	// write weights
	for (layer=1; layer<=layermNN-1; layer++) { 
	  
	  Int_t nq = neuronNN[layer]/10;
	  Int_t nr = neuronNN[layer] - nq*10;
	  
	  Int_t kk(0);
	  if (nr==0) kk = nq;
	  else       kk = nq+1;
	  
	  for (Int_t k=1; k<=kk; k++) {
	    Int_t jmin = 10*k - 9;
	    Int_t jmax = 10*k;
	    Int_t i, j;
	    if (neuronNN[layer]<jmax) jmax = neuronNN[layer];
	    for (j=jmin; j<=jmax; j++) *fout << ww_ref(layer + 1, j) << "   ";
	    *fout << endl;
	    for (i=1; i<=neuronNN[layer-1]; i++) {
	      for (j=jmin; j<=jmax; j++) *fout << w_ref(layer + 1, j, i) << "   ";
	      *fout << endl;
	    }
	    
	    // skip two empty lines
	    *fout << endl << endl;
	  }
	}
	for (layer=0; layer<layermNN; layer++) {
	  
	  // skip 2 empty lines
	  *fout << endl << endl;	  
	  *fout << tempNN[layer] << endl;
	}       	
      }
    }

    // close input file
    fout->close();
  }

  delete fout;

  if (!isOK) exit(1); // be brutal
}

//_______________________________________________________________________
TMVA_MethodCFMlpANN::TMVA_MethodCFMlpANN( TString jobName, vector<TString>* theVariables,  
					TTree* theTree, TString theOption, TDirectory* theTargetDir  )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir  )
{
  InitCFMlpANN();

  if (fOptions.Sizeof()<2) {
    fOptions = "3000:N-1:N-2";
    cout << "--- " << GetName() << ": problems with options; using default: " 
 	 << fOptions << endl;
  }  

  //--------------------------------------------------------------

  // parse the option string
  vector<Int_t>* nodes = parseOptionString( fOptions, fNvar, new vector<Int_t> );

  // sanity check: exactly two numbers in string
  if (nodes->size() < 1) {
    cout << "--- " << GetName() << ": Error: wrong number of arguments"
 	 << " in options string: " << fOptions
 	 << " | required format is: n_cycles:n_layers" << endl;
    exit(1);
  }
  fNcycles = (*nodes)[0];
  
  // total number of layers in ANN:
  // add 2 (input and output) layers to hidden layers
  fNlayers = 2 + (nodes->size() - 1); 
  fNodes   = new Int_t[fNlayers]; // number of nodes per layer (all layers)
  
  fNodes[0]           = fNvar; // input layer
  fNodes[fNlayers-1] = 2;      // output layer
  for (Int_t i=1; i<fNlayers-1; i++) fNodes[i] = ((*nodes)[i] < 2) ? 2 : (*nodes)[i];

  cout << "--- " << GetName() << ": use " << fNcycles << " training cycles" << endl;
  cout << "--- " << GetName() << ": use configuration (nodes per layer): in:";
  for (Int_t i=0; i<fNlayers; i++) cout << fNodes[i] << ":";
  cout << "out" << endl;

   // note that one variable is type
  if (0 != fTrainingTree) {
    
    // trainingTree should only contain those variables that are used in the MVA
    if (fTrainingTree->GetListOfBranches()->GetEntries() - 1 != fNvar) {
      cout << "--- " << GetName() << ": Error: mismatch in number of variables" 
 	   << " --> exit(1)" << endl;
      exit(1);
    }
      
    fNevt  = fTrainingTree->GetEntries();

    // Data LUT
    fData  = new TMatrix( fNevt, fNvar );
    fClass = new vector<Int_t>( fNevt );

    // count number of signal and background events
    fNsig = 0;
    fNbgd = 0;
    for (Int_t ievt = 0; ievt < fNevt; ievt++) 
      if ((Int_t)TMVA_Tools::GetValue( fTrainingTree, ievt, "type" ) == 1) 
	++fNsig;
      else                                                      
	++fNbgd;

    // numbers of events should match
    if (fNsig + fNbgd != fNevt) {
      cout << "--- " << GetName() << ": Error: mismatch in number of events" 
	   << " --> exit(1)" << endl;
      exit(1);
    }    

    // ---- fill LUTs
    
    Int_t isig = 0, ibgd = 0, ivar;
    for (Int_t ievt=0; ievt<fNevt; ievt++) {
      
      // identify signal and background events  
      if ((Int_t)TMVA_Tools::GetValue( fTrainingTree, ievt, "type" ) == 1) {
	(*fClass)[ievt] = 1;
	++isig;
      }
      else {
	(*fClass)[ievt] = 2;
	++ibgd;
      }
      
      // use normalized input Data
      for (ivar=0; ivar<fNvar; ivar++) {
	Double_t x = TMVA_Tools::GetValue( fTrainingTree, ievt, (*fInputVars)[ivar] );
	(*fData)( ievt, ivar ) = __N__( x, GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
      }
    }

    if (Verbose())
      cout << "--- " << GetName() << " <verbose>: " 
	   << isig << " signal and " << ibgd << " background"
	   << " events in trainingTree" << endl;
  }
  else {
    fNevt = 0;
    fNsig = 0;
    fNbgd = 0;
  }

  delete nodes;
}

//_______________________________________________________________________
TMVA_MethodCFMlpANN::TMVA_MethodCFMlpANN( vector<TString> *theVariables, 
					  TString theWeightFile,  
					  TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitCFMlpANN();
}

void TMVA_MethodCFMlpANN::InitCFMlpANN( void )
{
  fMethodName = "CFMlpANN";
  fMethod     = TMVA_Types::CFMlpANN;
  fTestvar    = fTestvarPrefix+GetMethodName();

  // initialize all pointers
  fNodes    = 0;
  fNeuronNN = 0;
  fWNN      = 0;
  fWwNN     = 0;
  fYNN      = 0;
  fTempNN   = 0;
  fXmaxNN   = 0;
  fXminNN   = 0;   
  fThis     = this;  

  fNevt     = 0;
  fNsig     = 0;
  fNbgd     = 0;

  // initialize dimensions
  TMVA_MethodCFMlpANN_nsel = 0;  
}

//_______________________________________________________________________
TMVA_MethodCFMlpANN::~TMVA_MethodCFMlpANN( void )
{
  // let's clean up
  delete fData;
  delete fClass;
  delete fNodes;

  delete [] fNeuronNN;
  delete [] fWNN;
  delete [] fWwNN;
  delete [] fYNN;
  delete [] fTempNN;
  
  delete [] fXmaxNN;
  delete [] fXminNN;   
}

//_______________________________________________________________________
void TMVA_MethodCFMlpANN::Train( void )
{
  //--------------------------------------------------------------

  // default sanity checks
  if (!CheckSanity()) { 
    cout << "--- " << GetName() << ": Error in ::Train sanity check failed" << endl;
    exit(1);
  }

  Double_t* dumDat  = 0;
  Int_t* ntrain  = new Int_t(fNevt);
  Int_t* ntest   = new Int_t(0);
  Int_t* nvar    = new Int_t(fNvar);
  Int_t* nlayers = new Int_t(fNlayers);
  Int_t* nodes   = new Int_t[*nlayers]; 
  for (Int_t i=0; i<*nlayers; i++) nodes[i] = fNodes[i]; // full copy of class member
  Int_t* ncycles = new Int_t(fNcycles);

  //please check
  //train_nn__( dumDat, dumDat, ntrain, ntest, nvar, nlayers, nodes, ncycles );
  
  delete nodes;
  delete ntrain;
  delete ntest;
  delete nvar; 
  //  delete [] nodes; --> problem, should one solve :-)
  delete ncycles;
  delete nlayers;
}

//_______________________________________________________________________
Double_t TMVA_MethodCFMlpANN::GetMvaValue( TMVA_Event *e )
{
  Double_t myMVA = 0;
  Bool_t isOK = kTRUE;
  vector<Double_t>* inputVec = new vector<Double_t>( fNvar );

  for (Int_t ivar=0; ivar<fNvar; ivar++) 
    (*inputVec)[ivar] = __N__( e->GetData(ivar), GetXminNorm( ivar ), GetXmaxNorm( ivar ) );

  myMVA = evalANN( inputVec, isOK );
  if (!isOK) {
    cout << "--- " << GetName() << ": Problem in ::evalANN (!isOK) for event " << e
	 << " ==> exit(1)"
	 << endl;
    exit(1);
  }
  delete inputVec;
  return myMVA;
}

//_______________________________________________________________________
Double_t TMVA_MethodCFMlpANN::evalANN( vector<Double_t>* inVar, Bool_t& isOK )
{
  Double_t* xeev = new Double_t[fNvar];

  // hardcopy
  for (Int_t ivar=0; ivar<fNvar; ivar++) xeev[ivar] = (*inVar)[ivar];
  
  // ---- now apply the weights: get NN output
  isOK = kTRUE;
  for (Int_t jvar=0; jvar<fNvar; jvar++) {

    if (fXmaxNN[jvar] < xeev[jvar]     ) xeev[jvar] = fXmaxNN[jvar];
    if (fXminNN[jvar] > xeev[jvar]     ) xeev[jvar] = fXminNN[jvar];
    if (fXmaxNN[jvar] == fXminNN[jvar]) {
      isOK = kFALSE;
      xeev[jvar] = 0;
    }
    else {
      xeev[jvar] = xeev[jvar] - ((fXmaxNN[jvar] + fXminNN[jvar])/2);    
      xeev[jvar] = xeev[jvar] / ((fXmaxNN[jvar] - fXminNN[jvar])/2);    
    }
  }
    
  nn_ava( xeev );

  delete [] xeev;

  // return NN output, note: fYNN[..][0] = -fYNN[..][1]
  // transform to confine it within [0,1] (originally in [-1,1])
  return 0.5*(1.0 + fYNN[fLayermNN-1][0]);
}

//_______________________________________________________________________
void  TMVA_MethodCFMlpANN::nn_ava( Double_t* xeev )
{  
  for (Int_t ivar=0; ivar<fNeuronNN[0]; ivar++) fYNN[0][ivar] = xeev[ivar];
  
  for (Int_t layer=0; layer<fLayermNN-1; layer++) {
    for (Int_t j=0; j<fNeuronNN[layer+1]; j++) {

      Double_t x( 0 );
      for (Int_t k=0; k<fNeuronNN[layer]; k++) 
	x = x + fYNN[layer][k]*fWNN[layer+1][j][k];

      x = x + fWwNN[layer+1][j];      
      fYNN[layer+1][j] = nn_fonc( layer+1, x );
    }
  }  
}

//_______________________________________________________________________
Double_t TMVA_MethodCFMlpANN::nn_fonc( Int_t i, Double_t u ) const
{
  Double_t f(0);
  
  if      (u/fTempNN[i] >  170) f = +1;
  else if (u/fTempNN[i] < -170) f = -1;
  else {
    Double_t yy = exp(-u/fTempNN[i]);
    f  = (1 - yy)/(1 + yy);
  }

  return f;
}

//_______________________________________________________________________
void TMVA_MethodCFMlpANN::WriteWeightsToFile( void )
{  
   // write coefficients to file
   // not used; weights are saved in TMVA_MethodCFMlpANN_f2c
}
  
//_______________________________________________________________________
void TMVA_MethodCFMlpANN::ReadWeightsFromFile( void )
{
   // read coefficients from file
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": reading weight file: " << fname << endl;  

  Bool_t isOK = kTRUE;

  // open file
  ifstream* fin = new ifstream( fname );

  if (!fin->good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
	 << "unable to open input file: " << fname << endl;
    isOK = kFALSE;
  }
  else {
      
    // read variable names and min/max
    // NOTE: the latter values are mandatory for the normalisation 
    // in the reader application !!!
    TString var;
    Double_t xmin, xmax;
    for (Int_t ivar=0; ivar<fNvar; ivar++) {
      *fin >> var >> xmin >> xmax;
      
      // sanity check
      if (var != (*fInputVars)[ivar]) {
	cout << "--- " << GetName() << ": Error while reading weight file; "
	     << "unknown variable: " << var << " at position: " << ivar << ". "
	     << "Expected variable: " << (*fInputVars)[ivar] << " ==> abort" << endl;
	exit(1);
      }
      
      // set min/max
      this->SetXminNorm( ivar, xmin );
      this->SetXmaxNorm( ivar, xmax );
    }  

    // read number of variables and classes
    Int_t nva(0), lclass(0);
    *fin >> nva >> lclass;
      
    if (fNvar != nva) { // wrong file
      cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
	   << "mismatch in number of variables" << endl;
    }
    else {

      // number of output classes must be 2
      if (lclass != 2) { // wrong file
	cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
	     << "mismatch in number of classes" << endl;
      }
      else {
	  
	// check that we are not at the end of the file
	if (fin->eof( )) {
	  cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
	       << "EOF while reading input file: " << fname << endl;
	}
	else {
	    
	  fXmaxNN = new Double_t[fNvar];
	  fXminNN = new Double_t[fNvar];
	    
	  // read extrema of input variables
	  for (Int_t ivar=0; ivar<fNvar; ivar++) 
	    *fin >> fXmaxNN[ivar] >> fXminNN[ivar];
	    
	  // read number of layers (sum of: input + output + hidden)
	  *fin >> fLayermNN;
	    
	  fNeuronNN = new Int_t     [fLayermNN];
	  fWNN      = new Double_t**[fLayermNN];
	  fWwNN     = new Double_t* [fLayermNN];
	  fYNN      = new Double_t* [fLayermNN];
	  fTempNN   = new Double_t  [fLayermNN];

	  Int_t layer(0);
	  for (layer=0; layer<fLayermNN; layer++) {
	      
	    // read number of neurons for each layer
	    *fin >> fNeuronNN[layer];
	      
	    Int_t Nneu = fNeuronNN[layer];
	      
	    fWNN [layer] = new Double_t*[Nneu];
	    fWwNN[layer] = new Double_t [Nneu];
	    fYNN [layer] = new Double_t [Nneu];
	    if (layer > 0)
	      for (Int_t neu=0; neu<Nneu; neu++) 
		fWNN[layer][neu] = new Double_t[fNeuronNN[layer-1]];
	  }
	    
	  // to read dummy lines
	  const Int_t nchar( 100 );
	  char* dumchar = new char[nchar];
	    
	  // read weights
	  for (layer=0; layer<fLayermNN-1; layer++) { 
	      
	    Int_t nq = fNeuronNN[layer+1]/10;
	    Int_t nr = fNeuronNN[layer+1] - nq*10;
	      
	    Int_t kk(0);
	    if (nr==0) kk = nq;
	    else       kk = nq+1;
	      
	    for (Int_t k=0; k<kk; k++) {
	      Int_t jmin = 10*(k+1) - 10;
	      Int_t jmax = 10*(k+1) - 1;
	      if (fNeuronNN[layer+1]-1<jmax) jmax = fNeuronNN[layer+1]-1;
	      for (Int_t j=jmin; j<=jmax; j++) *fin >> fWwNN[layer+1][j];
	      for (Int_t i=0; i<fNeuronNN[layer]; i++) 
		for (Int_t l=jmin; l<=jmax; l++) *fin >> fWNN[layer+1][l][i];

	      // skip two empty lines
	      fin->getline( dumchar, nchar );
	    }
	  }
	  for (layer=0; layer<fLayermNN; layer++) {
	      
	    // skip 2 empty lines
	    fin->getline( dumchar, nchar );
	    fin->getline( dumchar, nchar );
	      
	    *fin >> fTempNN[layer];
	  }	    
	}
      }
    }

    // close input file
    fin->close( );
  }

  delete fin;

  // sanity check
  if (fNvar != fNeuronNN[0]) {
    cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: mismatch in zeroth layer:"
	 << fNvar << " " << fNeuronNN[0] << " ==> exit(1)" << endl;
    exit(1);
  }

  if (!isOK) exit(1); // be brutal
}

//_______________________________________________________________________
void  TMVA_MethodCFMlpANN::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       << " special histos to file: " << fBaseDir->GetPath() << endl;
}

