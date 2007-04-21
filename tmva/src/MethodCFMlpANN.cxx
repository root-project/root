// @(#)root/tmva $Id: MethodCFMlpANN.cxx,v 1.13 2007/04/19 10:32:04 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodCFMlpANN                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Begin_Html
/*
  Interface to Clermond-Ferrand artificial neural network 

  <p>
  The CFMlpANN belong to the class of Multilayer Perceptrons (MLP), which are 
  feed-forward networks according to the following propagation schema:<br>
  <center>
  <img vspace=10 src="gif/tmva_mlp.gif" align="bottom" alt="Schema for artificial neural network"> 
  </center>
  The input layer contains as many neurons as input variables used in the MVA.
  The output layer contains two neurons for the signal and background
  event classes. In between the input and output layers are a variable number
  of <i>k</i> hidden layers with arbitrary numbers of neurons. (While the 
  structure of the input and output layers is determined by the problem, the 
  hidden layers can be configured by the user through the option string
  of the method booking.) <br>

  As indicated in the sketch, all neuron inputs to a layer are linear 
  combinations of the neuron output of the previous layer. The transfer
  from input to output within a neuron is performed by means of an "activation
  function". In general, the activation function of a neuron can be 
  zero (deactivated), one (linear), or non-linear. The above example uses
  a sigmoid activation function. The transfer function of the output layer
  is usually linear. As a consequence: an ANN without hidden layer should 
  give identical discrimination power as a linear discriminant analysis (Fisher).
  In case of one hidden layer, the ANN computes a linear combination of 
  sigmoid.  <br>

  The learning method used by the CFMlpANN is only stochastic.
*/
// End_Html
//_______________________________________________________________________

#include "TMatrix.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TMath.h"
#include <string>
#include <cmath>
#include "TMVA/MethodCFMlpANN.h"
#include "TMVA/MethodCFMlpANN_def.h"

#include <iostream>
using std::cout;
using std::endl;



ClassImp(TMVA::MethodCFMlpANN)

// initialization of global variable
namespace TMVA {
   Int_t MethodCFMlpANN_nsel = 0;
}

TMVA::MethodCFMlpANN* TMVA::MethodCFMlpANN::fgThis = 0;

//_______________________________________________________________________
TMVA::MethodCFMlpANN::MethodCFMlpANN( TString jobName, TString methodTitle, DataSet& theData, 
                                      TString theOption, TDirectory* theTargetDir  )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir  )
   , fNodes(0)
   , fYNN(0)
{
   // standard constructor
   // option string: "n_training_cycles:n_hidden_layers"  
   // default is:  n_training_cycles = 5000, n_layers = 4 
   //
   // * note that the number of hidden layers in the NN is: 
   //   n_hidden_layers = n_layers - 2
   //
   // * since there is one input and one output layer. The number of         
   //   nodes (neurons) is predefined to be:
   //   n_nodes[i] = nvars + 1 - i (where i=1..n_layers)                  
   //
   //   with nvars being the number of variables used in the NN.             
   //
   // Hence, the default case is: n_neurons(layer 1 (input)) : nvars       
   //                             n_neurons(layer 2 (hidden)): nvars-1     
   //                             n_neurons(layer 3 (hidden)): nvars-1     
   //                             n_neurons(layer 4 (out))   : 2           
   //
   // This artificial neural network usually needs a relatively large      
   // number of cycles to converge (8000 and more). Overtraining can       
   // be efficienctly tested by comparing the signal and background        
   // output of the NN for the events that were used for training and      
   // an independent data sample (with equal properties). If the separation
   // performance is significantly better for the training sample, the     
   // NN interprets statistical effects, and is hence overtrained. In       
   // this case, the number of cycles should be reduced, or the size       
   // of the training sample increased.                                    
   //
   InitCFMlpANN();
  
   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   // note that one variable is type
   if (HasTrainingTree()) {
    
      Int_t nEvtTrain = Data().GetNEvtTrain();

      // Data LUT
      fData  = new TMatrix( nEvtTrain, GetNvar() );
      fClass = new vector<Int_t>( nEvtTrain );
      // ---- fill LUTs

      Event& event = GetEvent();

      Int_t ivar;
      for (Int_t ievt=0; ievt<nEvtTrain; ievt++) {
         ReadTrainingEvent(ievt);

         // identify signal and background events  
         (*fClass)[ievt] = (event.IsSignal()?1:2);
      
         // use normalized input Data
         for (ivar=0; ivar<GetNvar(); ivar++) {
            (*fData)( ievt, ivar ) = GetEventValNormalized(ivar);
         }
      }

      fLogger << kVERBOSE << Data().GetNEvtSigTrain() << " signal and " 
              << Data().GetNEvtBkgdTrain() << " background" << " events in trainingTree" << Endl;
   }
}

//_______________________________________________________________________
TMVA::MethodCFMlpANN::MethodCFMlpANN( DataSet & theData, 
                                      TString theWeightFile,  
                                      TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
   , fNodes(0)
{
   // construction from weight file
   InitCFMlpANN();
  
   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options: NCycles=xx              :the number of training cycles
   //               HiddenLayser="N-1,N-2"  :the specification of the hidden layers
 
   DeclareOptionRef(fNcycles=3000,"NCycles","Number of training cycles");
   DeclareOptionRef(fLayerSpec="N-1,N-2","HiddenLayers","Specification of the hidden layers");
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::ProcessOptions() 
{
   // decode the options in the option string

   MethodBase::ProcessOptions();

   if(fYNN!=0) {
      for(Int_t i=0; i<fNlayers; i++) delete[] fYNN[i];
      delete[] fYNN;
      fYNN=0;
   }

   fNodes = new Int_t[20]; // number of nodes per layer (maximum 20 layers)
   fNlayers = 2;
   int currentHiddenLayer = 1;
   TString layerSpec(fLayerSpec);
   while(layerSpec.Length()>0) {
      TString sToAdd = "";
      if(layerSpec.First(',')<0) {
         sToAdd = layerSpec;
         layerSpec = "";
      } else {
         sToAdd = layerSpec(0,layerSpec.First(','));
         layerSpec = layerSpec(layerSpec.First(',')+1,layerSpec.Length());
      }
      int nNodes = 0;
      if(sToAdd.BeginsWith("N") || sToAdd.BeginsWith("n")) { sToAdd.Remove(0,1); nNodes = GetNvar(); }
      nNodes += atoi(sToAdd);
      fNodes[currentHiddenLayer++] = nNodes;
      fNlayers++;
   }
   fNodes[0]          = GetNvar(); // number of input nodes
   fNodes[fNlayers-1] = 2;         // number of output nodes

   fLogger << kINFO << "use " << fNcycles << " training cycles" << Endl;
   fLogger << kINFO << "use configuration (nodes per layer): in=";
   for (Int_t i=0; i<fNlayers-1; i++) fLogger << kINFO << fNodes[i] << ":";
   fLogger << kINFO << fNodes[fNlayers-1] << "=out" << Endl;
   
   fYNN = new Double_t* [fNlayers];
   for(Int_t i=0;i<fNlayers;i++)
      fYNN[i] = new Double_t[fNodes[i]];

}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::InitCFMlpANN( void )
{
   // default initialisation called by all constructors
   SetMethodName( "CFMlpANN" );
   SetMethodType( TMVA::Types::kCFMlpANN );
   SetTestvarName();

   // initialize all pointers
   fgThis    = this;  

   // initialize dimensions
   TMVA::MethodCFMlpANN_nsel = 0;  
}

//_______________________________________________________________________
TMVA::MethodCFMlpANN::~MethodCFMlpANN( void )
{
   // destructor
   delete fData;
   delete fClass;
   delete[] fNodes;

   if(fYNN!=0) {
      for(Int_t i=0; i<fNlayers; i++) delete[] fYNN[i];
      delete[] fYNN;
      fYNN=0;
   }
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::Train( void )
{
   // training of the Clement-Ferrand NN classifier

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   Double_t dumDat(0);
   Int_t ntrain(Data().GetNEvtTrain());
   Int_t ntest(0);
   Int_t nvar(GetNvar());
   Int_t nlayers(fNlayers);
   Int_t *nodes = new Int_t[nlayers];
   Int_t ncycles(fNcycles);

   for (Int_t i=0; i<nlayers; i++) nodes[i] = fNodes[i]; // full copy of class member
   
   // please check
#ifndef R__WIN32
   Train_nn( &dumDat, &dumDat, &ntrain, &ntest, &nvar, &nlayers, nodes, &ncycles );
#else
   fLogger << kWARNING << "<Train> sorry CFMlpANN is not yet implemented on Windows" << Endl;
#endif  


   //    cout << "Weights after training:" << endl;
   //    cout << "=======================" << endl;
   //    PrintWeights(cout);

   delete [] nodes;
}

//_______________________________________________________________________
Double_t TMVA::MethodCFMlpANN::GetMvaValue()
{
   // returns CFMlpANN output (normalised within [0,1])
   Bool_t isOK = kTRUE;
   static vector<Double_t> inputVec = vector<Double_t>( GetNvar() );

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) 
      inputVec[ivar] = GetEventValNormalized(ivar);

   Double_t myMVA = EvalANN( &inputVec, isOK );
   if (!isOK) fLogger << kFATAL << "<EvalANN> returns (!isOK) for event " << Endl;

   //   delete inputVec;
   return myMVA;
}

//_______________________________________________________________________
Double_t TMVA::MethodCFMlpANN::EvalANN( vector<Double_t>* inVar, Bool_t& isOK )
{
   // evaluates NN value as function of input variables
   Double_t* xeev = new Double_t[GetNvar()];

   // hardcopy
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      xeev[ivar] = (*inVar)[ivar];
   }
  
   // ---- now apply the weights: get NN output
   isOK = kTRUE;
   for (Int_t jvar=0; jvar<GetNvar(); jvar++) {

      if (fVarn_1.xmax[jvar] < xeev[jvar]    ) xeev[jvar] = fVarn_1.xmax[jvar];
      if (fVarn_1.xmin[jvar] > xeev[jvar]    ) xeev[jvar] = fVarn_1.xmin[jvar];
      if (fVarn_1.xmax[jvar] == fVarn_1.xmin[jvar]) {
         isOK = kFALSE;
         xeev[jvar] = 0;
      }
      else {
         xeev[jvar] = xeev[jvar] - ((fVarn_1.xmax[jvar] + fVarn_1.xmin[jvar])/2);    
         xeev[jvar] = xeev[jvar] / ((fVarn_1.xmax[jvar] - fVarn_1.xmin[jvar])/2);    
      }
   }
    
   NN_ava( xeev );

   delete [] xeev;

   Double_t retVal = 0.5*(1.0 + fYNN[fParam_1.layerm-1][0]);

   return retVal;
}

//_______________________________________________________________________
void  TMVA::MethodCFMlpANN::NN_ava( Double_t* xeev )
{  
   // auxiliary functions
   //   cout << "Layer 0:  ";
   for (Int_t ivar=0; ivar<neur_1.neuron[0]; ivar++) {
      fYNN[0][ivar] = xeev[ivar];
      // cout << setw(15) << fYNN[0][ivar] << "      ";
   }
   //   cout << endl;

   for (Int_t layer=1; layer<fParam_1.layerm; layer++) {
      for (Int_t j=1; j<=neur_1.neuron[layer]; j++) {

         Double_t x = Ww_ref(neur_1.ww, layer+1,j); // init with the bias layer

         for (Int_t k=1; k<=neur_1.neuron[layer-1]; k++) { // neurons of originating layer
            x += fYNN[layer-1][k-1]*W_ref(neur_1.w, layer+1, j, k);
         }
         fYNN[layer][j-1] = NN_fonc( layer, x );
      }
   }  
}

//_______________________________________________________________________
Double_t TMVA::MethodCFMlpANN::NN_fonc( Int_t i, Double_t u ) const
{
   // activation function
   Double_t f(0);
  
   if      (u/fDel_1.temp[i] >  170) f = +1;
   else if (u/fDel_1.temp[i] < -170) f = -1;
   else {
      Double_t yy = TMath::Exp(-u/fDel_1.temp[i]);
      f  = (1 - yy)/(1 + yy);
   }

   return f;
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::ReadWeightsFromStream( istream & istr )
{
   // read back the weight from the training from file (stream)
   TString var;

   // read number of variables and classes
   Int_t nva(0), lclass(0);
   istr >> nva >> lclass;
      
   if (GetNvar() != nva) // wrong file
      fLogger << kFATAL << "<ReadWeightsFromFile> mismatch in number of variables" << Endl;

   // number of output classes must be 2
   if (lclass != 2) // wrong file
      fLogger << kFATAL << "<ReadWeightsFromFile> mismatch in number of classes" << Endl;
          
   // check that we are not at the end of the file
   if (istr.eof( ))
      fLogger << kFATAL << "<ReadWeightsFromStream> reached EOF prematurely " << Endl;



   // read extrema of input variables
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) 
      istr >> fVarn_1.xmax[ivar] >> fVarn_1.xmin[ivar];
            
   // read number of layers (sum of: input + output + hidden)
   istr >> fParam_1.layerm;
            
   fYNN = new Double_t* [fParam_1.layerm];
   for (Int_t layer=0; layer<fParam_1.layerm; layer++) {              
      // read number of neurons for each layer
      istr >> neur_1.neuron[layer];
      fYNN [layer] = new Double_t [neur_1.neuron[layer]];
   }
            
   // to read dummy lines
   const Int_t nchar( 100 );
   char* dumchar = new char[nchar];
            
   // read weights
   for (Int_t layer=1; layer<fParam_1.layerm-1; layer++) {
              
      Int_t nq = neur_1.neuron[layer]/10;
      Int_t nr = neur_1.neuron[layer] - nq*10;
              
      Int_t kk(0);
      if (nr==0) kk = nq;
      else       kk = nq+1;
              
      for (Int_t k=1; k<=kk; k++) {
         Int_t jmin = 10*k - 9;
         Int_t jmax = 10*k;
         if (neur_1.neuron[layer]<jmax) jmax = neur_1.neuron[layer];
         for (Int_t j=jmin; j<=jmax; j++) 
            istr >> Ww_ref(neur_1.ww, layer+1, j);
         for (Int_t i=1; i<=neur_1.neuron[layer-1]; i++) 
            for (Int_t j=jmin; j<=jmax; j++) 
               istr >> W_ref(neur_1.w, layer+1, j, i);

         // skip two empty lines
         istr.getline( dumchar, nchar );
      }
   }
   for (Int_t layer=0; layer<fParam_1.layerm; layer++) {
              
      // skip 2 empty lines
      istr.getline( dumchar, nchar );
      istr.getline( dumchar, nchar );
              
      istr >> fDel_1.temp[layer];
   }            

   // sanity check
   if (GetNvar() != neur_1.neuron[0]) {
      fLogger << kFATAL << "<ReadWeightsFromFile> mismatch in zeroth layer:"
              << GetNvar() << " " << neur_1.neuron[0] << Endl;
   }

   fNlayers = fParam_1.layerm;

   
   //    cout << "Weights after reading:" << endl;
   //    cout << "======================" << endl;
   //    PrintWeights(cout);

}

//_______________________________________________________________________
Int_t TMVA::MethodCFMlpANN::DataInterface( Double_t* /*tout2*/, Double_t*  /*tin2*/, 
                                           Int_t* /* icode*/, Int_t*  /*flag*/, 
                                           Int_t*  /*nalire*/, Int_t* nvar, 
                                           Double_t* xpg, Int_t* iclass, Int_t* ikend )
{
   // data interface function 
   
   // icode and ikend are dummies needed to match f2c mlpl3 functions
   *ikend = 0; 

   // retrieve pointer to current object (CFMlpANN must be a singleton class!)
   TMVA::MethodCFMlpANN* opt = TMVA::MethodCFMlpANN::This();

   // sanity checks
   if (0 == xpg) {
      fLogger << kFATAL << "ERROR in MethodCFMlpANN_DataInterface zero pointer xpg" << Endl;
   }
   if (*nvar != opt->GetNvar()) {
      fLogger << kFATAL << "ERROR in MethodCFMlpANN_DataInterface mismatch in num of variables: "
              << *nvar << " " << opt->GetNvar() << Endl;
   }

   // fill variables
   *iclass = (int)opt->GetClass( TMVA::MethodCFMlpANN_nsel );
   for (Int_t ivar=0; ivar<opt->GetNvar(); ivar++) 
      xpg[ivar] = (double)opt->GetData( TMVA::MethodCFMlpANN_nsel, ivar );

   ++TMVA::MethodCFMlpANN_nsel;

   return 0;
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::WriteWeightsToStream( std::ostream & o ) const
{
   // write number of variables and classes
   o << fParam_1.nvar << "    " << fParam_1.lclass << endl;
   
   // number of output classes must be 2
   if (fParam_1.lclass != 2) // wrong file
      fLogger << kFATAL << "<WriteNNWeightsToStream> mismatch in number of classes" << Endl;


   // write extrema of input variables
   for (Int_t ivar=0; ivar<fParam_1.nvar; ivar++) 
      o << fVarn_1.xmax[ivar] << "   " << fVarn_1.xmin[ivar] << endl;
        
   // write number of layers (sum of: input + output + hidden)
   o << fParam_1.layerm << endl;
        
   for (Int_t layer=0; layer<fParam_1.layerm; layer++) {              
      // write number of neurons for each layer
      o << neur_1.neuron[layer] << "     ";
   }
   o << endl;
        
   // write weights
   for (Int_t layer=1; layer<=fParam_1.layerm-1; layer++) { 
          
      Int_t nq = neur_1.neuron[layer]/10;
      Int_t nr = neur_1.neuron[layer] - nq*10;
          
      Int_t kk(0);
      if (nr==0) kk = nq;
      else       kk = nq+1;
          
      for (Int_t k=1; k<=kk; k++) {
         Int_t jmin = 10*k - 9;
         Int_t jmax = 10*k;
         Int_t i, j;
         if (neur_1.neuron[layer]<jmax) jmax = neur_1.neuron[layer];
         for (j=jmin; j<=jmax; j++) {

            // o << neur_1.ww[j*max_nLayers_ + layer - 6] << "   ";
            //define Ww_ref(a_1,a_2)    wwNN[(a_2)*max_nLayers_ + a_1 - 7]
            o << Ww_ref(neur_1.ww, layer+1, j);
         }
         o << endl;
         for (i=1; i<=neur_1.neuron[layer-1]; i++) {
            for (j=jmin; j<=jmax; j++) {
               // o << neur_1.w[(i*max_nNodes_ + j)*max_nLayers_ + layer - 186] << "   ";
               //#define W_ref(a_1,a_2,a_3) wNN [((a_3)*max_nNodes_ + (a_2))*max_nLayers_ + a_1 - 187]

               o << W_ref(neur_1.w, layer+1, j, i);

            }
            o << endl;
         }
            
         // skip two empty lines
         o << endl << endl;
      }
   }
   for (Int_t layer=0; layer<fParam_1.layerm; layer++) {
          
      // skip 2 empty lines
      o << endl << endl;          
      o << fDel_1.temp[layer] << endl;
   }      
}

//_______________________________________________________________________
void TMVA::MethodCFMlpANN::PrintWeights( std::ostream & o ) const
{
   // write the weights of the neural net

   // write number of variables and classes
   o << "Number of vars " << fParam_1.nvar << endl;
   o << "Output nodes   " << fParam_1.lclass << endl;
   
   // write extrema of input variables
   for (Int_t ivar=0; ivar<fParam_1.nvar; ivar++) 
      o << "Var " << ivar << " [" << fVarn_1.xmin[ivar] << " - " << fVarn_1.xmax[ivar] << "]" << endl;
        
   // write number of layers (sum of: input + output + hidden)
   o << "Number of layers " << fParam_1.layerm << endl;
   
   o << "Nodes per layer ";
   for (Int_t layer=0; layer<fParam_1.layerm; layer++)
      // write number of neurons for each layer
      o << neur_1.neuron[layer] << "     ";   
   o << endl;
        
   // write weights
   for (Int_t layer=1; layer<=fParam_1.layerm-1; layer++) { 
          
      Int_t nq = neur_1.neuron[layer]/10;
      Int_t nr = neur_1.neuron[layer] - nq*10;
          
      Int_t kk(0);
      if (nr==0) kk = nq;
      else       kk = nq+1;
          
      for (Int_t k=1; k<=kk; k++) {
         Int_t jmin = 10*k - 9;
         Int_t jmax = 10*k;
         Int_t i, j;
         if (neur_1.neuron[layer]<jmax) jmax = neur_1.neuron[layer];
         for (j=jmin; j<=jmax; j++) {

            //o << neur_1.ww[j*max_nLayers_ + layer - 6] << "   ";
            o << Ww_ref(neur_1.ww, layer+1, j) << "   ";

         }
         o << endl;
         //for (i=1; i<=neur_1.neuron[layer-1]; i++) {
         for (i=1; i<=neur_1.neuron[layer-1]; i++) {
            for (j=jmin; j<=jmax; j++) {
               //               o << neur_1.w[(i*max_nNodes_ + j)*max_nLayers_ + layer - 186] << "   ";
               o << W_ref(neur_1.w, layer+1, j, i) << "   ";
            }
            o << endl;
         }
            
         // skip two empty lines
         o << endl;
      }
   }
   for (Int_t layer=0; layer<fParam_1.layerm; layer++) {
      o << "Del.temp in layer " << layer << " :  " << fDel_1.temp[layer] << endl;
   }      
}

