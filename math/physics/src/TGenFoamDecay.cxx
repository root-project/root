/*************************************************************************
* Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TGenFoamDecay.h"


//const Int_t kMAXP = 18;



Double_t TGenFoamDecay::Density(int nDim, Double_t *Xarg)
{
	
	//queue for random numbers
    queue<double> rndQueue;
	
	//put rnd numbers into queue
	for( int i = 0; i < 3*fNt-4; i++)
	{
		rndQueue.push( Xarg[i] );
	}  
	
	//make decay and take d(LIPS)
	double wtdecay =  _decay.Generate( rndQueue );
		
	//get out particles
	TLorentzVector pf[fNt];
	for( int i = 0; i < fNt; i++ )
	{
		pf[i] = *(_decay.GetDecay( i ));
	}
		
	//calculate integrand
	double integrand = Integrand( fNt, pf );	
	
	return wtdecay * integrand;
}

//__________________________________________________________________________________________________
Double_t TGenFoamDecay::Integrand( int fNt, TLorentzVector * pf )
{
	return 1.0; //default and probably overloaded for matrix element
} 

//__________________________________________________________________________________________________
TGenFoamDecay::TGenFoamDecay(const TGenFoamDecay &gen)
{
   //copy constructor
   fNt      = gen.fNt;
   fTeCmTm  = gen.fTeCmTm;
   fBeta[0] = gen.fBeta[0];
   fBeta[1] = gen.fBeta[1];
   fBeta[2] = gen.fBeta[2];
   _decay   = gen._decay;
   _foam    = gen._foam;
   _pseRan  = gen._pseRan;
   for (Int_t i=0;i<fNt;i++) 
   {
      fMass[i]   = gen.fMass[i];
      fDecPro[i] = gen.fDecPro[i];
   }
}

//__________________________________________________________________________________________________
TGenFoamDecay& TGenFoamDecay::operator=(const TGenFoamDecay &gen)
{
   // Assignment operator
   TObject::operator=(gen);
   fNt      = gen.fNt;
   fTeCmTm  = gen.fTeCmTm;
   fBeta[0] = gen.fBeta[0];
   fBeta[1] = gen.fBeta[1];
   fBeta[2] = gen.fBeta[2];
   _decay   = gen._decay;
   _foam    = gen._foam;
   _pseRan  = gen._pseRan;
   for (Int_t i=0;i<fNt;i++) 
   {
      fMass[i]   = gen.fMass[i];
      fDecPro[i] = gen.fDecPro[i];
   }
   return *this;
}

//__________________________________________________________________________________________________
Double_t TGenFoamDecay::Generate(void)
{

	_foam->MakeEvent();  
	
	return _foam->GetMCwt( );

}

//__________________________________________________________________________________
TLorentzVector *TGenFoamDecay::GetDecay(Int_t n) 
{ 
   
   if (n>fNt) return 0;
   
   //return Lorentz vector corresponding to decay of n-th particle
   return _decay.GetDecay( n );
}

//_____________________________________________________________________________________
Bool_t TGenFoamDecay::SetDecay(TLorentzVector &P, Int_t nt, 
   const Double_t *mass) 
{

   kMAXP = nt;

   Int_t n;
   fNt = nt;
   if (fNt<2 || fNt>18) return kFALSE;  // no more then 18 particle


   fTeCmTm = P.Mag();           // total energy in C.M. minus the sum of the masses
   for (n=0;n<fNt;n++) {
      fMass[n]  = mass[n];
      fTeCmTm  -= mass[n];
   }

   if (fTeCmTm<=0) return kFALSE;    // not enough energy for this decay

   _decay.SetDecay(P, fNt, fMass);  //set decay to TDecay

	// initialize FOAM
	//=========================================================
	if (Chat > 0 )
	{
		cout<<"*****   Foam version "<< _foam->GetVersion() <<"    *****"<<endl;
	}
	_foam->SetkDim(        3*fNt-4);      // Mandatory!!!
	_foam->SetnCells(      nCells);    // optional
	_foam->SetnSampl(      nSampl);    // optional
	_foam->SetnBin(        nBin);      // optional
	_foam->SetOptRej(      OptRej);    // optional
	_foam->SetOptDrive(    OptDrive);  // optional
	_foam->SetEvPerBin(    EvPerBin);  // optional
	_foam->SetChat(        Chat);      // optional
	//===============================
	_foam->SetRho(this);
	_foam->SetPseRan(&_pseRan);
	
	// Initialize simulator
	_foam->Initialize(); 

   return kTRUE; 
}

//_____________________________________________________________________________________
void  TGenFoamDecay::Finalize( void )
{
	Double_t MCresult,MCerror;
	Double_t eps = 0.0005;
	Double_t Effic, WtMax, AveWt, Sigma;
	Double_t IntNorm, Errel;
	_foam->Finalize(   IntNorm, Errel);     // final printout
	_foam->GetIntegMC( MCresult, MCerror);  // get MC intnegral
	_foam->GetWtParams(eps, AveWt, WtMax, Sigma); // get MC wt parameters
	long nCalls=_foam->GetnCalls();
	Effic=0; if(WtMax>0) Effic=AveWt/WtMax;
	cout << "================================================================" << endl;
	cout << " MCresult= " << MCresult << " +- " << MCerror << " RelErr= "<< MCerror/MCresult << endl;
	cout << " Dispersion/<wt>= " << Sigma/AveWt << endl;
	cout << "      <wt>/WtMax= " << Effic <<",    for epsilon = "<<eps << endl;
	cout << " nCalls (initialization only) =   " << nCalls << endl;
	cout << "================================================================" << endl;	
}

//_____________________________________________________________________________________
void TGenFoamDecay::GetIntegMC(Double_t & integral, Double_t & error)
{
	_foam->	GetIntegMC( integral, error);
}
