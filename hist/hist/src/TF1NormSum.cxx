// @(#)root/hist:$Id$
// Authors: L. Moneta, A. Flandi   08/2014
//
//
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  ROOT  Team, CERN/PH-SFT                        *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
//  TF1NormSum.cxx
//
//
//
#include "TROOT.h"
#include "TClass.h"
#include "TMath.h"
#include "TF1NormSum.h"
#include "Math/WrappedFunction.h"
#include "Math/WrappedTF1.h"





//ClassImp(TF1NormSum)


// function to find and rename duplicate parameters with the same name

template<class Iterator> 
void FixDuplicateNames(Iterator begin, Iterator end) { 


   // make a map of values 

   std::multimap<TString, int > parMap; 
   for (Iterator it = begin; it !=  end; ++it) { 
      parMap.insert( std::make_pair( *it, std::distance(begin,it) ) );
   }
   for ( auto & elem : parMap) { 
      TString name = elem.first; 
      int n = parMap.count( name); 
      if (n > 1 ) {
          std::pair <std::multimap<TString,int>::iterator, std::multimap<TString,int>::iterator> ret;
          ret = parMap.equal_range(name);
          int i = 0; 
          for (std::multimap<TString,int>::iterator it=ret.first; it!=ret.second; ++it) {
             *(begin+it->second) = TString::Format("%s%d",name.Data(),++i);
          }
      }
   }

   // for (Iterator it = begin; it !=  end; ++it) 
   //    std::cout << *it << "  ";
   // std::cout << std::endl;

}



void TF1NormSum::InitializeDataMembers(const std::vector <std::shared_ptr < TF1 >> &functions, const std::vector <Double_t> &coeffs, Double_t scale)
{

   fScale           = scale; 
   fFunctions       = functions;
   fCoeffs          = coeffs;
   fNOfFunctions    = functions.size();
   fCstIndexes      = std::vector < Int_t     > (fNOfFunctions);
   fParNames        = std::vector<TString> (fNOfFunctions);
   fParNames.reserve(3*fNOfFunctions);  // enlarge capacity for function parameters

   for (unsigned int n=0; n < fNOfFunctions; n++)
   {
      int npar = fFunctions[n] -> GetNpar();
      fCstIndexes[n]      = fFunctions[n] -> GetParNumber("Constant");//return -1 if there is no constant parameter
      //std::cout << " cst index of function " << n << " : " << fCstIndexes[n] << std::endl;
      //std::cout << "nofparam of function " << n <<" : " << fNOfParams[n] << std::endl;
      fParNames[n] = TString::Format("Coeff%d",n);
      //printf("examing function %s \n",fFunctions[n]->GetName() );
      if (fCstIndexes[n]!= -1)                                        //if there exists a constant parameter
      {
         fFunctions[n] -> FixParameter(fCstIndexes[n], 1.);          //fixes the parameters called "Constant" to 1
         int k = 0;                                                  //index for the temp arry, k wil go form 0 until fNofNonCstParameter
         for (int i=0; i<npar; i++)                         //go through all the parameter to
         {
            if (i==fCstIndexes[n])   continue;                      //go to next step if this is the constant parameter
            fParNames.push_back(  fFunctions[n] -> GetParName(i) ); 
            k++;
         }
      }
      else { 
         for (int i=0; i < npar; i++)                        //go through all the parameter to
         {
            fParNames.push_back( fFunctions[n] -> GetParName(i) ); 
         }
      }
      //normalize the functions if it is not already done (do at the end so constant parameter is not zero)
      if (!fFunctions[n] -> IsEvalNormalized())  fFunctions[n]  -> SetNormalized(true);
   }

   FixDuplicateNames(fParNames.begin()+fNOfFunctions, fParNames.end());

}
TF1NormSum::TF1NormSum()
{
   fNOfFunctions  = 0;
   fScale         = 1.;
   fFunctions     = std::vector< std::shared_ptr < TF1 >>(0) ;     // Vector of size fNOfFunctions containing TF1 functions
   fCoeffs        = std::vector < Double_t  >(0) ;        // Vector of size fNOfFunctions containing coefficients in front of each function
   fCstIndexes = std::vector < Int_t     > (0);   
}

//_________________________________________________________________
TF1NormSum::TF1NormSum(const std::vector <TF1*> &functions, const std::vector <Double_t> &coeffs, Double_t scale)
{
   std::vector <std::shared_ptr < TF1 > >f;
   for (unsigned int i = 0; i<functions.size(); i++)
   {
    f[i] = std::shared_ptr < TF1 >((TF1*)functions[i]->Clone());
   }
   
   InitializeDataMembers(f,coeffs,scale);
}

//______________________________________________________________________________
TF1NormSum::TF1NormSum(TF1* function1, TF1* function2, Double_t coeff1, Double_t coeff2, Double_t scale)
{
   // TF1NormSum constructor taking 2 functions, and 2 coefficients (if not equal to 1)
   
   std::vector < std::shared_ptr < TF1 > > functions(2);
   std::vector < Double_t > coeffs(2);
   TF1 * fnew1 = 0;
   TF1 * fnew2 = 0;
   // need to use Copy because clone does not work for functor-based functions
   if (function1) { 
      fnew1 = (TF1*) function1->IsA()->New();
      function1->Copy(*fnew1); 
   }
   if (function2) { 
      fnew2 = (TF1*) function2->IsA()->New();
      function2->Copy(*fnew2); 
   }
   if (fnew1 == nullptr || fnew2 == nullptr)
      Fatal("TF1NormSum","Invalid input functions - Abort");

   std::shared_ptr < TF1 > f1( fnew1);
   std::shared_ptr < TF1 > f2( fnew2);
   
   functions       = {f1, f2};
   coeffs          = {coeff1,    coeff2};
   
   InitializeDataMembers(functions, coeffs,scale);
}

//______________________________________________________________________________
TF1NormSum::TF1NormSum(TF1* function1, TF1* function2, TF1* function3, Double_t coeff1, Double_t coeff2, Double_t coeff3, Double_t scale)
{
   // TF1NormSum constructor taking 3 functions, and 3 coefficients (if not equal to 1)
   
   std::vector < std::shared_ptr < TF1 > > functions(3);
   std::vector < Double_t > coeffs(3);
   TF1 * fnew1 = 0;
   TF1 * fnew2 = 0;
   TF1 * fnew3 = 0;
   if (function1) { 
      fnew1 = (TF1*) function1->IsA()->New();
      function1->Copy(*fnew1); 
   }
   if (function2) { 
      fnew2 = (TF1*) function2->IsA()->New();
      function2->Copy(*fnew2); 
   }
   if (function3) { 
      fnew3 = (TF1*) function3->IsA()->New();
      function3->Copy(*fnew2); 
   }
   if (!fnew1 || !fnew2  || !fnew3 )
      Fatal("TF1NormSum","Invalid input functions - Abort");

   std::shared_ptr < TF1 > f1( fnew1);
   std::shared_ptr < TF1 > f2( fnew2);
   std::shared_ptr < TF1 > f3( fnew3);
   
   
   functions       = {f1, f2, f3};
   coeffs          = {coeff1,    coeff2,    coeff3};
   
   InitializeDataMembers(functions, coeffs,scale);
}

//_________________________________________________________________
TF1NormSum::TF1NormSum(const TString &formula, Double_t xmin, Double_t xmax)
{
   //  TF1NormSum constructortaking any addition of formulas with coefficient or not
   // example 1 : 2.*expo + gauss + 0.5* gauss
   // example 2 : expo + 0.3*f1 if f1 is defined in the list of fucntions
   
   TF1::InitStandardFunctions();
   
   TObjArray *arrayall    = formula.Tokenize("*+");
   TObjArray *arraytimes  = formula.Tokenize("*") ;
   Int_t noffunctions     = (formula.Tokenize("+")) -> GetEntries();
   Int_t nofobj           = arrayall  -> GetEntries();
   Int_t nofcoeffs        = nofobj - noffunctions;
   
   std::vector < std::shared_ptr < TF1 >     > functions(noffunctions);
   std::vector < Double_t > coeffs(noffunctions);
   std::vector < TString  > funcstringall(nofobj);
   std::vector < Int_t    > indexsizetimes(nofcoeffs+1);
   std::vector < Bool_t   > isacoeff(nofobj);//1 is it is a coeff, 0 if it is a functions
   
   for (int i=0; i<nofobj; i++)
   {
      funcstringall[i] = ((TObjString*)((*arrayall)[i])) -> GetString();
      funcstringall[i].ReplaceAll(" ","");
   }
   //algorithm to determine which object is a coefficient and which is a function
   //uses the fact that the last item of funcstringtimes[i].Tokenize("+") is always a coeff.
   Int_t j = 0;
   Int_t k = 1;
   for (int i=0; i<nofcoeffs+1; i++)
   {
      indexsizetimes[i] = ( ( ( (TObjString*)(*arraytimes)[i] ) -> GetString() ).Tokenize("+") ) -> GetEntries();
      while (k < indexsizetimes[i])
      {
         isacoeff[k+j-1] = 0;
         k++;
      }
      j = j+indexsizetimes[i];
      if (j==nofobj)    isacoeff[j-1] = 0;    //the last one is never a coeff
      else              isacoeff[j-1] = 1;
      k = 1;
   }
   k = 0;
   for (int i=0; i<noffunctions; i++)
   {
      if (isacoeff[k]==0)
      {
         coeffs[i]    = 1.;
         TF1* f = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(funcstringall[k]));
         if (!f)   Error("TF1NormSum", "Function %s does not exist", funcstringall[k].Data());
         functions[i] = std::shared_ptr < TF1 > ((TF1*)f->Clone(TString::Format("function_%s_%d",funcstringall[k].Data(), i)));
         functions[i]->SetRange(xmin,xmax);
         k++;
      }
      else
      {
         coeffs[i]    = funcstringall[k].Atof();
         TF1* f  = (TF1*)(gROOT -> GetListOfFunctions() -> FindObject(funcstringall[k+1]));
         if (!f)   Error("TF1NormSum", "Function %s does not exist", funcstringall[k+1].Data());
         functions[i] = std::shared_ptr < TF1 >((TF1*)f->Clone(TString::Format("function_%s_%d",funcstringall[k+1].Data(), i) ));
         functions[i]->SetRange(xmin,xmax);
         k=k+2;
      }
   }
   InitializeDataMembers(functions, coeffs,1.);
   
   /*for (auto f : functions)
    {
    f->Print();
    }
    for (auto c : coeffs)
    {
    std::cout << "coeff " << c << std::endl;
    }*/
}

//_________________________________________________________________
double TF1NormSum::operator()(double* x, double* p)
{
   // Overload the parenthesis to add the functions
   if (p!=0)   TF1NormSum::SetParameters(p);                           // first refresh the parameters
   
   Double_t sum = 0.;
   for (unsigned int n=0; n<fNOfFunctions; n++)
   {
      sum += fCoeffs[n]*(fFunctions[n] -> EvalPar(x,0));
   }
   // normalize by a scale parameter (typically the bin width)
   return fScale * sum;
}

//_________________________________________________________________   
std::vector<double>  TF1NormSum::GetParameters() const {
   // return array of parameters

   std::vector<double> params(GetNpar() );
   int offset = 0;
   int nOfNonCstParams = 0;
   for (unsigned int n=0; n<fNOfFunctions; n++)
   {
      params[n] = fCoeffs[n];   // copy the coefficients
      offset += nOfNonCstParams;           // offset to go along the list of parameters
      int k = 0;
      for (int j = 0; j < fFunctions[n]->GetNpar(); ++j) {
         if (j != fCstIndexes[n]) {
            params[k+fNOfFunctions+offset] = fFunctions[n]->GetParameter(j);
            k++;
         }
      }
      nOfNonCstParams = k; 
   }
   return params;
}
//_________________________________________________________________
void TF1NormSum::SetParameters(const double* params)//params should have the size [fNOfFunctions][fNOfNonCstParams]
{
   // Initialize array of all parameters.
   // double *params must contains first an array of the coefficients, then an array of the parameters.

   for (unsigned int n=0; n<fNOfFunctions; n++)                         //initialization of the coefficients
   {
      fCoeffs[n] = params[n];
   }
   Int_t    offset     = 0;
   int k = 0;  // k indicates the nnumber of non-constant parameter per function
   for (unsigned int n=0; n<fNOfFunctions; n++)
   {
      bool equalParams = true;
      Double_t * funcParams = fFunctions[n]->GetParameters();
      int npar = fFunctions[n]->GetNpar();
      offset += k;      // offset to go along the list of parameters
      k = 0; // reset k value for next function
      for (int i = 0; i < npar; ++i) {
         // constant parameters can be only one
         if (i != fCstIndexes[n])
         {
            // check if they are equal
            equalParams &= (funcParams[i] == params[k+fNOfFunctions+offset] );
            funcParams[i] = params[k+fNOfFunctions+offset];
            k++;
         }
      }
      // update function integral if not equal
      if (!equalParams) fFunctions[n]->Update(); 
            
   }
}

//______________________________________________________________________________
void TF1NormSum::SetParameters(Double_t p0, Double_t p1, Double_t p2, Double_t p3, Double_t p4,
                               Double_t p5, Double_t p6, Double_t p7, Double_t p8, Double_t p9, Double_t p10)
{
   // Initialize array of all parameters.
   // Overload the TF1::SetParameters() method.
   // A maximum of 10 parameters must be used, with first the coefficients, then the parameters
   
   const double params[] = {p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10};
   TF1NormSum::SetParameters(params);
   
}
//return the number of (non constant) paramters including the coefficients: for 2 functions: c1,c2,p0,p1,p2,p3...
Int_t TF1NormSum::GetNpar() const
{
   Int_t nofparams = 0;
   for (unsigned int n=0; n<fNOfFunctions; ++n)
   {
      nofparams += fFunctions[n]->GetNpar();
      if (fCstIndexes[n] >= 0) nofparams -= 1;
   }
   return nofparams + fNOfFunctions;                                   //fNOfFunctions for the  coefficientws
}
