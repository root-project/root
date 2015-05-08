//
//  TF1NormSum.cxx
//  
//
//  Created by Aurélie Flandi on 27.08.14.
//
//
#include "Riostream.h"
#include "TROOT.h"
#include "TMath.h"
#include "TF1NormSum.h"
#include "TClass.h"
#include "Math/WrappedFunction.h"
#include "Math/WrappedTF1.h"
#include "Math/BrentMinimizer1D.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "Math/MinimizerOptions.h"


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



void TF1NormSum::InitializeDataMembers(const std::vector <std::shared_ptr < TF1 >> &functions, const std::vector <Double_t> &coeffs)
{
   
   fFunctions       = functions;
   fCoeffs          = coeffs;
   fNOfFunctions    = functions.size();
   fNOfParams       = std::vector < Int_t     > (fNOfFunctions);
   fParams          = std::vector < Double_t* > (fNOfFunctions);
   fCstIndexes      = std::vector < Int_t     > (fNOfFunctions);
   fNOfNonCstParams = std::vector < Int_t     > (fNOfFunctions);
   fParNames        = std::vector<TString> (fNOfFunctions);
   fParNames.reserve(3*fNOfFunctions);  // enlarge capacity for function parameters

   for (unsigned int n=0; n < fNOfFunctions; n++)
   {
      //normalize the functions if it is not already done
      if (!fFunctions[n] -> IsEvalNormalized())  fFunctions[n]  -> SetNormalized(true);
      fNOfParams[n]       = fFunctions[n] -> GetNpar();
      fNOfNonCstParams[n] = fNOfParams[n];
      fCstIndexes[n]      = fFunctions[n] -> GetParNumber("Constant");//return -1 if there is no constant parameter
      //std::cout << " cst index of function " << n << " : " << fCstIndexes[n] << std::endl;
      //std::cout << "nofparam of function " << n <<" : " << fNOfParams[n] << std::endl;
      fParNames[n] = TString::Format("Coeff%d",n);
      if (fCstIndexes[n]!= -1)                                        //if there exists a constant parameter
      {
         fFunctions[n] -> FixParameter(fCstIndexes[n], 1.);          //fixes the parameters called "Constant" to 1
         fNOfNonCstParams[n] -= 1;                                   //the number of non fixed parameter thus decreases
         std::vector <Double_t> temp(fNOfNonCstParams[n]);
         int k = 0;                                                  //index for the temp arry, k wil go form 0 until fNofNonCstParameter
         for (int i=0; i<fNOfParams[n]; i++)                         //go through all the parameter to
         {
            if (i==fCstIndexes[n])   continue;                      //go to next step if this is the constant parameter
            temp[k] = fFunctions[n] -> GetParameter(i);             //takes all the internal parameters instead of the constant one
            fParNames.push_back(  fFunctions[n] -> GetParName(i) ); 
            k++;
         }
         fParams[n] = temp.data();
      }
      else { 
         fParams[n] = fFunctions[n] -> GetParameters();
         for (int i=0; i<fNOfParams[n]; i++)                        //go through all the parameter to
         {
            fParNames.push_back( fFunctions[n] -> GetParName(i) ); 
         }
      }
   }

   FixDuplicateNames(fParNames.begin()+fNOfFunctions, fParNames.end());
}
TF1NormSum::TF1NormSum()
{
   fNOfFunctions  = 0;
   fFunctions     = std::vector< std::shared_ptr < TF1 >>(0) ;     // Vector of size fNOfFunctions containing TF1 functions
   fCoeffs        = std::vector < Double_t  >(0) ;        // Vector of size fNOfFunctions containing coefficients in front of each function
   fNOfParams     = std::vector < Int_t     >(0) ;     // Vector of size fNOfFunctions containing number of parameters for each function (does not contai the coefficients!)
   fNOfNonCstParams  = std::vector < Int_t   >(0) ;
   fParams = std::vector < Double_t* > (0);        // Vector of size [fNOfFunctions][fNOfNonCstParams] containing an array of (non constant) parameters
   // (non including coefficients) for each function
   fCstIndexes = std::vector < Int_t     > (0);
   

}

//_________________________________________________________________
TF1NormSum::TF1NormSum(const std::vector <TF1*> &functions, const std::vector <Double_t> &coeffs)
{
   std::vector <std::shared_ptr < TF1 > >f;
   for (unsigned int i = 0; i<functions.size(); i++)
   {
    f[i] = std::shared_ptr < TF1 >((TF1*)functions[i]->Clone());
   }
   
   InitializeDataMembers(f,coeffs);
}

//______________________________________________________________________________
TF1NormSum::TF1NormSum(TF1* function1, TF1* function2, Double_t coeff1, Double_t coeff2)
{
   // TF1NormSum constructor taking 2 functions, and 2 coefficients (if not equal to 1)
   
   std::vector < std::shared_ptr < TF1 > > functions(2);
   std::vector < Double_t > coeffs(2);
   std::shared_ptr < TF1 > f1((TF1*)function1->Clone());
   std::shared_ptr < TF1 > f2((TF1*)function2->Clone());
   
   functions       = {f1, f2};
   coeffs          = {coeff1,    coeff2};
   
   InitializeDataMembers(functions, coeffs);
}

//______________________________________________________________________________
TF1NormSum::TF1NormSum(TF1* function1, TF1* function2, TF1* function3, Double_t coeff1, Double_t coeff2, Double_t coeff3)
{
   // TF1NormSum constructor taking 3 functions, and 3 coefficients (if not equal to 1)
   
   std::vector < std::shared_ptr < TF1 > > functions(3);
   std::vector < Double_t > coeffs(3);
   std::shared_ptr < TF1 > f1((TF1*)function1->Clone());
   std::shared_ptr < TF1 > f2((TF1*)function2->Clone());
   std::shared_ptr < TF1 > f3((TF1*)function3->Clone());
   
   
   functions       = {f1, f2, f3};
   coeffs          = {coeff1,    coeff2,    coeff3};
   
   InitializeDataMembers(functions, coeffs);
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
   InitializeDataMembers(functions, coeffs);
   
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
   return sum;
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
   std::vector <std::vector <Double_t> > noncstparams(fNOfFunctions);
   std::vector <std::vector <Double_t> > totalparams (fNOfFunctions);
   Int_t    offset     = 0;
   Double_t fixedvalue = 1.;
   Int_t k = 0;
   for (unsigned int n=0; n<fNOfFunctions; n++)
   {
      totalparams[n]  = std::vector < Double_t > (fNOfParams[n]);       // temptotal is used for the TF1::SetParameters, so doesn't contains coefficients, but does contain cst parameters
      noncstparams[n] = std::vector < Double_t > (fNOfNonCstParams[n]); // temp is used for the class member fParams, so does not contain the cst parameters
      if (n>0)    offset += fNOfNonCstParams[n-1];                      // offset to go along the list of parameters
      
      k = 0;                                                            // incrementer for temp
      for (int i=0; i<fNOfParams[n]; i++)
      {
         if (i == fCstIndexes[n])
         {
            totalparams[n][i] = 1. ;
            //std::cout << " constant param of function " << n << " no " <<  i << " = " << totalparams[n][i] << std::endl;
         }
         else
         {
            noncstparams[n][k] = params[k+fNOfFunctions+offset];
            totalparams[n][i]  = params[k+fNOfFunctions+offset];
            //std::cout << " params " << k+fNOfFunctions+offset << " = " << params[k+fNOfFunctions+offset] << std::endl;
            k++;
         }
      }
      fParams[n]    =  noncstparams[n].data();                          // fParams doesn't take the coefficients, and neither the cst parameters
      fFunctions[n] -> SetParameters(totalparams[n].data());
      
      fixedvalue = 1.;
      if (fFunctions[n] -> GetNumber() == 200)  fixedvalue = 0.;        // if this is an exponential, the fixed value is zero
      fFunctions[n] -> FixParameter(fCstIndexes[n], fixedvalue);
      // fFunctions[n]->Print();
      //std::cout << "coeff " << n << " : " << fCoeffs[n] << std::endl;
      
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
   for (unsigned int n=0; n<fNOfFunctions; n++)
   {
      nofparams += fNOfNonCstParams[n];
   }
   return nofparams + fNOfFunctions;                                   //fNOfFunctions for the  coefficientws
}
