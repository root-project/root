// example of an unbinned likelihood fit performed using the Minimizer interface

// data range
const double xmin = 0; 
const double xmax = 10; 


double ModelFunction(const double * x, const double *p) { 
   
   return p[0]*TMath::Gaus(x[0], p[1], p[2], true) + p[3]* std::exp(-x[0]/ p[4] )/ (p[4]  * (exp (-xmin/p[4]) - exp( - xmax/ p[4] ) ) );  
} 


TF1 * CreateModel() { 
   // create the model of the Fit. Represent it as a sum of an exponential 
   // plus a Gaussian peak


   TF1 * 
   

}


TH1 * GenerateData(int n = 10000)  { 

   const int NSignal = 100; 
   cont double MPeak = 3; 
   const double SigmaPeak = 0.1; 
   const double backshape 3; 

   // use automatic limits and buffer to store the data
   TH1D * h1 = new TH1D("h1","h1",100,1,0); 

   // generate signal da
   for (int i = 0; i < 

}


void NLLMinimization() { 

}
