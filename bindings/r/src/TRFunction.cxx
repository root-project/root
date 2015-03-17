/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRFunction.h>
//______________________________________________________________________________
/* Begin_Html
<center><h2>TRFunction class</h2></center>

<p>
The TRFunction class lets you pass ROOT's functions to R's environment<br>
</p>
<p>
The next example was based in <br>
<a href="http://root.cern.ch/root/html/tutorials/fit/NumericalMinimization.C.html">
http://root.cern.ch/root/html/tutorials/fit/NumericalMinimization.C.html
</a><br>
<a href="http://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html">
http://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html</a><br>

</p>
End_Html
Begin_Latex

Let f(x,y)=(x-1)^{2} + 100(y-x^{2})^{2}, which is called the Rosenbrock
function.

It's known that this function has a minimum when y = x^{2} , and x = 1.
Let's get the minimum using R's optim package through ROOTR's interface.
In the code this function was called "Double_t RosenBrock(const TVectorD xx )", because for
optim, the input in your function deﬁnition must be a single vector.


The Gradient is formed by
#frac{#partial f}{#partial x} =  -400x(y - x^{2}) - 2(1 - x)

#frac{#partial f}{#partial y} =  200(y - x^{2});

The "TVectorD RosenBrockGrad(const TVectorD xx )" function
must have  a single vector as the argument a it will return a single vetor.

End_Latex
Begin_Html
<hr>
End_Html

#include<TRInterface.h>

//in the next function the pointer *double must be changed by TVectorD, because the pointer has no
//sense in R's environment.
Double_t RosenBrock(const TVectorD xx )
{
  const Double_t x = xx[0];
  const Double_t y = xx[1];
  const Double_t tmp1 = y-x*x;
  const Double_t tmp2 = 1-x;
  return 100*tmp1*tmp1+tmp2*tmp2;
}

TVectorD RosenBrockGrad(const TVectorD xx )
{
  const Double_t x = xx[0];
  const Double_t y = xx[1];
  TVectorD grad(2);
  grad[0]=-400 * x * (y - x * x) - 2 * (1 - x);
  grad[1]=200 * (y - x * x);
  return grad;
}


void Minimization()
{
 ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
 //passing RosenBrock function to R
 r["RosenBrock"]<<ROOT::R::TRFunction(RosenBrock);

 //passing RosenBrockGrad function to R
 r["RosenBrockGrad"]<<ROOT::R::TRFunction(RosenBrockGrad);

 //the option "method" could be "Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN","Brent"
 //the option "control" lets you put some constraints like:
 //"maxit" The maximum number of iterations
 //"abstol" The absolute convergence tolerance.
 //"reltol" Relative convergence tolerance.
 r<<"result <- optim( c(0.01,0.01), RosenBrock,method='BFGS',control = list(maxit = 1000000) )";

 //Getting results from R
 TVectorD  min=r.ParseEval("result$par");

 std::cout.precision(8);
 //printing results
 std::cout<<"-----------------------------------------"<<std::endl;
 std::cout<<"Minimum x="<<min[0]<<" y="<<min[1]<<std::endl;
 std::cout<<"Value at minimum ="<<RosenBrock(min)<<std::endl;

 //using the gradient
 r<<"optimHess(result$par, RosenBrock, RosenBrockGrad)";
 r<<"hresult <- optim(c(-1.2,1), RosenBrock, NULL, method = 'BFGS', hessian = TRUE)";
 //getting the minimum calculated with the gradient
 TVectorD  hmin=r.ParseEval("hresult$par");

 //printing results
 std::cout<<"-----------------------------------------"<<std::endl;
 std::cout<<"Minimization with the Gradient"<<endl;
 std::cout<<"Minimum x="<<hmin[0]<<" y="<<hmin[1]<<std::endl;
 std::cout<<"Value at minimum ="<<RosenBrock(hmin)<<std::endl;

}
*/


using namespace ROOT::R;
ClassImp(TRFunction)


//______________________________________________________________________________
TRFunction::TRFunction(): TObject()
{
   f = NULL;
}

//______________________________________________________________________________
TRFunction::TRFunction(const TRFunction &fun): TObject(fun)
{
   f = fun.f;
}

