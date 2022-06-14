/// \file
/// \ingroup tutorial_fit
/// \notebook -nodraw
/// Fit a 5d hyperplane by n points, using the linear fitter directly
///
/// This macro shows some features of the TLinearFitter class
/// A 5-d hyperplane is fit, a constant term is assumed in the hyperplane
/// equation `(y = a0 + a1*x0 + a2*x1 + a3*x2 + a4*x3 + a5*x4)`
///
/// \macro_output
/// \macro_code
///
/// \author Anna Kreshuk

#include "TLinearFitter.h"
#include "TF1.h"
#include "TRandom.h"

void fitLinear2()
{
   int n=100;
   int i;
   TRandom randNum;
   TLinearFitter *lf=new TLinearFitter(5);

   //The predefined "hypN" functions are the fastest to fit
   lf->SetFormula("hyp5");

   double *x=new double[n*10*5];
   double *y=new double[n*10];
   double *e=new double[n*10];

   //Create the points and put them into the fitter
   for (i=0; i<n; i++){
      x[0 + i*5] = randNum.Uniform(-10, 10);
      x[1 + i*5] = randNum.Uniform(-10, 10);
      x[2 + i*5] = randNum.Uniform(-10, 10);
      x[3 + i*5] = randNum.Uniform(-10, 10);
      x[4 + i*5] = randNum.Uniform(-10, 10);
      e[i] = 0.01;
      y[i] = 4*x[0+i*5] + x[1+i*5] + 2*x[2+i*5] + 3*x[3+i*5] + 0.2*x[4+i*5]  + randNum.Gaus()*e[i];
   }

   //To avoid copying the data into the fitter, the following function can be used:
   lf->AssignData(n, 5, x, y, e);
   //A different way to put the points into the fitter would be to use
   //the AddPoint function for each point. This way the points are copied and stored
   //inside the fitter

   //Perform the fitting and look at the results
   lf->Eval();
   TVectorD params;
   TVectorD errors;
   lf->GetParameters(params);
   lf->GetErrors(errors);
   for (int i=0; i<6; i++)
      printf("par[%d]=%f+-%f\n", i, params(i), errors(i));
   double chisquare=lf->GetChisquare();
   printf("chisquare=%f\n", chisquare);


   //Now suppose you want to add some more points and see if the parameters will change
   for (i=n; i<n*2; i++) {
      x[0+i*5] = randNum.Uniform(-10, 10);
      x[1+i*5] = randNum.Uniform(-10, 10);
      x[2+i*5] = randNum.Uniform(-10, 10);
      x[3+i*5] = randNum.Uniform(-10, 10);
      x[4+i*5] = randNum.Uniform(-10, 10);
      e[i] = 0.01;
      y[i] = 4*x[0+i*5] + x[1+i*5] + 2*x[2+i*5] + 3*x[3+i*5] + 0.2*x[4+i*5]  + randNum.Gaus()*e[i];
   }

   //Assign the data the same way as before
   lf->AssignData(n*2, 5, x, y, e);
   lf->Eval();
   lf->GetParameters(params);
   lf->GetErrors(errors);
   printf("\nMore Points:\n");
   for (int i=0; i<6; i++)
      printf("par[%d]=%f+-%f\n", i, params(i), errors(i));
   chisquare=lf->GetChisquare();
   printf("chisquare=%.15f\n", chisquare);


   //Suppose, you are not satisfied with the result and want to try a different formula
   //Without a constant:
   //Since the AssignData() function was used, you don't have to add all points to the fitter again
   lf->SetFormula("x0++x1++x2++x3++x4");

   lf->Eval();
   lf->GetParameters(params);
   lf->GetErrors(errors);
   printf("\nWithout Constant\n");
   for (int i=0; i<5; i++)
     printf("par[%d]=%f+-%f\n", i, params(i), errors(i));
   chisquare=lf->GetChisquare();
   printf("chisquare=%f\n", chisquare);

   //Now suppose that you want to fix the value of one of the parameters
   //Let's fix the first parameter at 4:
   lf->SetFormula("hyp5");
   lf->FixParameter(1, 4);
   lf->Eval();
   lf->GetParameters(params);
   lf->GetErrors(errors);
   printf("\nFixed Constant:\n");
   for (i=0; i<6; i++)
      printf("par[%d]=%f+-%f\n", i, params(i), errors(i));
   chisquare=lf->GetChisquare();
   printf("chisquare=%.15f\n", chisquare);

   //The fixed parameters can then be released by the ReleaseParameter method
   delete lf;

}

