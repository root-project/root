/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This executable gives an example of a very simple use of the genetic algorithm
/// of TMVA.
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Executable: TMVAGAexample
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker


#include <iostream> // Stream declarations
#include <vector>

#include "TMVA/GeneticAlgorithm.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/IFitterTarget.h"

using namespace std;

namespace TMVA {


class MyFitness : public IFitterTarget {
    public:
       MyFitness() : IFitterTarget() {
       }

       // the fitness-function goes here
       // the factors are optimized such that the return-value of this function is minimized
       // take care!! the fitness-function must never fail, .. means: you have to prevent
       // the function from reaching undefined values (such as x=0 for 1/x or so)
       //
       // HINT: to use INTEGER variables, it is sufficient to cast the "factor" in the fitness-function
       // to (int). In this case the variable-range has to be chosen +1 ( to get 0..5, take Interval(0,6) )
       // since the introduction of "Interval" ranges can be defined with a third parameter
       // which gives the number of bins within the interval. With that technique discrete values
       // can be achieved easier. The random selection out of this discrete numbers is completely uniform.
       //
       Double_t EstimatorFunction( std::vector<Double_t> & factors ){
           //return (10.- (int)factors.at(0) *factors.at(1) + (int)factors.at(2));
           return (10.- factors.at(0) *factors.at(1) + factors.at(2));

           //return 100.- (10 + factors.at(1)) *factors.at(2)* TMath::Abs( TMath::Sin(factors.at(0)) );
       }
};








void exampleGA(){
        std::cout << "\nEXAMPLE" << std::endl;
        // define all the parameters by their minimum and maximum value
        // in this example 3 parameters are defined.
        vector<Interval*> ranges;
        ranges.push_back( new Interval(0,15,30) );
        ranges.push_back( new Interval(0,13) );
        ranges.push_back( new Interval(0,5,3) );

        for( std::vector<Interval*>::iterator it = ranges.begin(); it != ranges.end(); it++ ){
           std::cout << " range: " << (*it)->GetMin() << "   " << (*it)->GetMax() << std::endl;
        }

        IFitterTarget* myFitness = new MyFitness();

        // prepare the genetic algorithm with an initial population size of 20
        // mind: big population sizes will help in searching the domain space of the solution
        // but you have to weight this out to the number of generations
        // the extreme case of 1 generation and populationsize n is equal to
        // a Monte Carlo calculation with n tries

        const TString name( "exampleGA" );
        const TString opts( "PopSize=100:Steps=30" );

        GeneticFitter mg( *myFitness, name, ranges, opts);
       // mg.SetParameters( 4, 30, 200, 10,5, 0.95, 0.001 );

        std::vector<Double_t> result;
        Double_t estimator = mg.Run(result);

         int n = 0;
         for( std::vector<Double_t>::iterator it = result.begin(); it<result.end(); it++ ){
             std::cout << "FACTOR " << n << " : " << (*it) << std::endl;
             n++;
         }

}



} // namespace TMVA

void TMVAGAexample2() {
   cout << "Start Test TMVAGAexample" << endl
        << "========================" << endl
        << endl;

   TMVA::exampleGA();

}


int main( int argc, char** argv )
{
   TMVAGAexample2();
   return 0;
}
