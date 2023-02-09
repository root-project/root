#  \file
#  \ingroup tutorial_fit
#  \notebook -nodraw
#  Example on how to use the new Minimizer class in ROOT
#   Show usage with all the possible minimizers.
#  Minimize the Rosenbrock function (a 2D -function)
#
#  input : minimizer name + algorithm name
#  randomSeed: = <0 : fixed value: 0 random with seed 0; >0 random with given seed
#
#  \macro_code
#
#  \author Lorenzo Moneta

import ROOT
import numpy as np


def RosenBrock(vecx):
    x = vecx[0]
    y = vecx[1]
    return (y - x**2)**2 + (1 - x)**2

# create minimizer giving a name and a name (optionally) for the specific algorithm
#  possible choices are:
#     minimizerName                  algoName
#
#     Minuit                     Migrad, Simplex,Combined,Scan  (default is Migrad)
#     Minuit2                    Migrad, BFGS, Simplex,Combined,Scan  (default is Migrad)
#     GSLMultiMin                ConjugateFR, ConjugatePR, BFGS, BFGS2, SteepestDescent
#     GSLSimAn
#     Genetic


def NumericalMinimization(minimizerName="Minuit2",
                          algoName="",
                          randomSeed=-1):

    minimizer = ROOT.Math.Factory.CreateMinimizer(minimizerName, algoName)
    if (not minimizer):
        raise RuntimeError(
            "Cannot create minimizer \"{}\". Maybe the required library was not built?".format(minimizerName))

    # Set tolerance and other minimizer parameters, one can also use default
    # values

    minimizer.SetMaxFunctionCalls(1000000)  # working for Minuit/Minuit2
    # for GSL minimizers - no effect in Minuit/Minuit2
    minimizer.SetMaxIterations(10000)
    minimizer.SetTolerance(0.001)
    minimizer.SetPrintLevel(1)

    # Create function wrapper for minimizer

    f = ROOT.Math.Functor(RosenBrock, 2)

    # Evaluate function at a point
    x0 = np.array([-1., 2.])
    print("f(-1,1.2) = ", f(x0))

    # Starting point
    variable = [-1., 1.2]
    step = [0.01, 0.01]
    if (randomSeed >= 0):
        r = ROOT.TRandom2(randomSeed)
        variable[0] = r.Uniform(-20, 20)
        variable[1] = r.Uniform(-20, 20)

    minimizer.SetFunction(f)

    # Set the free variables to be minimized !
    minimizer.SetVariable(0, "x", variable[0], step[0])
    minimizer.SetVariable(1, "y", variable[1], step[1])

    # Do the minimization
    ret = minimizer.Minimize()

    xs = minimizer.X()
    print("Minimum: f({} , {}) = {}".format(xs[0],xs[1],minimizer.MinValue()))

    # Real minimum is f(xmin) = 0
    if (ret and minimizer.MinValue() < 1.E-4):
        print("Minimizer {} - {} converged to the right minimum!".format(minimizerName, algoName))
    else:
        print("Minimizer {} - {} failed to converge !!!".format(minimizerName, algoName))
        raise RuntimeError("NumericalMinimization failed to converge!")


if __name__ == "__main__":
    NumericalMinimization()
