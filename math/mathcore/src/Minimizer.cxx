/*
 * Copyright (c) 2023, CERN
 */

#include <Math/Minimizer.h>

#include <Math/Error.h>
#include <Math/Util.h>

namespace ROOT {
namespace Math {

/** set initial second derivatives
 */
bool Minimizer::SetCovarianceDiag(std::span<const double> g2, unsigned int n)
{
   MATH_UNUSED(g2);
   MATH_UNUSED(n);
   return false;
}

/** set initial values for covariance/error matrix
    The covariance matrix must be provided in compressed form (row-major ordered upper traingular part)
*/
bool Minimizer::SetCovariance(std::span<const double> cov, unsigned int nrow)
{
   MATH_UNUSED(cov);
   MATH_UNUSED(nrow);
   return false;
}

/// set a new upper/lower limited variable (override if minimizer supports them ) otherwise as default set an unlimited
/// variable
bool Minimizer::SetLimitedVariable(unsigned int ivar, const std::string &name, double val, double step, double lower,
                                   double upper)
{
   MATH_WARN_MSG("Minimizer::SetLimitedVariable", "Setting of limited variable not implemented - set as unlimited");
   MATH_UNUSED(lower);
   MATH_UNUSED(upper);
   return SetVariable(ivar, name, val, step);
}

/// set a new fixed variable (override if minimizer supports them )
bool Minimizer::SetFixedVariable(unsigned int ivar, const std::string &name, double val)
{
   MATH_ERROR_MSG("Minimizer::SetFixedVariable", "Setting of fixed variable not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(name);
   MATH_UNUSED(val);
   return false;
}
/// set the value of an already existing variable
bool Minimizer::SetVariableValue(unsigned int ivar, double value)
{
   MATH_ERROR_MSG("Minimizer::SetVariableValue", "Set of a variable value not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(value);
   return false;
}

/// set the step size of an already existing variable
bool Minimizer::SetVariableStepSize(unsigned int ivar, double value)
{
   MATH_ERROR_MSG("Minimizer::SetVariableStepSize", "Setting an existing variable step size not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(value);
   return false;
}
/// set the lower-limit of an already existing variable
bool Minimizer::SetVariableLowerLimit(unsigned int ivar, double lower)
{
   MATH_ERROR_MSG("Minimizer::SetVariableLowerLimit", "Setting an existing variable limit not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(lower);
   return false;
}
/// set the upper-limit of an already existing variable
bool Minimizer::SetVariableUpperLimit(unsigned int ivar, double upper)
{
   MATH_ERROR_MSG("Minimizer::SetVariableUpperLimit", "Setting an existing variable limit not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(upper);
   return false;
}

/// fix an existing variable
bool Minimizer::FixVariable(unsigned int ivar)
{
   MATH_ERROR_MSG("Minimizer::FixVariable", "Fixing an existing variable not implemented");
   MATH_UNUSED(ivar);
   return false;
}
/// release an existing variable
bool Minimizer::ReleaseVariable(unsigned int ivar)
{
   MATH_ERROR_MSG("Minimizer::ReleaseVariable", "Releasing an existing variable not implemented");
   MATH_UNUSED(ivar);
   return false;
}
/// query if an existing variable is fixed (i.e. considered constant in the minimization)
/// note that by default all variables are not fixed
bool Minimizer::IsFixedVariable(unsigned int ivar) const
{
   MATH_ERROR_MSG("Minimizer::IsFixedVariable", "Querying an existing variable not implemented");
   MATH_UNUSED(ivar);
   return false;
}
/// get variable settings in a variable object (like ROOT::Fit::ParamsSettings)
bool Minimizer::GetVariableSettings(unsigned int ivar, ROOT::Fit::ParameterSettings &pars) const
{
   MATH_ERROR_MSG("Minimizer::GetVariableSettings", "Querying an existing variable not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(pars);
   return false;
}
/** return covariance matrices element for variables ivar,jvar
    if the variable is fixed the return value is zero
    The ordering of the variables is the same as in the parameter and errors vectors
*/
double Minimizer::CovMatrix(unsigned int ivar, unsigned int jvar) const
{
   MATH_UNUSED(ivar);
   MATH_UNUSED(jvar);
   return 0;
}

/**
    Fill the passed array with the  covariance matrix elements
    if the variable is fixed or const the value is zero.
    The array will be filled as cov[i *ndim + j]
    The ordering of the variables is the same as in errors and parameter value.
    This is different from the direct interface of Minuit2 or TMinuit where the
    values were obtained only to variable parameters
*/
bool Minimizer::GetCovMatrix(double *covMat) const
{
   MATH_UNUSED(covMat);
   return false;
}

/**
    Fill the passed array with the Hessian matrix elements
    The Hessian matrix is the matrix of the second derivatives
    and is the inverse of the covariance matrix
    If the variable is fixed or const the values for that variables are zero.
    The array will be filled as h[i *ndim + j]
*/
bool Minimizer::GetHessianMatrix(double *hMat) const
{
   MATH_UNUSED(hMat);
   return false;
}

/**
   return global correlation coefficient for variable i
   This is a number between zero and one which gives
   the correlation between the i-th parameter  and that linear combination of all
   other parameters which is most strongly correlated with i.
   Minimizer must overload method if implemented
 */
double Minimizer::GlobalCC(unsigned int ivar) const
{
   MATH_UNUSED(ivar);
   return -1;
}

/**
   minos error for variable i, return false if Minos failed or not supported
   and the lower and upper errors are returned in errLow and errUp
   An extra flag  specifies if only the lower (option=-1) or the upper (option=+1) error calculation is run
*/
bool Minimizer::GetMinosError(unsigned int ivar, double &errLow, double &errUp, int option)
{
   MATH_ERROR_MSG("Minimizer::GetMinosError", "Minos Error not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(errLow);
   MATH_UNUSED(errUp);
   MATH_UNUSED(option);
   return false;
}

/**
   perform a full calculation of the Hessian matrix for error calculation
 */
bool Minimizer::Hesse()
{
   MATH_ERROR_MSG("Minimizer::Hesse", "Hesse not implemented");
   return false;
}

/**
   scan function minimum for variable i. Variable and function must be set before using Scan
   Return false if an error or if minimizer does not support this functionality
 */
bool Minimizer::Scan(unsigned int ivar, unsigned int &nstep, double *x, double *y, double xmin, double xmax)
{
   MATH_ERROR_MSG("Minimizer::Scan", "Scan not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(nstep);
   MATH_UNUSED(x);
   MATH_UNUSED(y);
   MATH_UNUSED(xmin);
   MATH_UNUSED(xmax);
   return false;
}

/**
   find the contour points (xi, xj) of the function for parameter ivar and jvar around the minimum
   The contour will be find for value of the function = Min + ErrorUp();
 */
bool Minimizer::Contour(unsigned int ivar, unsigned int jvar, unsigned int &npoints, double *xi, double *xj)
{
   MATH_ERROR_MSG("Minimizer::Contour", "Contour not implemented");
   MATH_UNUSED(ivar);
   MATH_UNUSED(jvar);
   MATH_UNUSED(npoints);
   MATH_UNUSED(xi);
   MATH_UNUSED(xj);
   return false;
}

/// get name of variables (override if minimizer support storing of variable names)
/// return an empty string if variable is not found
std::string Minimizer::VariableName(unsigned int ivar) const
{
   MATH_UNUSED(ivar);
   return std::string(); // return empty string
}

/// get index of variable given a variable given a name
/// return -1 if variable is not found
int Minimizer::VariableIndex(const std::string &name) const
{
   MATH_ERROR_MSG("Minimizer::VariableIndex", "Getting variable index from name not implemented");
   MATH_UNUSED(name);
   return -1;
}

} // namespace Math
} // namespace ROOT
