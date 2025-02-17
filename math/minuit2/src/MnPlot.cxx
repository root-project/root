// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnPlot.h"

#include <string>
namespace ROOT {

namespace Minuit2 {

void mnplot(double *xpt, double *ypt, char *chpt, int nxypt, int npagwd, int npagln);

void MnPlot::operator()(std::span<const std::pair<double, double>> points) const
{
   // call routine from Fortran minuit (mnplot) to plot the vector of (x,y) points
   std::vector<double> x;
   x.reserve(points.size());
   std::vector<double> y;
   y.reserve(points.size());
   std::string chpt;
   chpt.reserve(points.size() + 1);

   for (auto &ipoint : points) {
      x.push_back(ipoint.first);
      y.push_back(ipoint.second);
      chpt += '*';
   }

   mnplot(x.data(), y.data(), chpt.data(), points.size(), Width(), Length());
}

void MnPlot::operator()(double xmin, double ymin, std::span<const std::pair<double, double>> points) const
{
   // call routine from Fortran minuit (mnplot) to plot the vector of (x,y) points + minimum values
   std::vector<double> x;
   x.reserve(points.size() + 2);
   x.push_back(xmin);
   x.push_back(xmin);
   std::vector<double> y;
   y.reserve(points.size() + 2);
   y.push_back(ymin);
   y.push_back(ymin);
   std::string chpt;
   chpt.reserve(points.size() + 3);
   chpt += ' ';
   chpt += 'X';

   for (auto &ipoint : points) {
      x.push_back(ipoint.first);
      y.push_back(ipoint.second);
      chpt += '*';
   }

   mnplot(x.data(), y.data(), chpt.data(), points.size() + 2, Width(), Length());
}

} // namespace Minuit2

} // namespace ROOT
