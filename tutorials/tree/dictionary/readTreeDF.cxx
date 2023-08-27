// Author: Alvaro Tolosa-Delgado CERN 07/2023
// Author: Jorge Agramunt Ros IFIC(Valencia,Spain) 07/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Example of reading a TTree using the RDataFrame interface
 * Documentation of RDataFrame: https://root.cern/doc/master/classROOT_1_1RDataFrame.html
 */

#include <ROOT/RDataFrame.hxx>

#include "data2Tree.hxx"

void readTreeDF()
{

   ROOT::RDataFrame df{"myTree", "testFile.root"};
   df.Display({"branch1.time", "branch1.energy", "branch2.time", "branch2.energy"}, /*nRows*/ 10)->Print();
}
