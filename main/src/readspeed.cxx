/* Copyright (C) 2020 Enrico Guiraud
   See the LICENSE file in the top directory for more information. */

#include "ReadSpeedCLI.hxx"
#include "ReadSpeed.hxx"

using namespace ReadSpeed;

int main(int argc, char **argv)
{
   auto args = ParseArgs(argc, argv);

   if (!args.fShouldRun)
      return 1; // ParseArgs has printed the --help, has run the --test or has encountered an issue and logged about it

   PrintThroughput(EvalThroughput(args.fData, args.fNThreads));

   return 0;
}
