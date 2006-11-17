#include "variables.C"
#include "correlations.C"
#include "efficiencies.C"
#include "mvas.C"
#include "mutransform.C"

void plotall( TString fin = "TMVA.root" )
{
  cout << "=== execute: variables()" << endl;
  variables( fin );

  cout << "=== execute: correlations()" << endl;
  correlations( fin );

  cout << "=== execute: mvas()" << endl;
  mvas( fin );

  cout << "=== execute: efficiencies()" << endl;
  efficiencies( fin );

  cout << "=== execute: ztransform()" << endl;
  mutransform( fin );
}
