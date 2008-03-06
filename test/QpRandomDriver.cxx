#include <stdlib.h>
#include "Riostream.h"

#include "TQpDataDens.h"
#include "TQpVar.h"
#include "TQpProbDens.h"
#include "TGondzioSolver.h"

Bool_t SolutionMatches(TQpVar *vars,TQpVar *soln,TQpVar *temp,Double_t tol);

int main(int argc,char *argv[])
{
  Int_t n1 = 5, m1 = 2, m2 = 2;
  if (argc >= 4) {
    n1 = atoi(argv[1]);
    m1 = atoi(argv[2]);
    m2 = atoi(argv[3]);
  } else {
    cout << endl 
         << " Usage: QpRandomDriver n my mz " << endl
         << " where n  = # primal variables, " << endl
         << "       my = # equality constraints, " << endl
         << "       mz = # inequality constraints " << endl << endl;
    return 1;
  }

  Int_t nnzQ = (Int_t) .20*(n1*n1);
  Int_t nnzA = (Int_t) .15*(m1*n1);
  Int_t nnzC = (Int_t) .10*(m2*n1);

  if (nnzQ < 3*n1) nnzQ = 3*n1;
  if (nnzA < 3*m1) nnzA = 3*m1;
  if (nnzC < 3*m2) nnzC = 2*m2;

  TQpProbDens *qp = new TQpProbDens(n1,m1,m2);
  TQpDataDens *prob;
  TQpVar *soln;
  qp->MakeRandomData(prob,soln,0,0,0);
//  qp->MakeRandomData(prob,soln,nnzQ,nnzA,nnzC);
  TQpVar      *vars  = qp->MakeVariables(prob);
  TQpResidual *resid = qp->MakeResiduals(prob);

  TGondzioSolver *s = new TGondzioSolver(qp,prob);
  
  const Int_t status = s->Solve(prob,vars,resid);
  delete s;

  if (status == 0) {
    cout.precision(4);
    cout << endl << "Computed solution:\n\n" <<endl <<endl;
    vars->fX.Print();

    TQpVar *temp = qp->MakeVariables(prob);

    cout << endl << "Checking the solution...";
    if (SolutionMatches(vars,soln,temp,1e-4)) {
      cout << "The solution appears to be correct." <<endl;
    } else {
      cout << endl << "The solution may be wrong "
	"(or the generated problem may be ill conditioned.)" <<endl;
    }
    delete temp;
  } else {
    cout << "Could not solve this problem." <<endl;
  }

  delete vars;
  delete soln;
  delete prob;
  delete qp;

  return status;
}

Bool_t SolutionMatches(TQpVar *vars,TQpVar *soln,TQpVar *temp,double tol)
{
  temp = vars;

  // Only x is  significant
  temp->fX -= soln->fX;

  if ((temp->fX).NormInf()/(1+(soln->fX).NormInf()) < tol)
    return kTRUE;
  else
    return kFALSE;
}

