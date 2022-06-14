//script to test error handling


ROOT::R::TRInterface& Exception()
{
ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
r.SetVerbose(1);

//passing bad command 1
r<<"%";
//passing bad command 2
r<<".";

//trying to get an object from a bad command 
TMatrixD m=r["%"];

//The next lines are not supported yet
//Requires segfault signal handling
//r["ss"]<<(double (*)(double))sin;
//r<<"ss()"
return r;
}
