{
gSystem->Load("libSmatrix");
using namespace ROOT::Math;
SMatrix<double,2,2> m;
//cout << m << endl;

double * d = m.Array();
d[1] = 1;
d[2] = 2;
d[3] = 3;
//m.Print();
m.print(cout);

TFile f("testSmatrix.root","RECREATE");

f.WriteObjectAny(&m,"SMatrix<double,2,2>","m");

f.Close();

cout << "\nReading File..\n\n";

TFile f2("testSmatrix.root");

SMatrix<double,2,2>  * p_m2 =  (SMatrix<double,2,2> *) f2.GetObjectUnchecked("m");
SMatrix<double,2,2> m2 = *p_m2;
m2.print(cout);
cout << endl; 

cout << " Test read matrix = original matrix "; 
if ( m2 == m ) 
  cout << "  OK " << endl;
else 
cout << "  Failed " << endl;

f2.Close();
}
