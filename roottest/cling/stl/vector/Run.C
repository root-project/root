{
gROOT->ProcessLine(".L t01.C+");
vector<float> v1(10);
vector<float> *v2 = 0;
v2 = mask(v1,2.0);
cout << "Return value: " << v2 << endl; 
}
