{
gROOT->ProcessLine(".L MyOpClass.C+");
MyOpClass obj;
obj.value()["33"];
const char *val = obj.value()["33"];
fprintf(stderr,"val==%s\n",val);
return strcmp(val,"33")!=0;
}
