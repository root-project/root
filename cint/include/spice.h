#include <array.h>
#include <readfile.h>

void ReadSpice(const char* fname,array& x,array& y,int ndata,int offset)
{
  double res, span;
  ReadFile f(fname);
  while(f.read()) {
    if(f.argc>1) {
      if(strcmp(f.argv[1],".tran")==0) {
	res = atof(f.argv[2]);
	span = atof(f.argv[3]);
      }
      if(strcmp(f.argv[1],"time")==0) {
	break;
      }
    }
  }

  int size = (span / res) *1.05;
  if(ndata) size = ndata;
  int i=0;
  printf("Array size = %g/%g = %d\n",span,res,size);
  x.resize(size);
  y.resize(size);
  f.read();
  f.read();

  for(j=0;j<offset;j++) f.read();
  while(f.read() && i<size) {
    x[i] = atof(f.argv[1]);
    y[i] = atof(f.argv[2]);
    ++i;
  }
  if(0==ndata && i!=size) {
    x.resize(i);
    y.resize(i);
  }
  printf("Array size = %g/%g = %d true size %d\n",span,res,size,i);
}



