#include <array.h>
#include <readfile.h>

/***********************************************************************
 * void ReadSpice()
 * 
 * Description:
 *  Read SPICE .print result from text report.
 ***********************************************************************/
void ReadSpice(const char* fname,array& x,array& y,int ndata=0,int offset=0)
{
  double res, span;
  ReadFile f(fname);
 skip2:
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


  int size = (span / res)+1;
  if(ndata) size = ndata;
  int i=0;
  x.resize(size);
  y.resize(size);
  f.read();
  if(f.argc!=1 || 0!=strcmp(f.argv[1],"x")) goto skip2;
  f.read();

  for(int jx=0;jx<offset;jx++) f.read();
  while(f.read() && i<size && f.argc>1) {
    //char *p=strstr(f.argv[2],"E+00"); if(p) *p=0;
    x[i] = atof(f.argv[1]);
    y[i] = atof(f.argv[2]);
    printf("%s=%g\t\t%s=%g\n",f.argv[1],x[i],f.argv[2],y[i]);
    ++i;
  }
  if(0==ndata && i!=size) {
    x.resize(i);
    y.resize(i);
  }
  printf("Array size = %g/%g = %d true size %d\n",span,res,size,i);
}


void ReadSpice(const char* fname,double x[],double y[]
	       ,int ndata=0,int offset=0)
{
  double res, span;
  ReadFile f(fname);
 skip2:
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


  int size = (span / res)+1;
  if(ndata) size = ndata;
  int i=0;
  f.read();
  if(f.argc!=1 || 0!=strcmp(f.argv[1],"x")) goto skip2;
  f.read();

  for(int jx=0;jx<offset;jx++) f.read();
  while(f.read() && i<size && f.argc>1) {
    //char *p=strstr(f.argv[2],"E+00"); if(p) *p=0;
    x[i] = atof(f.argv[1]);
    y[i] = atof(f.argv[2]);
    //printf("%s=%g\t\t%s=%g\n",f.argv[1],x[i],f.argv[2],y[i]);
    ++i;
  }
  printf("Array size = %g/%g = %d true size %d\n",span,res,size,i);
}




