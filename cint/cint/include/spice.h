/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
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


/***********************************************************************
 * void ReadSpice()
 * 
 * Description:
 *  Read SPICE .print result from text report.
 ***********************************************************************/
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

/***********************************************************************
 * double todouble(const char* expr)
 * 
 * Description:
 *  Convert  f,p,n,u,m,k,meg,g,t to 1e-15,-12,-9,-6,-3,+3,+6,+9,+12
 ***********************************************************************/
double todouble(const char* expr) {
  double result= atof(expr); 
  if(strstr(expr,"a")) result *= 1e-18;
  else if(strstr(expr,"f")) result *= 1e-15;
  else if(strstr(expr,"p")) result *= 1e-12;
  else if(strstr(expr,"n")) result *= 1e-9;
  else if(strstr(expr,"u")) result *= 1e-6;
  else if(strstr(expr,"m")) result *= 1e-3;
  else if(strstr(expr,"k")) result *= 1e3;
  else if(strstr(expr,"meg")) result *= 1e6;
  else if(strstr(expr,"g")) result *= 1e9;
  else if(strstr(expr,"t")) result *= 1e12;
  else if(strstr(expr,"A")) result *= 1e-18;
  else if(strstr(expr,"F")) result *= 1e-15;
  else if(strstr(expr,"P")) result *= 1e-12;
  else if(strstr(expr,"N")) result *= 1e-9;
  else if(strstr(expr,"U")) result *= 1e-6;
  else if(strstr(expr,"M")) result *= 1e-3;
  else if(strstr(expr,"K")) result *= 1e3;
  else if(strstr(expr,"MEG")) result *= 1e6;
  else if(strstr(expr,"G")) result *= 1e9;
  else if(strstr(expr,"T")) result *= 1e12;
  return(result);
}


