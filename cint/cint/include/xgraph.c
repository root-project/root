/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
* xgraph.c
*
*  makecint -A -dl xgraph.sl -c xgraph.c
************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G__XGRAPHSL

/************************************************************************
* xgraph library
************************************************************************/

int xgraph_open(char *filename,char *title)
{
	FILE *fp;
	fp=fopen(filename,"w");
	if(title) fprintf(fp,"TitleText: %s\n",title);
	fclose(fp);
	return(0);
}

int xgraph_data(char *filename,double *xdata,double *ydata,int ndata,char *name)
{
	int i;
	FILE *fp;
	fp=fopen(filename,"a");
	fprintf(fp,"\n");
	fprintf(fp,"\"%s\"\n",name);
	for(i=0;i<ndata;i++) {
		fprintf(fp,"%e %e\n",xdata[i],ydata[i]);
	}
	fclose(fp);
	return(0);
}

int xgraph_invoke(char *filename,double xmin,double xmax,double ymin,double ymax,int xlog,int ylog,char *xunit,char *yunit)
{
	char temp1[200],temp2[200],temp3[80];
	char *pc;

	sprintf(temp1,"xgraph ");
	if( 0!=xmin || 0!=xmax ) {
		sprintf(temp2,"%s -lx %g,%g ",temp1,xmin,xmax);
		strcpy(temp1,temp2);
	}
	if( 0!=ymin || 0!=ymax ) {
		sprintf(temp2,"%s -ly %g,%g ",temp1,ymin,ymax);
		strcpy(temp1,temp2);
	}
	if(xlog) {
		sprintf(temp2,"%s -lnx",temp1);
		strcpy(temp1,temp2);
	}
	if(ylog) {
		sprintf(temp2,"%s -lny",temp1);
		strcpy(temp1,temp2);
	}
	if(xunit && xunit[0]!='\0') {
		strcpy(temp3,xunit);
		pc=temp3;
		while((pc=strchr(pc,' '))) *pc='_';
		sprintf(temp2,"%s -x %s",temp1,temp3);
		strcpy(temp1,temp2);
	}
	if(yunit && yunit[0]!='\0') {
		strcpy(temp3,yunit);
		pc=temp3;
#ifdef NEVER
		while(pc=strchr(pc,' ')) *pc='_';
		sprintf(temp2,"%s -y %s",temp1,temp3);
#else
		pc=strchr(pc,' ');
		if(pc) {
			*pc='\0';
			sprintf(temp2,"%s -y %s %s",temp1,temp3,pc+1);
		}
		else {
			sprintf(temp2,"%s -y %s",temp1,temp3);
		}
#endif
		strcpy(temp1,temp2);
	}

	sprintf(temp2,"%s %s &",temp1,filename);
	fprintf(stderr,"%s\n",temp2);
	system(temp2);

	return(0);
}

