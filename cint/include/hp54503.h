/*****************************************************************
* hp54503.h
*
*****************************************************************/
#ifndef G__HP54503_H
#define G__HP54503_H

#include <hpib.h>
#include <array.h>

class hp54503 {
	int hpibaddr;
	int ch[5];
	array *pa[5];
	int ndig;

      public:
	hp54503(int addr);
	hp54503& operator <<(char *s);
	hp54503& operator >>(char *s);
	hp54503& operator >>(char c);
	hp54503& operator >>(array& a);
	hp54503& operator >>(int chin);
      private:
	void get_preamble(int *ppoints
			,double *pxinc,double *pxorigin,int *pxref
			,double *pyinc,double *pyorigin,int *pyref);
	void init(void);
};


hp54503::hp54503(int addr)
{
 	hpibaddr=addr; 
	ndig=0;
} 

hp54503& hp54503::operator <<(char *s)
{
	if(!G__eid) init();
	hpib_output(G__eid,hpibaddr,s,1);
	return(*this);
}

hp54503& hp54503::operator >>(int chin)
{
	ch[ndig-1] = chin;	
	return(*this);
}

hp54503& hp54503::operator >>(array& a)
{
	pa[ndig++]=&a;
	return(*this);
} 

hp54503& hp54503::operator >>(char *s)
{
	get_data();
	return(*this);
} 

hp54503& hp54503::operator >>(char c)
{
	get_data();
	return(*this);
} 

void hp54503::get_data(void)
{
	int i,n;
	char command[80];
	char headers[50];
	unsigned char data[2000];
	int points;
	int xref,yref;
	double xinc,xorigin,yinc,yorigin;

	if(!G__eid) init();
	hpib_clear(G__eid,hpibaddr);
	hpib_output(G__eid,hpibaddr,"WAVEFORM:FORMAT BYTE",1);
	for(i=1;i<ndig;i++) {
		sprintf(command,"WAVEFORM:SOURCE CHANNEL%d",ch[i]);
		hpib_output(G__eid,hpibaddr,command,1);
		get_preamble(&points
			,&xinc,&xorigin,&xref
			,&yinc,&yorigin,&yref);
		pa[i]->resize(points);
		hpib_output(G__eid,hpibaddr,"WAVEFORM:DATA?",1);
		hpib_enter(G__eid,hpibaddr,headers,11);
		hpib_enter(G__eid,hpibaddr,data,points+1);

		for(n=0;n<points;n++) {
			pa[i]->dat[n]=(data[n]-yref)*yinc+yorigin;
		}
	}
	pa[0]->resize(points);
	for(n=0;n<points;n++) {
		pa[0]->dat[n]=(n-yref)*xinc+xorigin;
	}
}


void hp54503::get_preamble(int *ppoints
			,double *pxinc,double *pxorigin,int *pxref
			,double *pyinc,double *pyorigin,int *pyref)
{
	char preamble[101];
	char *start,*pc;
	hpib_output(G__eid,hpibaddr,"WAVEFORM:PREAMBLE?",1);
	hpib_enter(G__eid,hpibaddr,preamble,100);

	pc=strchr(preamble,',');
	pc=strchr(pc+1,',');
	start = pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*ppoints = atoi(start);

	pc=strchr(pc+1,',');
	start = pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*pxinc = atof(start);

	start=pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*pxorigin = atof(start);

	start=pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*pxref = atoi(start);
	
	start = pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*pyinc = atof(start);

	start=pc+1;
	pc = strchr(start,',');
	*pc='\0';
	*pyorigin = atof(start);

	start=pc+1;
	*pyref = atoi(start);
}

void hp54503::init()
{
	G__eid=hpib_open(G__sc);
	hpib_clear(G__eid,hpibaddr);
}

#endif
