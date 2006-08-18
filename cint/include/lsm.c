/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* lsm.c
*
* Least square method library
*
*   makecint -A -dl lsm.sl -c lsm.c
*
*************************************************************************/

#define G__LSMSL

/*************************************************************************
* Least Square method
*
* Input : 
*    double x[n],y[n]   : arbitrary data points 
*    int n              : number of data point
*    int m              : polinominal order  1 <= m <= 20
*
* Output :
*    double c[m+1]      : y=c[m]*x^m + c[m-1]*x^(m-1) .... c[1]*x + c[0]
*
* Return :
*        0              : success
*        -1             : m>=n || m<1 || m>20
* 
*************************************************************************/

int lstsq(double x[],double y[],int n,int m,double c[])
{
	int mp1,mp2,m2,i,j,k,l;
	double w1,w2,w3,pivot,aik,a[21][22],w[42];

	/* Invalid condition check */
	if(m>=n || m<1 || m>20) return(-1);

	/* Set constants */
	mp1 = m+1;
	mp2 = m+2;
	m2 = 2*m;
	
	for(i=0;i<m2;i++) {
		w1 = 0.0 ; 
		for(j=0;j<n;j++) {
			w2=w3=x[j];
			for(k=0;k<i;k++) 
				w2 *= w3;
			w1 += w2;
		}
		w[i]=w1;
	}

	for(i=0;i<mp1;i++) {
		for(j=0;j<mp1;j++) {
			l=i+j-1;
			a[i][j]=w[l];
		}
	}

	a[0][0]=n;
	w1 = 0.0;
	for(i=0;i<n;i++) 
		w1 += y[i];

	a[0][mp1] = w1;
	for(i=0;i<m;i++) {
		w1 = 0.0;
		for(j=0;j<n;j++) {
			w2=w3=x[j];
			for(k=0;k<i;k++) 
				w2 *= w3;
			w1 += y[j] * w2;
		}
		a[i+1][mp1] = w1;
	}

	for(k=0;k<mp1;k++) {
		pivot = a[k][k];
		for(j=k;j<mp2;j++) 
			a[k][j] /= pivot;
		for(i=0;i<mp1;i++) {
			if(i != k) {
				aik = a[i][k];
				for(j=k;j<mp2;j++)
					a[i][j] -= aik * a[k][j];
			}
		}
	}


	/* Copy result to c[] */
	for(i=0;i<mp1;i++)
		c[i] = a[i][mp1];

	return(0);
}

/*************************************************************************
* eval_lstsq
*
*
*************************************************************************/
int eval_lstsq(double x[],double z[],int n,int m,double c[])
{
	int i,j;
	double xn;
	double result;

	/* calculate c[] */
	/* if(lstsq(x,y,n,m,c) == -1) return(-1); */

	/* calculate z[] */
	for(i=0;i<n;i++) {
		result = 0;
		xn=1;
		for(j=0;j<=m;j++) {
			result += c[j] * xn ;
			xn *= x[i] ;
		}
		z[i] = result;
	}

	return(0);
}
