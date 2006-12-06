/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// ADD NEW TEST

class complex {
 public:
  double re,im;
  complex(double r=0,double i=0) { re=r; im=i; }
  complex& operator=(complex& a) {
    re=a.re;
    im=a.im;
  }
  double operator[](int i) { return(i?re:im); }
  double operator()(int i) { return(i?re:im); }
  double operator++(void) { return(re++); }
  double operator++(int i) { return(++re); }
  double operator--(void) { return(re--); }
  double operator--(int i) { return(--re); }
};

complex complex::operator+(complex& a,complex& b)
{
  complex c;
  c.re=a.re+b.re;
  c.im=a.im+b.im;
  return(c);
}


int operator<(complex& a,complex& b) { return(a.re<b.re); }
int operator<=(complex& a,complex& b) { return(a.re<=b.re); }
int operator>(complex& a,complex& b) { return(a.re>b.re); }
int operator>=(complex& a,complex& b) { return(a.re>=b.re); }
int operator==(complex& a,complex& b) { return(a.re==b.re); }
int operator!(complex& a) { return(a.re?1:0); }
int operator!=(complex& a,complex& b) { return(a.re!=b.re); }



main()
{
  complex a(1),b(2),c(0);

  printf("%d %d\n",!a,!c);
  printf("a<b=%d a<=b=%d a>b=%d a>=b=%d a==b=%d a!=b=%d !b=%d\n"
	 ,a<b,a<=b,a>b,a>=b,a==b,a!=b,!b);

  if(a<b) printf("a<b o\n");
  else    printf("a<b x\n");
  if(a<=b) printf("a<=b o\n");
  else     printf("a<=b x\n");
  if(a>b) printf("a>b o\n");
  else    printf("a>b x\n");
  if(a>=b) printf("a>=b o\n");
  else     printf("a>=b x\n");
  if(a==b) printf("a==b o\n");
  else     printf("a==b x\n");
  if(a!=b) printf("a!=b o\n");
  else     printf("a!=b x\n");
  if(!b) printf("!b o\n");
  else   printf("!b x\n");
  if(!c) printf("!c o\n");
  else   printf("!c x\n");
}
