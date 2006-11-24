/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* 
 ===========================================================================
  -*- mode: c -*-   Last Modified Time-stamp: <01/12/31 18:21:37 hata>

  sslib253.c  Cint Version  MATRIX OK! VECTOR NG! cint -03 で実行のこと  
  畑和彦 <kazuhiko.hata@nifty.ne.jp>
 ===========================================================================
*/
/*
 *----------------------------------------------------------------------------
 * ★ 科学技術計算サブルーチンライブラリー
 *    p.89  2.5.3 実係数連立１次方程式　ガウス・ジョルダン法（ＧＡＵＪＯＲ）
 *				分割、module化 OK 確認済み
 *               option base 0    ポインタによる引き数渡し
 *
 *   機能        係数が同じであるk組の実係数連立１次方程式
 *                       Ａ[m, m]・Ｘ[m,k] = Ｂ[m,k]
 *               の解をgaussjordan法で一度に求める。
 *               掃き出す際、軸(pivot)として列要素の絶対値最大を選択するため
 *               行列の入れ換えを行う
 *    書式  int gaujor(double *a, int l, int m, int n, double eps)
 *
 *   引き数  入出力
 *           a   :a[L][N]なる実配列名で実係数連立１次方程式の係数行列および
 *                定数行列である。演算後は定数行列部、すなわちa[i][j],
 *                i=0,1,2,...,M-1, j=M,M+1,M+2,...,N-1に解が得られる。
 *            入力
 *           l   :メインプログラムで宣言した配列Aの第１添字の値を、整定数か整
 *                　変数名で与える。　L≧M
 *           m,n :配列Aの内、演算対象となる列数を、整定数か整変数名で与える
 *                但し、N=M+K、Kは定数行列の列数である。N>M  80≧M≧2
 *           eps :収束判定値を実定数か実変数名で与える。EPS>0
 *            出力
 *           関数値   :エラーフラグである。
 *                       0   : エラーなし
 *                       1   : 掃き出しの途中、ピボット要素がepsより小さい
 *                      -1   : m<2, m>80, m≧n, eps≦0 のいずれかである
 *   スレーブサブルーチン    : なし
 *----------------------------------------------------------------------------
 */
 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* #define CGI */
#undef  MATRIX         // <-------- ここです。
//#define MATRIX         // <-------- ここです。
#define MAIN

#define NMAX 50
#define L 3
#define M 3
#define N 5
#define EPS 1.0e-16

int gaujor2(double a[NMAX][NMAX], int l, int m, int n, double eps);
int gaujor(double *, int, int, int, double);

#ifndef MATRIX
int gaujor(double *a, int l, int m, int n, double eps)
{
  int i,ii,j,k,lc,lr,iw;
  double wmax,w,pivot,api;
  static int work[80];
  
  if(m<1 || m>79 || m>=n || eps<=0)
    return(-1);
  for(i=0; i<m; i++)
    work[i] = i;
  for(k=0; k<m; k++){
    wmax = 0.0;
    for(ii=k; ii<m; ii++){
      for(i=k; i<m; i++){
	w = fabs(*(a+i*n+i));
	if(w>=wmax){
	  wmax = w;
	  lc = i;
	  lr = ii;
	}
      }
    }
    pivot = *(a+lr*n+lc);
    api = fabs(pivot);
    if(api<=eps)
      return(-1);
    if(lc!=k){
      iw = work[k];
      work[k] = work[lc];
      work[lc] = iw;
      for(i=0; i<m; i++){
	w = *(a+i*n+k);
	*(a+i*n+k) = *(a+i*n+lc);
	*(a+i*n+lc) = w;
      }
    }
    if(lr!=k){
      for(j=k; j<n; j++){
	w = *(a+lr*n+j);
	*(a+lr*n+j) = *(a+k*n+j);
	*(a+k*n+j) = w;
      }
    }
    for(i=k+1;i<n; i++)
      *(a+k*n+i) = *(a+k*n+i) / pivot;
    for(i=0; i<m; i++){
      if(i!=k){
	w = *(a+i*n+k);
	for(j=k+1; j<n; j++)
	  *(a+i*n+j) = *(a+i*n+j) - w * *(a+k*n+j);
      }
    }
  }
  for(i=0; i<m; i++){
    do{
      k = work[i];
      if(k==i) break;
      iw = work[k];
      work[k] = work[i];
      work[i] = iw;
      for(j=m; j<n; j++){
	w = *(a+k*n+j);
	*(a+k*n+j) = *(a+i*n+j);
	*(a+i*n+j) = w;
      }
    } while(1);
  }
  return(0);
}
#endif

#ifdef MATRIX
int gaujor2(double a[NMAX][NMAX], int l, int m, int n, double eps)
{
  int i,ii,j,k,lc,lr,iw;
  double wmax,w,pivot,api;
  static int work[80];
  
  if(m<1 || m>79 || m>=n || eps<=0)
    return(-1);
  for(i=0; i<m; i++)
    work[i] = i;
  for(k=0; k<m; k++){
    wmax = 0.0;
    for(ii=k; ii<m; ii++){
      for(i=k; i<m; i++){
	w = fabs(a[i][i]);
	if(w>=wmax){
	  wmax = w;
	  lc = i;
	  lr = ii;
	}
      }
    }
    pivot = a[lr][lc];
    api = fabs(pivot);
    if(api<=eps)
      return(-1);
    if(lc!=k){
      iw = work[k];
      work[k] = work[lc];
      work[lc] = iw;
      for(i=0; i<m; i++){
	w = a[i][k];
	a[i][k] = a[i][lc];
	a[i][lc] = w;
      }
    }
    if(lr!=k){
      for(j=k; j<n; j++){
	w = a[lr][j];
	a[lr][j] = a[k][j];
	a[k][j] = w;
      }
    }
    for(i=k+1;i<n; i++)
      a[k][i] = a[k][i] / pivot;
    for(i=0; i<m; i++){
      if(i!=k){
	w = a[i][k];
	for(j=k+1; j<n; j++)
	  a[i][j] = a[i][j] - w * a[k][j];
      }
    }
  }
  for(i=0; i<m; i++){
    do{
      k = work[i];
      if(k==i) break;
      iw = work[k];
      work[k] = work[i];
      work[i] = iw;
      for(j=m; j<n; j++){
	w = a[k][j];
	a[k][j] = a[i][j];
	a[i][j] = w;
      }
    } while(1);
  }
  return(0);
}
#endif

#ifdef MAIN
int main(void)
{
  int i,j,g /* ,l,m,n */ ;
  
#ifdef MATRIX
  static double a[NMAX][NMAX]={{5.0, 1.0, 1.0, 10.0, 18.0},
				 {1.0,-7.0, 2.0, -7.0, -9.0},
				 {1.0, 1.0, 6.0, 21.0, 11.0}};
#endif
  
#ifndef MATRIX
  static double a[L][N]={{5.0, 1.0, 1.0, 10.0, 18.0},
			   {1.0,-7.0, 2.0, -7.0, -9.0},
			   {1.0, 1.0, 6.0, 21.0, 11.0}};
#endif

#if !defined(G__MSC_VER) && !defined(_MSC_VER)
  system("clear");
#endif

  /* printf("\x01b[2J"); */                /* clear screan */
  printf("\n");
#if 0
  printf("              ★ 科学技術計算サブルーチンライブラリー（Ｃ）\n");
  printf("      2.5.3 実係数連立１次方程式　ガウス・ジョルダン法（ＧＡＵＪＯＲ）\n\n");
#endif
  printf("    given matrix:\n");
  
  for(i=0; i<L; i++){
    printf("    ");
    for(j=0; j<N; j++)
      printf("% 6.2f", a[i][j]);
    printf("\n");
  }
  
/******************************************************************************
  printf("              given equation:\n\n");
  printf("          [no. 1]                      [no. 2]\n");
  printf("               5x1 + x2 + x3 = 10           5x1 + x2 + x3 = 18\n");
  printf("                x1 -7x2 +2x3 = -7            x1 -7x2 +2x3 = -9\n");
  printf("                x1 + x2 +6x3 = 21            x1 + x2 +6x3 = 11\n\n");
  ******************************************************************************/
  
#ifdef MATRIX
  g = gaujor2(a, L, M, N, EPS);
#endif
  
#ifndef MATRIX
  //g = gaujor(*a, L, M, N, EPS);
  g = gaujor(a[0], L, M, N, EPS);
#endif
  
  printf("\n    solution by gaujor:\n");
  
  for(i=1; i<=N-M;i++)
    printf("          [no. %d]     ",i);
  printf("\n");

  for(i=0; i<L; i++){
    for(j=M; j<N; j++)
      printf("    x%d =% 10.6E",i+1,a[i][j]);
    printf("\n");
  }
  return 0;
}
#endif


