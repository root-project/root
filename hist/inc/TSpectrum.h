// @(#)root/hist:$Name:  $:$Id: TSpectrum.h,v 1.7 2003/07/10 09:55:44 brun Exp $
// Author: Miroslav Morhac   27/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum
#define ROOT_TSpectrum

/////////////////////////////////////////////////////////////////////////////
//	THIS FILE CONTAINS HEADERS FOR ADVANCED				   //
//	SPECTRA PROCESSING FUNCTIONS.	   				   //
//									   //
//	ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	ONE-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	TWO-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	ONE-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//	TWO-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//									   //
//	Miroslav Morhac							   //
//	Institute of Physics						   //
//	Slovak Academy of Sciences					   //
//	Dubravska cesta 9, 842 28 BRATISLAVA				   //
//	SLOVAKIA							   //
//									   //
//	email:fyzimiro@savba.sk,	 fax:+421 7 54772479		   //
//									   //
/////////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TH1
#include "TH1.h"
#endif

const int BACK1_ORDER2 =0;
const int BACK1_ORDER4 =1;
const int BACK1_ORDER6 =2;
const int BACK1_ORDER8 =3;
const int BACK1_INCREASING_WINDOW =0;
const int BACK1_DECREASING_WINDOW =1;
const bool BACK1_EXCLUDE_COMPTON =false;
const bool BACK1_INCLUDE_COMPTON =true;

const int SMOOTH1_3POINTS =3;
const int SMOOTH1_5POINTS =5;
const int SMOOTH1_7POINTS =7;
const int SMOOTH1_9POINTS =9;
const int SMOOTH1_11POINTS =11;
const int SMOOTH1_13POINTS =13;
const int SMOOTH1_15POINTS =15;

const bool SEARCH1_EXCLUDE_MARKOV =false;
const bool SEARCH1_INCLUDE_MARKOV =true;

const int FIT1_OPTIM_CHI_COUNTS =0;
const int FIT1_OPTIM_CHI_FUNC_VALUES =1;
const int FIT1_OPTIM_MAX_LIKELIHOOD =2;
const int FIT1_ALPHA_HALVING =0;
const int FIT1_ALPHA_OPTIMAL =1;
const int FIT1_FIT_POWER2 =2;
const int FIT1_FIT_POWER4 =4;
const int FIT1_FIT_POWER6 =6;
const int FIT1_FIT_POWER8 =8;
const int FIT1_FIT_POWER10 =10;
const int FIT1_FIT_POWER12 =12;
const int FIT1_TAYLOR_ORDER_FIRST =0;
const int FIT1_TAYLOR_ORDER_SECOND =1;
const int FIT1_NUM_OF_REGUL_CYCLES =100;

const int TRANSFORM1_HAAR =0;
const int TRANSFORM1_WALSH =1;
const int TRANSFORM1_COS =2;
const int TRANSFORM1_SIN =3;
const int TRANSFORM1_FOURIER =4;
const int TRANSFORM1_HARTLEY =5;
const int TRANSFORM1_FOURIER_WALSH =6;
const int TRANSFORM1_FOURIER_HAAR =7;
const int TRANSFORM1_WALSH_HAAR =8;
const int TRANSFORM1_COS_WALSH =9;
const int TRANSFORM1_COS_HAAR =10;
const int TRANSFORM1_SIN_WALSH =11;
const int TRANSFORM1_SIN_HAAR =12;
const int TRANSFORM1_FORWARD =0;
const int TRANSFORM1_INVERSE =1;

const int MAX_NUMBER_OF_PEAKS1= 1000;

	class TSpectrumOneDimFit{
        public:
		int number_of_peaks;//input parameter, should be>0
		int number_of_iterations;//input parameter, should be >0
                int xmin;//first fitted channel
                int xmax;//last fitted channel
        	double alpha;//convergence coefficient, input parameter, it should be positive number and <=1
        	double chi; //here the function returns resulting chi square
                int statistic_type; //type of statistics, possible values FIT1_OPTIM_CHI_COUNTS (chi square statistics with counts as weighting coefficients), FIT1_OPTIM_CHI_FUNC_VALUES (chi square statistics with function values as weighting coefficients),FIT1_OPTIM_MAX_LIKELIHOOD
                int alpha_optim;//optimization of convergence coefficients, possible values FIT1_ALPHA_HALVING, FIT1_ALPHA_OPTIMAL, see manual
                int power;//possible values FIT1_FIT_POWER2,4,6,8,10,12, see manual
                int fit_taylor;//order of Taylor expansion, possible values FIT1_TAYLOR_ORDER_FIRST, FIT1_TAYLOR_ORDER_SECOND
		double position_init[MAX_NUMBER_OF_PEAKS1];//initial values of peaks positions, input parameters
		double position_calc[MAX_NUMBER_OF_PEAKS1];//calculated values of fitted positions, output parameters
		double position_err[MAX_NUMBER_OF_PEAKS1];//position errors
                bool fix_position[MAX_NUMBER_OF_PEAKS1];//logical vector which allows to fix appropriate positions (not fit). However they are present in the estimated functional
        	double amp_init[MAX_NUMBER_OF_PEAKS1];//initial values of peaks amplitudes, input parameters
	        double amp_calc[MAX_NUMBER_OF_PEAKS1];//calculated values of fitted amplitudes, output parameters
        	double amp_err[MAX_NUMBER_OF_PEAKS1];//amplitude errors
	        bool fix_amp[MAX_NUMBER_OF_PEAKS1];//logical vector which allows to fix appropriate amplitudes (not fit). However they are present in the estimated functional
        	double area[MAX_NUMBER_OF_PEAKS1];//calculated areas of peaks
	        double area_err[MAX_NUMBER_OF_PEAKS1];//errors of peak areas
        	double sigma_init;//sigma parameter, meaning analogical to the above given parameters, see manual
	        double sigma_calc;
        	double sigma_err;
	        bool fix_sigma;
        	double t_init;//t parameter, meaning analogical to the above given parameters, see manual
	        double t_calc;
        	double t_err;
	        bool fix_t;
        	double b_init;//b parameter, meaning analogical to the above given parameters, see manual
	        double b_calc;
        	double b_err;
	        bool fix_b;
        	double s_init;//s parameter, meaning analogical to the above given parameters, see manual
	        double s_calc;
        	double s_err;
	        bool fix_s;
        	double a0_init;//backgroud is estimated as a0+a1*x+a2*x*x
	        double a0_calc;
        	double a0_err;
	        bool fix_a0;
        	double a1_init;
	        double a1_calc;
        	double a1_err;
	        bool fix_a1;
        	double a2_init;
	        double a2_calc;
        	double a2_err;
	        bool fix_a2;
	};

class TSpectrum : public TNamed {
protected:
   Int_t         fMaxPeaks;       //Maximum number of peaks to be found
   Int_t         fNPeaks;         //number of peaks found
   Float_t      *fPosition;       //!array of current peak positions
   Float_t      *fPositionX;      //!X position of peaks
   Float_t      *fPositionY;      //!Y position of peaks
   Float_t       fResolution;     //resolution of the neighboring peaks
   TH1          *fHistogram;      //resulting histogram

public:
   TSpectrum();
   TSpectrum(Int_t maxpositions, Float_t resolution=1);
   virtual ~TSpectrum();
   virtual const char *Background(TH1 *hist,int niter, Option_t *option="goff");
   const char *Background1(float *spectrum,int size,int niter);
   const char *Deconvolution1(float *source,const float *resp,int size,int niter);
   TH1        *GetHistogram() const {return fHistogram;}
   Int_t       GetNPeaks() const {return fNPeaks;}
   Float_t    *GetPositionX() const {return fPositionX;}
   Float_t    *GetPositionY() const {return fPositionY;}
   virtual Int_t  Search(TH1 *hist, Double_t sigma, Option_t *option="goff", Double_t threshold=0.05);
   void        SetResolution(Float_t resolution=1);

   //new functions April 2003
   const char *Background1General(float *spectrum,int size,int number_of_iterations,int direction,int filter_order,bool compton);
   const char *Smooth1(float *spectrum,int size,int points);
   const char *Deconvolution1HighResolution(float *source,const float *resp,int size,int number_of_iterations,int number_of_repetitions,double boost);
   const char *Deconvolution1Unfolding(float *source,const float **resp,int sizex,int sizey,int number_of_iterations);
   Int_t Search1HighRes(float *source,float *dest, int size, float sigma, double threshold,
                        bool background_remove,int decon_iterations,bool markov, int aver_window);
   double      Lls(double a);

      //auxiliary functions for 1. parameter fit functions
   double Erfc(double x);
   double Derfc(double x);
   double Deramp(double i,double i0,double sigma,double t,double s,double b);
   double Deri0(double i,double amp,double i0,double sigma,double t,double s,double b);
   double Derderi0(double i,double amp,double i0,double sigma);
   double Dersigma(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double t,double s,double b);
   double Derdersigma(int num_of_fitted_peaks,double i,const double* parameter,double sigma);
   double Dert(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double b);
   double Ders(int num_of_fitted_peaks,double i,const double* parameter,double sigma);
   double Derb(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double t,double b);
   double Dera1(double i);
   double Dera2(double i);
   double Shape(int num_of_fitted_peaks,double i,const double *parameter,double sigma,double t,double s,double b,double a0,double a1,double a2);
   double Area(double a,double sigma,double t,double b);
   double Derpa(double sigma,double t,double b);
   double Derpsigma(double a,double t,double b);
   double Derpt(double a,double sigma,double b);
   double Derpb(double a,double sigma,double t,double b);
   double Ourpowl(double a,int pw);
   void   StiefelInversion(double **a,int rozmer);

   const char* Fit1Awmi(float *source, TSpectrumOneDimFit* p,int size);
   const char* Fit1Stiefel(float *source, TSpectrumOneDimFit* p,int size);

//////////AUXILIARY FUNCTIONS FOR 1. DIMENSIONAL TRANSFORM, FILTER AND ENHANCE FUNCTIONS ////////////////////////

   void  Haar(float *working_space,int num,int direction);
   void  Walsh(float *working_space,int num);
   void  BitReverse(float *working_space,int num);
   void  Fourier(float *working_space,int num,int hartley,int direction,int zt_clear);
   void  BitReverseHaar(float *working_space,int shift,int num,int start);
   int   GeneralExe(float *working_space,int zt_clear,int num,int degree,int type);
   int   GeneralInv(float *working_space,int num,int degree,int type);

   const char *Transform1(const float *source,float *dest,int size,int type,int direction,int degree);
   const char *Filter1Zonal(const float *source,float *dest,int size,int type,int degree,int xmin, int xmax,float filter_coeff);
   const char *Enhance1(const float *source,float *dest,int size,int type,int degree,int xmin, int xmax,float enhance_coeff);

   ClassDef(TSpectrum,2)  //Peak Finder, background estimator, Deconvolution
};

#endif

