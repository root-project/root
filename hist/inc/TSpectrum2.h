// @(#)root/hist:$Name:  $:$Id: TSpectrum.h,v 1.5 2001/06/22 16:10:17 rdm Exp $
// Author: Miroslav Morhac   27/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum2
#define ROOT_TSpectrum2

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


#ifndef ROOT_TH2
#include "TH2.h"
#endif

const int BACK2_ORDER2 =0;
const int BACK2_ORDER4 =1;
const int BACK2_ORDER6 =2;
const int BACK2_ORDER8 =3;
const int BACK2_INCREASING_WINDOW =0;
const int BACK2_DECREASING_WINDOW =1;
const int BACK2_SUCCESSIVE_FILTERING =0;
const int BACK2_ONE_STEP_FILTERING =1;

const int SMOOTH2_3POINTS =3;
const int SMOOTH2_5POINTS =5;
const int SMOOTH2_7POINTS =7;
const int SMOOTH2_9POINTS =9;
const int SMOOTH2_11POINTS =11;
const int SMOOTH2_13POINTS =13;
const int SMOOTH2_15POINTS =15;

const bool SEARCH2_EXCLUDE_MARKOV =false;
const bool SEARCH2_INCLUDE_MARKOV =true;

const int FIT2_OPTIM_CHI_COUNTS =0;
const int FIT2_OPTIM_CHI_FUNC_VALUES =1;
const int FIT2_OPTIM_MAX_LIKELIHOOD =2;
const int FIT2_ALPHA_HALVING =0;
const int FIT2_ALPHA_OPTIMAL =1;
const int FIT2_FIT_POWER2 =2;
const int FIT2_FIT_POWER4 =4;
const int FIT2_FIT_POWER6 =6;
const int FIT2_FIT_POWER8 =8;
const int FIT2_FIT_POWER10 =10;
const int FIT2_FIT_POWER12 =12;
const int FIT2_TAYLOR_ORDER_FIRST =0;
const int FIT2_TAYLOR_ORDER_SECOND =1;
const int FIT2_NUM_OF_REGUL_CYCLES =100;

const int TRANSFORM2_HAAR =0;
const int TRANSFORM2_WALSH =1;
const int TRANSFORM2_COS =2;
const int TRANSFORM2_SIN =3;
const int TRANSFORM2_FOURIER =4;
const int TRANSFORM2_HARTLEY =5;
const int TRANSFORM2_FOURIER_WALSH =6;
const int TRANSFORM2_FOURIER_HAAR =7;
const int TRANSFORM2_WALSH_HAAR =8;
const int TRANSFORM2_COS_WALSH =9;
const int TRANSFORM2_COS_HAAR =10;
const int TRANSFORM2_SIN_WALSH =11;
const int TRANSFORM2_SIN_HAAR =12;
const int TRANSFORM2_FORWARD =0;
const int TRANSFORM2_INVERSE =1;

const int MAX_NUMBER_OF_PEAKS2= 1000;

	class TSpectrumTwoDimFit{
        public:
		int number_of_peaks;//input parameter, should be>0
		int number_of_iterations;//input parameter, should be >0
                int xmin;//first fitted channel in x direction
                int xmax;//last fitted channel in x direction
                int ymin;//first fitted channel in y direction
                int ymax;//last fitted channel in y direction
        	double alpha;//convergence coefficient, input parameter, it should be positive number and <=1
        	double chi; //here the function returns resulting chi square
                int statistic_type; //type of statistics, possible values FIT2_OPTIM_CHI_COUNTS (chi square statistics with counts as weighting coefficients), FIT2_OPTIM_CHI_FUNC_VALUES (chi square statistics with function values as weighting coefficients),FIT2_OPTIM_MAX_LIKELIHOOD
                int alpha_optim;//optimization of convergence coefficients, possible values FIT2_ALPHA_HALVING, FIT2_ALPHA_OPTIMAL, see manual
                int power;//possible values FIT21_FIT_POWER2,4,6,8,10,12, see manual
                int fit_taylor;//order of Taylor expansion, possible values FIT2_TAYLOR_ORDER_FIRST, FIT2_TAYLOR_ORDER_SECOND
		double position_init_x[MAX_NUMBER_OF_PEAKS2];//initial values of x positions of 2D peaks, input parameters
		double position_calc_x[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted x positions of 2D peaks, output parameters
		double position_err_x[MAX_NUMBER_OF_PEAKS2];//x position errors of 2D peaks
                bool fix_position_x[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate x positions of 2D peaks (not fit). However they are present in the estimated functional
		double position_init_y[MAX_NUMBER_OF_PEAKS2];//initial values of y positions of 2D peaks, input parameters
		double position_calc_y[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted y positions of 2D peaks, output parameters
		double position_err_y[MAX_NUMBER_OF_PEAKS2];//y position errors of 2D peaks
                bool fix_position_y[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate y positions of 2D peaks (not fit). However they are present in the estimated functional
		double position_init_x1[MAX_NUMBER_OF_PEAKS2];//initial values of x positions of 1D ridges, input parameters
		double position_calc_x1[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted x positions of 1D ridges, output parameters
		double position_err_x1[MAX_NUMBER_OF_PEAKS2];//x position errors of 1D ridges
                bool fix_position_x1[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate x positions of 1D ridges (not fit). However they are present in the estimated functional
		double position_init_y1[MAX_NUMBER_OF_PEAKS2];//initial values of y positions of 1D ridges, input parameters
		double position_calc_y1[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted y positions of 1D ridges, output parameters
		double position_err_y1[MAX_NUMBER_OF_PEAKS2];//y position errors of 1D ridges
                bool fix_position_y1[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate y positions of 1D ridges (not fit). However they are present in the estimated functional
        	double amp_init[MAX_NUMBER_OF_PEAKS2];//initial values of 2D peaks amplitudes, input parameters
	        double amp_calc[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted amplitudes of 2D peaks, output parameters
        	double amp_err[MAX_NUMBER_OF_PEAKS2];//amplitude errors of 2D peaks
	        bool fix_amp[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate amplitudes of 2D peaks (not fit). However they are present in the estimated functional
        	double amp_init_x1[MAX_NUMBER_OF_PEAKS2];//initial values of 1D ridges amplitudes, input parameters
	        double amp_calc_x1[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted amplitudes of 1D ridges, output parameters
        	double amp_err_x1[MAX_NUMBER_OF_PEAKS2];//amplitude errors of 1D ridges
	        bool fix_amp_x1[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate amplitudes of 1D ridges (not fit). However they are present in the estimated functional
        	double amp_init_y1[MAX_NUMBER_OF_PEAKS2];//initial values of 1D ridges amplitudes, input parameters
	        double amp_calc_y1[MAX_NUMBER_OF_PEAKS2];//calculated values of fitted amplitudes of 1D ridges, output parameters
        	double amp_err_y1[MAX_NUMBER_OF_PEAKS2];//amplitude errors of 1D ridges
	        bool fix_amp_y1[MAX_NUMBER_OF_PEAKS2];//logical vector which allows to fix appropriate amplitudes of 1D ridges (not fit). However they are present in the estimated functional
        	double volume[MAX_NUMBER_OF_PEAKS2];//calculated areas of peaks
	        double volume_err[MAX_NUMBER_OF_PEAKS2];//errors of peak areas
        	double sigma_init_x;//sigma x parameter, meaning analogical to the above given parameters, see manual
	        double sigma_calc_x;
        	double sigma_err_x;
	        bool fix_sigma_x;
        	double sigma_init_y;//sigma y parameter, meaning analogical to the above given parameters, see manual
	        double sigma_calc_y;
        	double sigma_err_y;
	        bool fix_sigma_y;
        	double ro_init;// correlation coefficient - see manual
	        double ro_calc;
        	double ro_err;
	        bool fix_ro;
        	double txy_init;//t parameter for 2D peaks, meaning analogical to the above given parameters, see manual
	        double txy_calc;
        	double txy_err;
	        bool fix_txy;
        	double sxy_init;//s parameter for 2D peaks, meaning analogical to the above given parameters, see manual
	        double sxy_calc;
        	double sxy_err;
	        bool fix_sxy;
        	double tx_init;//t parameter for 1D ridges (x direction), meaning analogical to the above given parameters, see manual
	        double tx_calc;
        	double tx_err;
	        bool fix_tx;
        	double ty_init;//t parameter for 1D ridges (y direction), meaning analogical to the above given parameters, see manual
	        double ty_calc;
        	double ty_err;
	        bool fix_ty;
        	double sx_init;//s parameter for 1D ridges (x direction), meaning analogical to the above given parameters, see manual
	        double sx_calc;
        	double sx_err;
	        bool fix_sx;
        	double sy_init;//s parameter for 1D ridges (y direction), meaning analogical to the above given parameters, see manual
	        double sy_calc;
        	double sy_err;
	        bool fix_sy;
        	double bx_init;//b parameter for 1D ridges (x direction), meaning analogical to the above given parameters, see manual
	        double bx_calc;
        	double bx_err;
	        bool fix_bx;
        	double by_init;//b parameter for 1D ridges (y direction), meaning analogical to the above given parameters, see manual
	        double by_calc;
        	double by_err;
	        bool fix_by;
        	double a0_init;//backgroud is estimated as a0+ax*x+ay*y
	        double a0_calc;
        	double a0_err;
	        bool fix_a0;
        	double ax_init;
	        double ax_calc;
        	double ax_err;
	        bool fix_ax;
        	double ay_init;
	        double ay_calc;
        	double ay_err;
	        bool fix_ay;
	};

class TSpectrum2 : public TNamed {
protected:
   Int_t         fMaxPeaks;       //Maximum number of peaks to be found
   Int_t         fNPeaks;         //number of peaks found
   Float_t      *fPosition;       //!array of current peak positions
   Float_t      *fPositionX;      //!X position of peaks
   Float_t      *fPositionY;      //!Y position of peaks
   Float_t       fResolution;     //resolution of the neighboring peaks
   TH1          *fHistogram;      //resulting histogram

public:
   TSpectrum2();
   TSpectrum2(Int_t maxpositions, Float_t resolution=1);
   virtual ~TSpectrum2();
   virtual const char *Background(TH1 *hist,int niter, Option_t *option="goff");
   const char *Background2(float **spectrum,int sizex,int sizey,int niter);
   const char *Deconvolution2(float **source,const float **resp,int sizex,int sizey,int niter);
   TH1        *GetHistogram() const {return fHistogram;}
   Int_t       GetNPeaks() const {return fNPeaks;}
   Float_t    *GetPositionX() const {return fPositionX;}
   Float_t    *GetPositionY() const {return fPositionY;}
   int         PeakEvaluate(const double *temp,int size,int xmax,double xmin,bool markov);
   virtual Int_t       Search(TH1 *hist, Double_t sigma, Option_t *option="goff");
   Int_t       Search2(float **source,int sizex,int sizey,double sigma);
   void        SetResolution(Float_t resolution=1);

   //new functions April 2003
   const char *Background2RectangularRidges(float **spectrum,int sizex,int sizey,int number_of_iterations_x,int number_of_iterations_y,int direction,int filter_order,int filter_type);
   const char *Background2RectangularRidgesX(float **spectrum,int sizex,int sizey,int number_of_iterations,int direction,int filter_order);
   const char *Background2RectangularRidgesY(float **spectrum,int sizex,int sizey,int number_of_iterations,int direction,int filter_order);
   const char *Background2SkewRidges(float **spectrum,int sizex,int sizey,int number_of_iterations_x,int number_of_iterations_y,int direction,int filter_order);
   const char *Background2NonlinearRidges(float **spectrum,int sizex,int sizey,int number_of_iterations_x,int number_of_iterations_y,int direction,int filter_order);
   const char *Smooth2(float **spectrum,int sizex,int sizey,int pointsx,int pointsy);
   void DecFourier2(double *working_space,int num,int iter,int inv);
   const char *Deconvolution2HighResolution(float** source,const float** resp,int sizex,int sizey,int number_of_iterations,int number_of_repetitions,double boost);
   Int_t Search2General(float **source,int sizex,int sizey,double sigma,int threshold,bool markov,int aver_window);
   double      Lls(double a);

      //auxiliary functions for 2. parameter fit functions
   double Erfc(double x);
   double Derfc(double x);
   double Ourpowl(double a,int pw);
   void   StiefelInversion(double **a,int rozmer);

   double Shape2(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double ro,double a0,double ax,double ay,double txy,double sxy,double tx,double ty,double sx,double sy,double bx,double by);
   double Deramp2(double x,double y,double x0,double y0,double sigmax,double sigmay,double ro,double txy,double sxy,double bx,double by);
   double Derampx(double x,double x0,double sigmax,double tx,double sx,double bx);
   double Deri02(double x,double y,double a,double x0,double y0,double sigmax,double sigmay,double ro,double txy,double sxy,double bx,double by);
   double Derderi02(double x,double y,double a,double x0,double y0,double sigmax,double sigmay,double ro);
   double Derj02(double x,double y,double a,double x0,double y0,double sigmax,double sigmay,double ro,double txy,double sxy,double bx,double by);
   double Derderj02(double x,double y,double a,double x0,double y0,double sigmax,double sigmay,double ro);
   double Deri01(double x,double ax,double x0,double sigmax,double tx,double sx,double bx);
   double Derderi01(double x,double ax,double x0,double sigmax);
   double Dersigmax(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double ro,double txy,double sxy,double tx,double sx,double bx,double by);
   double Derdersigmax(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double ro);
   double Dersigmay(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double ro,double txy,double sxy,double ty,double sy,double bx,double by);
   double Derdersigmay(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double ro);
   double Derro(int num_of_fitted_peaks,double x,double y,const double *parameter,double sx,double sy,double r);
   double Dertxy(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double bx,double by);
   double Dersxy(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay);
   double Dertx(int num_of_fitted_peaks,double x,const double *parameter,double sigmax,double bx);
   double Derty(int num_of_fitted_peaks,double x,const double *parameter,double sigmax,double bx);
   double Dersx(int num_of_fitted_peaks,double x,const double *parameter,double sigmax);
   double Dersy(int num_of_fitted_peaks,double x,const double *parameter,double sigmax);
   double Derbx(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double txy,double tx,double bx,double by);
   double Derby(int num_of_fitted_peaks,double x,double y,const double *parameter,double sigmax,double sigmay,double txy,double ty,double bx,double by);
   double Derpa2(double sx,double sy,double ro);
   double Derpsigmax(double a,double sy,double ro);
   double Derpsigmay(double a,double sx,double ro);
   double Derpro(double a,double sx,double sy,double ro);
   double Volume(double a,double sx,double sy,double ro);

   const char* Fit2Awmi(float **source, TSpectrumTwoDimFit* p,int sizex,int sizey);
   const char* Fit2Stiefel(float **source, TSpectrumTwoDimFit* p,int sizex,int sizey);

//////////AUXILIARY FUNCTIONS FOR 2. DIMENSIONAL TRANSFORM, FILTER AND ENHANCE FUNCTIONS ////////////////////////
   void  Haar(float *working_space,int num,int direction);
   void  Walsh(float *working_space,int num);
   void  BitReverse(float *working_space,int num);
   void  Fourier(float *working_space,int num,int hartley,int direction,int zt_clear);
   void  BitReverseHaar(float *working_space,int shift,int num,int start);
   int   GeneralExe(float *working_space,int zt_clear,int num,int degree,int type);
   int   GeneralInv(float *working_space,int num,int degree,int type);

   void  HaarWalsh2(float **working_matrix,float *working_vector,int numx,int numy,int direction,int type);
   void  FourCos2(float **working_matrix,float *working_vector,int numx,int numy,int direction,int type);
   void  General2(float **working_matrix,float *working_vector,int numx,int numy,int direction,int type,int degree);

   const char *Transform2(const float **source,float **dest,int sizex,int sizey,int type,int direction,int degree);
   const char *Filter2Zonal(const float **source,float **dest,int sizex,int sizey,int type,int degree,int xmin, int xmax,int ymin,int ymax,float filter_coeff);
   const char *Enhance2(const float **source,float **dest,int sizex,int sizey,int type,int degree,int xmin, int xmax,int ymin,int ymax,float enhance_coeff);

   ClassDef(TSpectrum2,1)  //Peak Finder, background estimator, Deconvolution for 2-D histograms
};

#endif

