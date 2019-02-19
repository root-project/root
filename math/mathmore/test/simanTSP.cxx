// example of simulated annealing
// traveling salesman problem
// example from GSL siman_tsp, see also
// http://www.gnu.org/software/gsl/manual/html_node/Traveling-Salesman-Problem.html
//
//
// minimize total distance when visiting a set of cities
//
// you can run in ROOT by doing
//
//.x simanTSP.cxx+
// running FulLSearch() you can check that the result is correct
//
#include <cmath>
#include <vector>
#include <algorithm>

#include "Math/GSLSimAnnealing.h"
#include "Math/GSLRndmEngines.h"
#include "Math/SMatrix.h"
#include "Math/Math.h"
#include "TH1.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TApplication.h"

bool showGraphics = false;

using namespace ROOT::Math;

/* in this table, latitude and longitude are obtained from the US
   Census Bureau, at http://www.census.gov/cgi-bin/gazetteer */

struct s_tsp_city {
  const char * name;
  double lat, longitude;        /* coordinates */
};
typedef struct s_tsp_city Stsp_city;

Stsp_city cities[] = {{"Santa Fe",    35.68,   105.95},
                      {"Phoenix",     33.54,   112.07},
                      {"Albuquerque", 35.12,   106.62},
                      {"Clovis",      34.41,   103.20},
                      {"Durango",     37.29,   107.87},
                      {"Dallas",      32.79,    96.77},
                      {"Tesuque",     35.77,   105.92},
                      {"Grants",      35.15,   107.84},
                      {"Los Alamos",  35.89,   106.28},
                      {"Las Cruces",  32.34,   106.76},
                      {"Cortez",      37.35,   108.58},
                      {"Gallup",      35.52,   108.74}};

#define N_CITIES (sizeof(cities)/sizeof(Stsp_city))

double distance_matrix[N_CITIES][N_CITIES];

/* distance between two cities */
double city_distance(Stsp_city c1, Stsp_city c2)
{
   const double earth_radius = 6375.000; /* 6000KM approximately */
   /* sin and std::cos of lat and long; must convert to radians */
   double sla1 = std::sin(c1.lat*M_PI/180), cla1 = std::cos(c1.lat*M_PI/180),
      slo1 = std::sin(c1.longitude*M_PI/180), clo1 = std::cos(c1.longitude*M_PI/180);
   double sla2 = std::sin(c2.lat*M_PI/180), cla2 = std::cos(c2.lat*M_PI/180),
      slo2 = std::sin(c2.longitude*M_PI/180), clo2 = std::cos(c2.longitude*M_PI/180);

   double x1 = cla1*clo1;
   double x2 = cla2*clo2;

   double y1 = cla1*slo1;
   double y2 = cla2*slo2;

   double z1 = sla1;
   double z2 = sla2;

   double dot_product = x1*x2 + y1*y2 + z1*z2;

   double angle = std::acos(dot_product);

   /* distance is the angle (in radians) times the earth radius */
   return angle*earth_radius;
}


void print_distance_matrix()
{
  unsigned int i, j;

  for (i = 0; i < N_CITIES; ++i) {
    printf("# ");
    for (j = 0; j < N_CITIES; ++j) {
      printf("%15.8f   ", distance_matrix[i][j]);
    }
    printf("\n");
  }
}

// re-implement GSLSimAnFunc for re-defining the metric and function



class MySimAnFunc : public GSLSimAnFunc {

public:

   MySimAnFunc( std::vector<double> & allDist)
   {
      calculate_distance_matrix();
      // initial route is just the sequantial order
      for (unsigned int i = 0; i < N_CITIES; ++i)
         fRoute[i] = i;

      // keep track of all the found distances
      fDist = &allDist;
   }


   virtual ~MySimAnFunc() {}

   unsigned int Route(unsigned int i) const { return fRoute[i]; }

   const unsigned int * Route()  const { return fRoute; }
   unsigned int * Route()   { return fRoute; }

   virtual MySimAnFunc * Clone() const { return new MySimAnFunc(*this); }

   std::vector<double> & AllDist() { return *fDist; }

   virtual double Energy() const {
      // calculate the energy


      double enrg = 0;
      for (unsigned int i = 0; i < N_CITIES; ++i) {
         /* use the distance_matrix to optimize this calculation; it had
            better be allocated!! */
         enrg += fDistanceMatrix( fRoute[i] ,  fRoute[ (i + 1) % N_CITIES ] );
      }

      //std::cout << "energy is " << enrg << std::endl;
      return enrg;
   }

   virtual double Distance(const GSLSimAnFunc & f) const {
      const MySimAnFunc * f2 = dynamic_cast<const MySimAnFunc *> (&f);
      assert (f2 != 0);
      double d = 0;
      // use change in permutations
      for (unsigned int i = 0; i < N_CITIES; ++i) {
         d += (( fRoute[i]  == f2->Route(i) ) ? 0 : 1 );
      }
      return d;
   }
   virtual void Step(const GSLRandomEngine & r, double ) {
      // swap to city in the matrix
      int x1, x2, dummy;

      /* pick the two cities to swap in the matrix; we leave the first
         city fixed */
      x1 = r.RndmInt(N_CITIES);
      do {
         x2 = r.RndmInt(N_CITIES);
      } while (x2 == x1);

      // swap x1 and x2
      dummy = fRoute[x1];
      fRoute[x1] = fRoute[x2];
      fRoute[x2] = dummy;

      //std::cout << "make step -swap  " << x1 << "  " << x2  << std::endl;

   }

   virtual void Print() {
      printf("  [");
      for (unsigned i = 0; i < N_CITIES; ++i) {
         printf(" %d ", fRoute[i]);
      }
      printf("]  ");
      fDist->push_back(Energy()); // store all found distances
   }

   // fast copy (need to keep base class type for using virtuality
   virtual MySimAnFunc & FastCopy(const GSLSimAnFunc & f) {
      const MySimAnFunc * rhs = dynamic_cast<const MySimAnFunc *>(&f);
      assert (rhs != 0);
      std::copy(rhs->fRoute, rhs->fRoute + N_CITIES, fRoute);
      return *this;
   }

   double Distance(int i, int j) const { return fDistanceMatrix(i,j); }

   void PrintRoute();  // used for debugging at the end

   void SetRoute(unsigned int * r) { std::copy(r,r+N_CITIES,fRoute); } // set a new route (used by exh. search)

private:

   void calculate_distance_matrix();

   // variable of the system - order how cities are visited
   unsigned int fRoute[N_CITIES];
   typedef SMatrix<double,N_CITIES,N_CITIES, MatRepSym<double,N_CITIES> > Matrix;
   Matrix fDistanceMatrix;

   std::vector<double> * fDist; // pointer to all distance vector

};


// calculate distance between the cities
void MySimAnFunc::calculate_distance_matrix()
{
  unsigned int i, j;
  double dist;

  for (i = 0; i < N_CITIES; ++i) {
    for (j = 0; j <= i; ++j) {
      if (i == j) {
        dist = 0;
      } else {
        dist = city_distance(cities[i], cities[j]);
      }
      fDistanceMatrix(i,j) = dist;
    }
  }
}

void MySimAnFunc::PrintRoute() {
   // print the route and distance
   double dtot = 0;
   for (unsigned int i = 0; i < N_CITIES; ++i) {
      std::cout << std::setw(20) << cities[Route(i)].name << " \t " << Route(i);
      int j = Route(i);
      int k = Route( (i+ 1) % N_CITIES );
      dtot += Distance(j,k);
      std::cout << "\tdistance [" << j <<  " ,  " << k  << " ]\t= " << Distance(j,k) << "\tTotal Distance\t =  "  << dtot << std::endl;
   }
   std::cout << "Total Route energy is " << dtot << std::endl;
}


// minimize using simulated annealing
void simanTSP(bool debug = true) {

   std::vector<double>  allDist;
   allDist.reserve(5000); // keep track of all distance for histogramming later on

   // create class
   MySimAnFunc f(allDist);

   GSLSimAnnealing siman;

   GSLSimAnParams & p = siman.Params();
   p.n_tries = 200;
   p.iters_fixed_T = 10;
   p.step_size = 1; // not used
   p.k = 1;
   p.t_initial = 5000;
   p.mu_t = 1.01;
   p.t_min = 0.5;

   // set the parameters

   // solve
   siman.Solve(f, debug);

   unsigned int niter = allDist.size();
   std::cout << "minimum found is for distance " << f.Energy()  << std::endl;
   if (debug) std::cout << "number of iterations is " << niter << std::endl;

   std::cout << "Best Route is \n";
   f.PrintRoute();

   // plot configurations
   double x0[N_CITIES+1];
   double y0[N_CITIES+1];
   double xmin[N_CITIES+1];
   double ymin[N_CITIES+1];


   // plot histograms with distances
   TH1 * h1 = new  TH1D("h1","Distance",niter+1,0,niter+1);
   for (unsigned int i = 1; i <=niter; ++i) {
      h1->SetBinContent(i,allDist[i]);
   }

   for (unsigned int i = 0; i < N_CITIES; ++i) {
      // invert long to have west on left side
      x0[i] = -cities[i].longitude;
      y0[i] = cities[i].lat;
      xmin[i] = -cities[f.Route(i)].longitude;
      ymin[i] = cities[f.Route(i)].lat;
   }
   // to close the curve
   x0[N_CITIES] = x0[0];
   y0[N_CITIES] = y0[0];
   xmin[N_CITIES] = xmin[0];
   ymin[N_CITIES] = ymin[0];

   if ( showGraphics )
   {

      TGraph *g0 = new TGraph(N_CITIES+1,x0,y0);
      TGraph *gmin = new TGraph(N_CITIES+1,xmin,ymin);

      TCanvas * c1 = new TCanvas("c","TSP",10,10,1000,800);
      c1->Divide(2,2);
      c1->cd(1);
      g0->Draw("alp");
      g0->SetMarkerStyle(20);
      c1->cd(2);
      gmin->SetMarkerStyle(20);
      gmin->Draw("alp");
      c1->cd(3);
      h1->Draw();
   }

}

unsigned int r1[N_CITIES];
unsigned int r2[N_CITIES];
unsigned int r3[N_CITIES];
unsigned int nfiter = 0;
double best_E, second_E, third_E;

// use STL algorithms for permutations
// algorithm does also the inverse
void do_all_perms(MySimAnFunc & f, int offset )
{
   unsigned int * r = f.Route();

   // do all permutation of N_CITIES -1 (keep first one at same place)
   while (std::next_permutation(r+offset,r+N_CITIES) ) {

      double E = f.Energy();
      nfiter++;
      /* now save the best 3 energies and routes */
      if (E < best_E) {
         third_E = second_E;
         std::copy(r2,r2+N_CITIES,r3);
         second_E = best_E;
         std::copy(r1,r1+N_CITIES,r2);
         best_E = E;
         std::copy(r,r+N_CITIES,r1);
      } else if (E < second_E) {
         third_E = second_E;
         std::copy(r2,r2+N_CITIES,r3);
         second_E = E;
         std::copy(r,r+N_CITIES,r2);
      } else if (E < third_E) {
         third_E = E;
         std::copy(r,r+N_CITIES,r3);
      }
   }
}

#ifdef O
/* James Theiler's recursive algorithm for generating all routes */
// version used in GSL
void do_all_perms(MySimAnFunc & f, int n) {

   if (n == (N_CITIES-1)) {
      /* do it! calculate the energy/cost for that route */
      const unsigned int * r = f.Route();
      /* now save the best 3 energies and routes */
      if (E < best_E) {
         third_E = second_E;
         std::copy(r2,r2+N_CITIES,r3);
         second_E = best_E;
         std::copy(r1,r1+N_CITIES,r2);
         best_E = E;
         std::copy(r,r+N_CITIES,r1);
      } else if (E < second_E) {
         third_E = second_E;
         std::copy(r2,r2+N_CITIES,r3);
         second_E = E;
         std::copy(r,r+N_CITIES,r2);
      } else if (E < third_E) {
         third_E = E;
         std::copy(r,r+N_CITIES,r3);
      }
  } else {
     unsigned int newr[N_CITIES];
     unsigned int j;
     int swap_tmp;
     const unsigned int * r = f.Route();
     std::copy(r,r+N_CITIES,newr);
     for (j = n; j < N_CITIES; ++j) {
        // swap j and n
        swap_tmp = newr[j];
        newr[j] = newr[n];
        newr[n] = swap_tmp;
        f.SetRoute(newr);
        do_all_perms(f, n+1);
     }
  }
}
#endif


// search by brute force the best route
// the full permutations will be Factorial (N_CITIES-1)
// which is approx 4 E+7 for 12 cities

void  FullSearch()
{
   // use still MySimAnFunc for initial routes and distance
   std::vector<double>  dummy;

   MySimAnFunc f(dummy);

   // intitial config

   const unsigned int * r = f.Route();
   std::copy(r,r+N_CITIES,r1);
   std::copy(r,r+N_CITIES,r2);
   std::copy(r,r+N_CITIES,r3);
   best_E = f.Energy();
   second_E = 1.E100;
   third_E = 1.E100;


   do_all_perms(f, 1 );

   std::cout << "number of calls " << nfiter << std::endl;


   printf("\n# exhaustive best route: ");
   std::cout << best_E << std::endl;
   f.SetRoute(r1);   f.PrintRoute();


   printf("\n# exhaustive second_best route: ");
   std::cout << second_E << std::endl;
   f.SetRoute(r2);   f.PrintRoute();

   printf("\n# exhaustive third_best route: ");
   std::cout << third_E << std::endl;
   f.SetRoute(r3);   f.PrintRoute();

}




#ifndef __CINT__
int main(int argc, char **argv)
{
   using std::cout;
   using std::endl;
   using std::cerr;

   bool verbose = false;

  // Parse command line arguments
  for (Int_t i=1 ;  i<argc ; i++) {
     std::string arg = argv[i] ;
     if (arg == "-g") {
      showGraphics = true;
     }
     if (arg == "-v") {
      showGraphics = true;
      verbose = true;
     }
     if (arg == "-h") {
        cerr << "Usage: " << argv[0] << " [-g] [-v]\n";
        cerr << "  where:\n";
        cerr << "     -g : graphics mode\n";
        cerr << "     -v : verbose  mode";
        cerr << endl;
        return -1;
     }
   }

   if ( showGraphics )
   {
      TApplication* theApp = 0;
      theApp = new TApplication("App",&argc,argv);
      theApp->Run();
      simanTSP(verbose);
      delete theApp;
      theApp = 0;
   }
   else
   {
      simanTSP(verbose);
   }

   // to check that the result is correct
   // FullSearch();
   return 0;
}
#endif

