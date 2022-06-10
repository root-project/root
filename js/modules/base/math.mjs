/**
 * A math namespace - all functions can be exported from base/math.mjs.
 * Also all these functions can be used with TFormula calcualtions
 * @namespace Math
 */

const kMACHEP  = 1.11022302462515654042363166809e-16,
      kMINLOG  = -708.396418532264078748994506896,
      kMAXLOG  = 709.782712893383973096206318587,
      kMAXSTIR = 108.116855767857671821730036754,
      kBig     = 4.503599627370496e15,
      kBiginv  =  2.22044604925031308085e-16,
      kSqrt2   = 1.41421356237309515,
      M_PI    =  3.14159265358979323846264338328;

/** @summary Polynomialeval function
  * @desc calculates a value of a polynomial of the form:
  * a[0]x^N+a[1]x^(N-1) + ... + a[N]
  * @memberof Math */
function Polynomialeval(x, a, N) {
   if (N==0) return a[0];

   let pom = a[0];
   for (let i = 1; i <= N; ++i)
      pom = pom *x + a[i];
   return pom;
}

/** @summary Polynomial1eval function
  * @desc calculates a value of a polynomial of the form:
  * x^N+a[0]x^(N-1) + ... + a[N-1]
  * @memberof Math */
function Polynomial1eval(x, a, N) {
   if (N==0) return a[0];

   let pom = x + a[0];
   for (let i = 1; i < N; ++i)
      pom = pom *x + a[i];
   return pom;
}

/** @summary lgam function, logarithm from gamma
  * @memberof Math */
function lgam(x) {
   let p, q, u, w, z;
   const kMAXLGM = 2.556348e305,
         LS2PI = 0.91893853320467274178,
   A = [
      8.11614167470508450300E-4,
      -5.95061904284301438324E-4,
      7.93650340457716943945E-4,
      -2.77777777730099687205E-3,
      8.33333333333331927722E-2
   ], B = [
      -1.37825152569120859100E3,
      -3.88016315134637840924E4,
      -3.31612992738871184744E5,
      -1.16237097492762307383E6,
      -1.72173700820839662146E6,
      -8.53555664245765465627E5
   ], C = [
   /* 1.00000000000000000000E0, */
      -3.51815701436523470549E2,
      -1.70642106651881159223E4,
      -2.20528590553854454839E5,
      -1.13933444367982507207E6,
      -2.53252307177582951285E6,
      -2.01889141433532773231E6
   ];

   if ((x >= Number.MAX_VALUE) || (x == Number.POSITIVE_INFINITY))
      return Number.POSITIVE_INFINITY;

   if ( x < -34.0 ) {
      q = -x;
      w = lgam(q);
      p = Math.floor(q);
      if ( p==q )//_unur_FP_same(p,q)
         return Number.POSITIVE_INFINITY;
      z = q - p;
      if ( z > 0.5 ) {
         p += 1.0;
         z = p - q;
      }
      z = q * Math.sin( Math.PI * z );
      if ( z < 1e-300 )
         return Number.POSITIVE_INFINITY;
      z = Math.log(Math.PI) - Math.log( z ) - w;
      return z;
   }
   if ( x < 13.0 ) {
      z = 1.0;
      p = 0.0;
      u = x;
      while ( u >= 3.0 ) {
         p -= 1.0;
         u = x + p;
         z *= u;
      }
      while ( u < 2.0 ) {
         if ( u < 1e-300 )
            return Number.POSITIVE_INFINITY;
         z /= u;
         p += 1.0;
         u = x + p;
      }
      if ( z < 0.0 ) {
         z = -z;
      }
      if ( u == 2.0 )
         return Math.log(z);
      p -= 2.0;
      x = x + p;
      p = x * Polynomialeval(x, B, 5 ) / Polynomial1eval( x, C, 6);
      return Math.log(z) + p;
   }
   if ( x > kMAXLGM )
      return Number.POSITIVE_INFINITY;

   q = ( x - 0.5 ) * Math.log(x) - x + LS2PI;
   if ( x > 1.0e8 )
      return q;

   p = 1.0/(x*x);
   if ( x >= 1000.0 )
      q += ((7.9365079365079365079365e-4 * p
            - 2.7777777777777777777778e-3) *p
            + 0.0833333333333333333333) / x;
   else
      q += Polynomialeval( p, A, 4 ) / x;
   return q;
}

/** @summary Stirling formula for the gamma function
  * @memberof Math */
function stirf(x) {
   let y, w, v;

   const STIR = [
      7.87311395793093628397E-4,
      -2.29549961613378126380E-4,
      -2.68132617805781232825E-3,
      3.47222221605458667310E-3,
      8.33333333333482257126E-2,
   ], SQTPI = Math.sqrt(2*Math.PI);

   w = 1.0/x;
   w = 1.0 + w * Polynomialeval( w, STIR, 4 );
   y = Math.exp(x);

/*   #define kMAXSTIR kMAXLOG/log(kMAXLOG)  */

   if( x > kMAXSTIR )
   { /* Avoid overflow in pow() */
      v = Math.pow( x, 0.5 * x - 0.25 );
      y = v * (v / y);
   }
   else
   {
      y = Math.pow( x, x - 0.5 ) / y;
   }
   y = SQTPI * y * w;
   return y;
}

/** @summary complementary error function
  * @memberof Math */
function erfc(a) {
   const erfP = [
      2.46196981473530512524E-10,
      5.64189564831068821977E-1,
      7.46321056442269912687E0,
      4.86371970985681366614E1,
      1.96520832956077098242E2,
      5.26445194995477358631E2,
      9.34528527171957607540E2,
      1.02755188689515710272E3,
      5.57535335369399327526E2
   ], erfQ = [
      1.32281951154744992508E1,
      8.67072140885989742329E1,
      3.54937778887819891062E2,
      9.75708501743205489753E2,
      1.82390916687909736289E3,
      2.24633760818710981792E3,
      1.65666309194161350182E3,
      5.57535340817727675546E2
   ], erfR = [
      5.64189583547755073984E-1,
      1.27536670759978104416E0,
      5.01905042251180477414E0,
      6.16021097993053585195E0,
      7.40974269950448939160E0,
      2.97886665372100240670E0
   ], erfS = [
      2.26052863220117276590E0,
      9.39603524938001434673E0,
      1.20489539808096656605E1,
      1.70814450747565897222E1,
      9.60896809063285878198E0,
      3.36907645100081516050E0
   ];

   let p,q,x,y,z;

   if( a < 0.0 )
      x = -a;
   else
      x = a;

   if( x < 1.0 )
      return 1.0 - erf(a);

   z = -a * a;

   if(z < -kMAXLOG)
      return (a < 0) ? 2.0 : 0.0;

   z = Math.exp(z);

   if( x < 8.0 ) {
      p = Polynomialeval( x, erfP, 8 );
      q = Polynomial1eval( x, erfQ, 8 );
   } else {
      p = Polynomialeval( x, erfR, 5 );
      q = Polynomial1eval( x, erfS, 6 );
   }
   y = (z * p)/q;

   if(a < 0)
      y = 2.0 - y;

   if(y == 0)
      return (a < 0) ? 2.0 : 0.0;

   return y;
}

/** @summary error function
  * @memberof Math */
function erf(x) {
   if(Math.abs(x) > 1.0)
      return 1.0 - erfc(x);

   const erfT = [
      9.60497373987051638749E0,
      9.00260197203842689217E1,
      2.23200534594684319226E3,
      7.00332514112805075473E3,
      5.55923013010394962768E4
   ], erfU = [
      3.35617141647503099647E1,
      5.21357949780152679795E2,
      4.59432382970980127987E3,
      2.26290000613890934246E4,
      4.92673942608635921086E4
   ];

   let z = x * x;

   return x * Polynomialeval(z, erfT, 4) / Polynomial1eval(z, erfU, 5);
}

/** @summary lognormal_cdf_c function
  * @memberof Math */
function lognormal_cdf_c(x, m, s, x0) {
   if (x0 === undefined) x0 = 0;
   let z = (Math.log((x-x0))-m)/(s*kSqrt2);
   if (z > 1.)  return 0.5*erfc(z);
   else         return 0.5*(1.0 - erf(z));
}

/** @summary lognormal_cdf_c function
  * @memberof Math */
function lognormal_cdf(x, m, s, x0 = 0) {
   let z = (Math.log((x-x0))-m)/(s*kSqrt2);
   if (z < -1.) return 0.5*erfc(-z);
   else         return 0.5*(1.0 + erf(z));
}

/** @summary normal_cdf_c function
  * @memberof Math */
function normal_cdf_c(x, sigma, x0 = 0) {
   let z = (x-x0)/(sigma*kSqrt2);
   if (z > 1.)  return 0.5*erfc(z);
   else         return 0.5*(1.-erf(z));
}

/** @summary normal_cdf function
  * @memberof Math */
function normal_cdf(x, sigma, x0 = 0) {
   let z = (x-x0)/(sigma*kSqrt2);
   if (z < -1.) return erfc(-z);
   else         return 0.5*(1.0 + erf(z));
}

/** @summary log normal pdf
  * @memberof Math */
function lognormal_pdf(x, m, s, x0 = 0) {
   if ((x-x0) <= 0)
      return 0.0;
   let tmp = (Math.log((x-x0)) - m)/s;
   return 1.0 / ((x-x0) * Math.abs(s) * Math.sqrt(2 * M_PI)) * Math.exp(-(tmp * tmp) /2);
}

/** @summary normal pdf
  * @memberof Math */
function normal_pdf(x, sigma = 1, x0 = 0) {
   let  tmp = (x-x0)/sigma;
   return (1.0/(Math.sqrt(2 * M_PI) * Math.abs(sigma))) * Math.exp(-tmp*tmp/2);
}

/** @summary gamma calculation
  * @memberof Math */
function gamma(x) {
   let p, q, z, i, sgngam = 1;

   if (x >= Number.MAX_VALUE)
      return x;

   q = Math.abs(x);

   if( q > 33.0 )
   {
      if( x < 0.0 )
      {
         p = Math.floor(q);
         if( p == q )
            return Number.POSITIVE_INFINITY;
         i = Math.round(p);
         if( (i & 1) == 0 )
            sgngam = -1;
         z = q - p;
         if( z > 0.5 )
         {
            p += 1.0;
            z = q - p;
         }
         z = q * Math.sin( Math.PI * z );
         if( z == 0 )
         {
            return sgngam > 0 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
         }
         z = Math.abs(z);
         z = Math.PI / (z * stirf(q) );
      }
      else
      {
         z = stirf(x);
      }
      return sgngam * z;
   }

   z = 1.0;
   while( x >= 3.0 )
   {
      x -= 1.0;
      z *= x;
   }

  let small = false;

   while(( x < 0.0 ) && !small)
   {
      if( x > -1.E-9 )
         small = true;
      else {
         z /= x;
         x += 1.0;
      }
   }

   while(( x < 2.0 ) && !small)
   {
      if( x < 1.e-9 )
         small = true;
      else {
         z /= x;
         x += 1.0;
      }
   }

   if (small) {
      if( x == 0 )
         return Number.POSITIVE_INFINITY;
      else
         return z/((1.0 + 0.5772156649015329 * x) * x);
   }

   if( x == 2.0 )
      return z;

   const P = [
      1.60119522476751861407E-4,
      1.19135147006586384913E-3,
      1.04213797561761569935E-2,
      4.76367800457137231464E-2,
      2.07448227648435975150E-1,
      4.94214826801497100753E-1,
      9.99999999999999996796E-1
   ], Q = [
      -2.31581873324120129819E-5,
      5.39605580493303397842E-4,
      -4.45641913851797240494E-3,
      1.18139785222060435552E-2,
      3.58236398605498653373E-2,
      -2.34591795718243348568E-1,
      7.14304917030273074085E-2,
      1.00000000000000000320E0 ];

   x -= 2.0;
   p = Polynomialeval( x, P, 6 );
   q = Polynomialeval( x, Q, 7 );
   return z * p / q;
}

/** @summary ndtri function
  * @memberof Math */
function ndtri(y0) {
   if ( y0 <= 0.0 )
      return Number.NEGATIVE_INFINITY;
   if ( y0 >= 1.0 )
      return Number.POSITIVE_INFINITY;

   const P0 = [
        -5.99633501014107895267E1,
         9.80010754185999661536E1,
        -5.66762857469070293439E1,
         1.39312609387279679503E1,
        -1.23916583867381258016E0
   ], Q0 = [
         1.95448858338141759834E0,
         4.67627912898881538453E0,
         8.63602421390890590575E1,
        -2.25462687854119370527E2,
         2.00260212380060660359E2,
        -8.20372256168333339912E1,
         1.59056225126211695515E1,
        -1.18331621121330003142E0
   ], P1 = [
         4.05544892305962419923E0,
         3.15251094599893866154E1,
         5.71628192246421288162E1,
         4.40805073893200834700E1,
         1.46849561928858024014E1,
         2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4
   ], Q1 = [
         1.57799883256466749731E1,
         4.53907635128879210584E1,
         4.13172038254672030440E1,
         1.50425385692907503408E1,
         2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4
   ], P2 = [
         3.23774891776946035970E0,
         6.91522889068984211695E0,
         3.93881025292474443415E0,
         1.33303460815807542389E0,
         2.01485389549179081538E-1,
         1.23716634817820021358E-2,
         3.01581553508235416007E-4,
         2.65806974686737550832E-6,
         6.23974539184983293730E-9
   ], Q2 = [
         6.02427039364742014255E0,
         3.67983563856160859403E0,
         1.37702099489081330271E0,
         2.16236993594496635890E-1,
         1.34204006088543189037E-2,
         3.28014464682127739104E-4,
         2.89247864745380683936E-6,
         6.79019408009981274425E-9
   ], s2pi = 2.50662827463100050242e0, dd = 0.13533528323661269189;

   let code = 1, y = y0, x, z, y2, x0, x1;

   if ( y > (1.0 - dd) ) {
      y = 1.0 - y;
      code = 0;
   }
   if ( y > dd ) {
      y = y - 0.5;
      y2 = y * y;
      x = y + y * (y2 * Polynomialeval( y2, P0, 4)/ Polynomial1eval( y2, Q0, 8 ));
      x = x * s2pi;
      return x;
   }
   x = Math.sqrt( -2.0 * Math.log(y) );
   x0 = x - Math.log(x)/x;
   z = 1.0/x;
   if ( x < 8.0 )
      x1 = z * Polynomialeval( z, P1, 8 )/ Polynomial1eval( z, Q1, 8 );
   else
      x1 = z * Polynomialeval( z, P2, 8 )/ Polynomial1eval( z, Q2, 8 );
   x = x0 - x1;
   if ( code != 0 )
      x = -x;
   return x;
}

/** @summary normal_quantile function
  * @memberof Math */
function normal_quantile(z, sigma) {
   return  sigma * ndtri(z);
}

/** @summary normal_quantile_c function
  * @memberof Math */
function normal_quantile_c(z, sigma) {
   return - sigma * ndtri(z);
}

/** @summary igamc function
  * @memberof Math */
function igamc(a,x) {
   // LM: for negative values returns 0.0
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)  return 0.0;

   if (x <= 0) return 1.0;

   if((x < 1.0) || (x < a))
      return (1.0 - igam(a,x));

   let ax = a * Math.log(x) - x - lgam(a);
   if( ax < -kMAXLOG )
      return 0.0;

   ax = Math.exp(ax);

   /* continued fraction */
   let y = 1.0 - a,
       z = x + y + 1.0,
       c = 0.0,
       pkm2 = 1.0,
       qkm2 = x,
       pkm1 = x + 1.0,
       qkm1 = z * x,
       ans = pkm1/qkm1,
       yc, r, t, pk,  qk;

   do {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      yc = y * c;
      pk = pkm1 * z  -  pkm2 * yc;
      qk = qkm1 * z  -  qkm2 * yc;
      if(qk)
      {
         r = pk/qk;
         t = Math.abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if( Math.abs(pk) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
   } while( t > kMACHEP );

   return ans * ax;
}

/** @summary igam function
  * @memberof Math */
function igam(a, x) {

   // LM: for negative values returns 1.0 instead of zero
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)  return 1.0;

   if (x <= 0)  return 0.0;

   if( (x > 1.0) && (x > a ) )
      return 1.0 - igamc(a,x);

   /* Compute  x**a * exp(-x) / gamma(a)  */
   let ax = a * Math.log(x) - x - lgam(a);
   if( ax < -kMAXLOG )
      return 0.0;

   ax = Math.exp(ax);

   /* power series */
   let r = a, c = 1.0, ans = 1.0;

   do {
      r += 1.0;
      c *= x/r;
      ans += c;
   } while( c/ans > kMACHEP );

   return ans * ax/a;
}


/** @summary igami function
  * @memberof Math */
function igami(a, y0) {
   // check the domain
   if (a <= 0) {
      console.error("igami : Wrong domain for parameter a (must be > 0)");
      return 0;
   }
   if (y0 <= 0) {
      return Number.POSITIVE_INFINITY;
   }
   if (y0 >= 1) {
      return 0;
   }
   const kMAXNUM = Number.MAX_VALUE;
   let x0 = kMAXNUM, x1 = 0, x, yl = 0, yh = 1, y, d, lgm, dithresh = 5.0 * kMACHEP, i, dir;

   /* approximation to inverse function */
   d = 1.0/(9.0*a);
   y = ( 1.0 - d - ndtri(y0) * Math.sqrt(d) );
   x = a * y * y * y;

   lgm = lgam(a);

   for( i=0; i<10; ++i ) {
      if ( x > x0 || x < x1 )
         break;
      y = igamc(a,x);
      if ( y < yl || y > yh )
         break;
      if ( y < y0 ) {
         x0 = x;
         yl = y;
      }
      else {
         x1 = x;
         yh = y;
      }
      /* compute the derivative of the function at this point */
      d = (a - 1.0) * Math.log(x) - x - lgm;
      if ( d < -kMAXLOG )
         break;
      d = -Math.exp(d);
      /* compute the step to the next approximation of x */
      d = (y - y0)/d;
      if ( Math.abs(d/x) < kMACHEP )
         return x;
      x = x - d;
   }
   /* Resort to interval halving if Newton iteration did not converge. */
   d = 0.0625;
   if ( x0 == kMAXNUM ) {
      if ( x <= 0.0 )
         x = 1.0;
      while ( x0 == kMAXNUM ) {
         x = (1.0 + d) * x;
         y = igamc( a, x );
         if ( y < y0 ) {
            x0 = x;
            yl = y;
            break;
         }
         d = d + d;
      }
   }
   d = 0.5;
   dir = 0;

   for( i=0; i<400; ++i ) {
      x = x1  +  d * (x0 - x1);
      y = igamc( a, x );
      lgm = (x0 - x1)/(x1 + x0);
      if ( Math.abs(lgm) < dithresh )
         break;
      lgm = (y - y0)/y0;
      if ( Math.abs(lgm) < dithresh )
         break;
      if ( x <= 0.0 )
         break;
      if ( y >= y0 ) {
         x1 = x;
         yh = y;
         if ( dir < 0 ) {
            dir = 0;
            d = 0.5;
         }
         else if ( dir > 1 )
            d = 0.5 * d + 0.5;
         else
            d = (y0 - yl)/(yh - yl);
         dir += 1;
      }
      else {
         x0 = x;
         yl = y;
         if ( dir > 0 ) {
            dir = 0;
            d = 0.5;
         }
         else if ( dir < -1 )
            d = 0.5 * d;
         else
            d = (y0 - yl)/(yh - yl);
         dir -= 1;
      }
   }
   return x;
}

/** @summary landau_pdf function
  * @desc LANDAU pdf : algorithm from CERNLIB G110 denlan
  *  same algorithm is used in GSL
  * @memberof Math */
function landau_pdf(x, xi, x0) {
   if (x0===undefined) x0 = 0;
   if (xi <= 0) return 0;
   const v = (x - x0)/xi;
   let u, ue, us, denlan;
   const p1 = [0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253],
         q1 = [1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063],
         p2 = [0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211],
         q2 = [1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714],
         p3 = [0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101],
         q3 = [1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675],
         p4 = [0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186],
         q4 = [1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511],
         p5 = [1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910],
         q5 = [1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357],
         p6 = [1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109],
         q6 = [1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939],
         a1 = [0.04166666667,-0.01996527778, 0.02709538966],
         a2 = [-1.845568670,-4.284640743];

   if (v < -5.5) {
      u   = Math.exp(v+1.0);
      if (u < 1e-10) return 0.0;
      ue  = Math.exp(-1/u);
      us  = Math.sqrt(u);
      denlan = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
   } else if(v < -1) {
      u   = Math.exp(-v-1);
      denlan = Math.exp(-u)*Math.sqrt(u)*
         (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
         (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
   } else if(v < 1) {
      denlan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
         (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
   } else if(v < 5) {
      denlan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
         (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
   } else if(v < 12) {
      u   = 1/v;
      denlan = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
         (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
   } else if(v < 50) {
      u   = 1/v;
      denlan = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
         (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
   } else if(v < 300) {
      u   = 1/v;
      denlan = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
         (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
   } else {
      u   = 1/(v-v*Math.log(v)/(v+1));
      denlan = u*u*(1+(a2[0]+a2[1]*u)*u);
   }
   return denlan/xi;
}

/** @summary Landau function
  * @memberof Math */
function Landau(x, mpv, sigma, norm) {
   if (sigma <= 0) return 0;
   const den = landau_pdf((x - mpv) / sigma, 1, 0);
   if (!norm) return den;
   return den/sigma;
}

/** @summary inc_gamma_c
  * @memberof Math */
function inc_gamma_c(a,x) {
   return igamc(a,x);
}

/** @summary inc_gamma
  * @memberof Math */
function inc_gamma(a,x) {
   return igam(a,x);
}

/** @summary lgamma
  * @memberof Math */
function lgamma(z) {
   return lgam(z);
}

/** @summary Probability density function of the beta distribution.
  * @memberof Math */
function beta_pdf(x, a, b) {
  if (x < 0 || x > 1.0) return 0;
  if (x == 0 ) {
     if (a < 1) return Number.POSITIVE_INFINITY;
     else if (a > 1) return  0;
     else if ( a == 1) return b; // to avoid a nan from log(0)*0
   }
   if (x == 1 ) {
      if (b < 1) return Number.POSITIVE_INFINITY;
      else if (b > 1) return  0;
      else if ( b == 1) return a; // to avoid a nan from log(0)*0
   }
   return Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
                    Math.log(x) * (a -1.) + Math.log1p(-x) * (b - 1.));
}

/** @summary beta
  * @memberof Math */
function beta(x,y) {
   return Math.exp(lgamma(x)+lgamma(y)-lgamma(x+y));
}

/** @summary chisquared_cdf_c
  * @memberof Math */
function chisquared_cdf_c(x,r,x0) {
   if (x0===undefined) x0 = 0;
   return inc_gamma_c ( 0.5 * r , 0.5*(x-x0) );
}

/** @summary Continued fraction expansion #1 for incomplete beta integral
  * @memberof Math */
function incbcf(a,b,x) {
   let xk, pk, pkm1, pkm2, qk, qkm1, qkm2,
       k1, k2, k3, k4, k5, k6, k7, k8,
       r, t, ans, thresh, n;

   k1 = a;
   k2 = a + b;
   k3 = a;
   k4 = a + 1.0;
   k5 = 1.0;
   k6 = b - 1.0;
   k7 = k4;
   k8 = a + 2.0;

   pkm2 = 0.0;
   qkm2 = 1.0;
   pkm1 = 1.0;
   qkm1 = 1.0;
   ans = 1.0;
   r = 1.0;
   n = 0;
   thresh = 3.0 * kMACHEP;
   do
   {

      xk = -( x * k1 * k2 )/( k3 * k4 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = ( x * k5 * k6 )/( k7 * k8 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if( qk !=0 )
         r = pk/qk;
      if( r != 0 )
      {
         t = Math.abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;

      if( t < thresh )
         break; // goto cdone;

      k1 += 1.0;
      k2 += 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 -= 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if( (Math.abs(qk) + Math.abs(pk)) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
      if( (Math.abs(qk) < kBiginv) || (Math.abs(pk) < kBiginv) )
      {
         pkm2 *= kBig;
         pkm1 *= kBig;
         qkm2 *= kBig;
         qkm1 *= kBig;
      }
   }
   while( ++n < 300 );

// cdone:
   return ans;
}

/** @summary Continued fraction expansion #2 for incomplete beta integral
  * @memberof Math */
function incbd(a,b,x) {
   let xk, pk, pkm1, pkm2, qk, qkm1, qkm2,
       k1, k2, k3, k4, k5, k6, k7, k8,
       r, t, ans, z, thresh, n;

   k1 = a;
   k2 = b - 1.0;
   k3 = a;
   k4 = a + 1.0;
   k5 = 1.0;
   k6 = a + b;
   k7 = a + 1.0;;
   k8 = a + 2.0;

   pkm2 = 0.0;
   qkm2 = 1.0;
   pkm1 = 1.0;
   qkm1 = 1.0;
   z = x / (1.0-x);
   ans = 1.0;
   r = 1.0;
   n = 0;
   thresh = 3.0 * kMACHEP;
   do
   {

      xk = -( z * k1 * k2 )/( k3 * k4 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = ( z * k5 * k6 )/( k7 * k8 );
      pk = pkm1 +  pkm2 * xk;
      qk = qkm1 +  qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if( qk != 0 )
         r = pk/qk;
      if( r != 0 )
      {
         t = Math.abs( (ans - r)/r );
         ans = r;
      }
      else
         t = 1.0;

      if( t < thresh )
         break; // goto cdone;

      k1 += 1.0;
      k2 -= 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 += 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if( (Math.abs(qk) + Math.abs(pk)) > kBig )
      {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
      if( (Math.abs(qk) < kBiginv) || (Math.abs(pk) < kBiginv) )
      {
         pkm2 *= kBig;
         pkm1 *= kBig;
         qkm2 *= kBig;
         qkm1 *= kBig;
      }
   }
   while( ++n < 300 );
//cdone:
   return ans;
}

/** @summary ROOT::Math::Cephes::pseries
  * @memberof Math */
function pseries(a,b,x) {
   let s, t, u, v, n, t1, z, ai;

   ai = 1.0 / a;
   u = (1.0 - b) * x;
   v = u / (a + 1.0);
   t1 = v;
   t = u;
   n = 2.0;
   s = 0.0;
   z = kMACHEP * ai;
   while( Math.abs(v) > z )
   {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v;
      n += 1.0;
   }
   s += t1;
   s += ai;

   u = a * Math.log(x);
   if( (a+b) < kMAXSTIR && Math.abs(u) < kMAXLOG )
   {
      t = gamma(a+b) / (gamma(a)*gamma(b));
      s = s * t * Math.pow(x,a);
   }
   else
   {
      t = lgam(a+b) - lgam(a) - lgam(b) + u + Math.log(s);
      if( t < kMINLOG )
         s = 0.0;
      else
         s = Math.exp(t);
   }
   return s;
}

/** @summary ROOT::Math::Cephes::incbet
  * @memberof Math */
function incbet(aa,bb,xx) {
   let a, b, t, x, xc, w, y, flag;

   if( aa <= 0.0 || bb <= 0.0 )
      return 0.0;

   // LM: changed: for X > 1 return 1.
   if  (xx <= 0.0)  return 0.0;
   if ( xx >= 1.0)  return 1.0;

   flag = 0;

/* - to test if that way is better for large b/  (comment out from Cephes version)
   if( (bb * xx) <= 1.0 && xx <= 0.95)
   {
   t = pseries(aa, bb, xx);
   goto done;
   }

**/
   w = 1.0 - xx;

/* Reverse a and b if x is greater than the mean. */
/* aa,bb > 1 -> sharp rise at x=aa/(aa+bb) */
   if( xx > (aa/(aa+bb)) )
   {
      flag = 1;
      a = bb;
      b = aa;
      xc = xx;
      x = w;
   }
   else
   {
      a = aa;
      b = bb;
      xc = w;
      x = xx;
   }

   if( flag == 1 && (b * x) <= 1.0 && x <= 0.95)
   {
      t = pseries(a, b, x);
      // goto done;
   } else {

   /* Choose expansion for better convergence. */
      y = x * (a+b-2.0) - (a-1.0);
      if( y < 0.0 )
         w = incbcf( a, b, x );
      else
         w = incbd( a, b, x ) / xc;

   /* Multiply w by the factor
      a      b   _             _     _
      x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

      y = a * Math.log(x);
      t = b * Math.log(xc);
      if( (a+b) < kMAXSTIR && Math.abs(y) < kMAXLOG && Math.abs(t) < kMAXLOG )
      {
         t = Math.pow(xc,b);
         t *= Math.pow(x,a);
         t /= a;
         t *= w;
         t *= gamma(a+b) / (gamma(a) * gamma(b));
         // goto done;
      } else {
      /* Resort to logarithms.  */
         y += t + lgam(a+b) - lgam(a) - lgam(b);
         y += Math.log(w/a);
         if( y < kMINLOG )
            t = 0.0;
         else
            t = Math.exp(y);
      }
   }

//done:

   if( flag == 1 )
   {
      if( t <= kMACHEP )
         t = 1.0 - kMACHEP;
      else
         t = 1.0 - t;
   }
   return  t;
}

/** @summary copy of ROOT::Math::Cephes::incbi
  * @memberof Math */
function incbi(aa,bb,yy0) {
   let a, b, y0, d, y, x, x0, x1, lgm, yp, di, dithresh, yl, yh, xt;
   let i, rflg, dir, nflg, ihalve = true;

   // check the domain
   if (aa <= 0) {
      // MATH_ERROR_MSG("Cephes::incbi","Wrong domain for parameter a (must be > 0)");
      return 0;
   }
   if (bb <= 0) {
      // MATH_ERROR_MSG("Cephes::incbi","Wrong domain for parameter b (must be > 0)");
      return 0;
   }

   const process_done = () => {
      if( rflg ) {
         if( x <= kMACHEP )
            x = 1.0 - kMACHEP;
         else
            x = 1.0 - x;
      }
      return x;
   };

   i = 0;
   if( yy0 <= 0 )
      return 0.0;
   if( yy0 >= 1.0 )
      return 1.0;
   x0 = 0.0;
   yl = 0.0;
   x1 = 1.0;
   yh = 1.0;
   nflg = 0;

   if( aa <= 1.0 || bb <= 1.0 )
   {
      dithresh = 1.0e-6;
      rflg = 0;
      a = aa;
      b = bb;
      y0 = yy0;
      x = a/(a+b);
      y = incbet( a, b, x );
      // goto ihalve; // will start
   }
   else
   {
      dithresh = 1.0e-4;
/* approximation to inverse function */

      yp = -ndtri(yy0);

      if( yy0 > 0.5 )
      {
         rflg = 1;
         a = bb;
         b = aa;
         y0 = 1.0 - yy0;
         yp = -yp;
      }
      else
      {
         rflg = 0;
         a = aa;
         b = bb;
         y0 = yy0;
      }

      lgm = (yp * yp - 3.0)/6.0;
      x = 2.0/( 1.0/(2.0*a-1.0)  +  1.0/(2.0*b-1.0) );
      d = yp * Math.sqrt( x + lgm ) / x
         - ( 1.0/(2.0*b-1.0) - 1.0/(2.0*a-1.0) )
         * (lgm + 5.0/6.0 - 2.0/(3.0*x));
      d = 2.0 * d;
      if( d < kMINLOG )
      {
         // x = 1.0;
         // goto under;
         x = 0.0;
         return process_done();
      }
      x = a/( a + b * Math.exp(d) );
      y = incbet( a, b, x );
      yp = (y - y0)/y0;
      if( Math.abs(yp) < 0.2 )
         ihalve = false; // instead goto newt; exclude ihalve for the first time
   }

  let mainloop = 1000;

  // endless loop until coverage
  while (mainloop-- > 0) {

   /* Resort to interval halving if not close enough. */
   // ihalve:
      while(ihalve) {

         dir = 0;
         di = 0.5;
         for( i=0; i<100; i++ )
         {
            if( i != 0 )
            {
               x = x0  +  di * (x1 - x0);
               if( x == 1.0 )
                  x = 1.0 - kMACHEP;
               if( x == 0.0 )
               {
                  di = 0.5;
                  x = x0  +  di * (x1 - x0);
                  if( x == 0.0 )
                     return process_done(); // goto under;
               }
               y = incbet( a, b, x );
               yp = (x1 - x0)/(x1 + x0);
               if( Math.abs(yp) < dithresh )
                  break; // goto newt;
               yp = (y-y0)/y0;
               if( Math.abs(yp) < dithresh )
                  break; // goto newt;
            }
            if( y < y0 )
            {
               x0 = x;
               yl = y;
               if( dir < 0 )
               {
                  dir = 0;
                  di = 0.5;
               }
               else if( dir > 3 )
                  di = 1.0 - (1.0 - di) * (1.0 - di);
               else if( dir > 1 )
                  di = 0.5 * di + 0.5;
               else
                  di = (y0 - y)/(yh - yl);
               dir += 1;
               if( x0 > 0.75 )
               {
                  if( rflg == 1 )
                  {
                     rflg = 0;
                     a = aa;
                     b = bb;
                     y0 = yy0;
                  }
                  else
                  {
                     rflg = 1;
                     a = bb;
                     b = aa;
                     y0 = 1.0 - yy0;
                  }
                  x = 1.0 - x;
                  y = incbet( a, b, x );
                  x0 = 0.0;
                  yl = 0.0;
                  x1 = 1.0;
                  yh = 1.0;
                  continue; // goto ihalve;
               }
            }
            else
            {
               x1 = x;
               if( rflg == 1 && x1 < kMACHEP )
               {
                  x = 0.0;
                  return process_done(); // goto done;
               }
               yh = y;
               if( dir > 0 )
               {
                  dir = 0;
                  di = 0.5;
               }
               else if( dir < -3 )
                  di = di * di;
               else if( dir < -1 )
                  di = 0.5 * di;
               else
                  di = (y - y0)/(yh - yl);
               dir -= 1;
            }
         }
         //math_error( "incbi", PLOSS );
         if( x0 >= 1.0 ) {
            x = 1.0 - kMACHEP;
            return process_done(); //goto done;
         }
         if( x <= 0.0 ) {
            //math_error( "incbi", UNDERFLOW );
            x = 0.0;
            return process_done(); //goto done;
         }
         break; // if here, break ihalve

      } // end of ihalve

      ihalve = true; // enter loop next time

   // newt:

      if( nflg )
         return process_done(); //goto done;
      nflg = 1;
      lgm = lgam(a+b) - lgam(a) - lgam(b);

      for( i=0; i<8; i++ )
      {
         /* Compute the function at this point. */
         if( i != 0 )
            y = incbet(a,b,x);
         if( y < yl )
         {
            x = x0;
            y = yl;
         }
         else if( y > yh )
         {
            x = x1;
            y = yh;
         }
         else if( y < y0 )
         {
            x0 = x;
            yl = y;
         }
         else
         {
            x1 = x;
            yh = y;
         }
         if( x == 1.0 || x == 0.0 )
            break;
         /* Compute the derivative of the function at this point. */
         d = (a - 1.0) * Math.log(x) + (b - 1.0) * Math.log(1.0-x) + lgm;
         if( d < kMINLOG )
            return process_done(); // goto done;
         if( d > kMAXLOG )
            break;
         d = Math.exp(d);
         /* Compute the step to the next approximation of x. */
         d = (y - y0)/d;
         xt = x - d;
         if( xt <= x0 )
         {
            y = (x - x0) / (x1 - x0);
            xt = x0 + 0.5 * y * (x - x0);
            if( xt <= 0.0 )
               break;
         }
         if( xt >= x1 )
         {
            y = (x1 - x) / (x1 - x0);
            xt = x1 - 0.5 * y * (x1 - x);
            if( xt >= 1.0 )
               break;
         }
         x = xt;
         if( Math.abs(d/x) < 128.0 * kMACHEP )
            return process_done(); // goto done;
      }
   /* Did not converge.  */
      dithresh = 256.0 * kMACHEP;

   } // endless loop instead of // goto ihalve;

// done:

   return process_done();
}

/** @summary Calculates the normalized (regularized) incomplete beta function.
  * @memberof Math */
function inc_beta(x,a,b) {
   return incbet(a,b,x);
}

const BetaIncomplete = inc_beta;

/** @summary ROOT::Math::beta_quantile
  * @memberof Math */
function beta_quantile(z,a,b) {
   return incbi(a,b,z);
}

/** @summary Complement of the cumulative distribution function of the beta distribution.
  * @memberof Math */
function beta_cdf_c(x,a,b) {
   return inc_beta(1-x, b, a);
}

/** @summary chisquared_cdf
  * @memberof Math */
function chisquared_cdf(x,r,x0=0) {
   return inc_gamma ( 0.5 * r , 0.5*(x-x0) );
}

/** @summary gamma_quantile_c function
  * @memberof Math */
function gamma_quantile_c(z, alpha, theta) {
   return theta * igami( alpha, z);
}

/** @summary gamma_quantile function
  * @memberof Math */
function gamma_quantile(z, alpha, theta) {
   return theta * igami( alpha, 1.- z);
}

/** @summary breitwigner_cdf_c function
  * @memberof Math */
function breitwigner_cdf_c(x,gamma, x0 = 0) {
   return 0.5 - Math.atan(2.0 * (x-x0) / gamma) / M_PI;
}

/** @summary breitwigner_cdf function
  * @memberof Math */
function breitwigner_cdf(x, gamma, x0 = 0) {
   return 0.5 + Math.atan(2.0 * (x-x0) / gamma) / M_PI;
}

/** @summary cauchy_cdf_c function
  * @memberof Math */
function cauchy_cdf_c(x, b, x0 = 0) {
   return 0.5 - Math.atan( (x-x0) / b) / M_PI;
}

/** @summary cauchy_cdf function
  * @memberof Math */
function cauchy_cdf(x, b, x0 = 0) {
   return 0.5 + Math.atan( (x-x0) / b) / M_PI;
}

/** @summary cauchy_pdf function
  * @memberof Math */
function cauchy_pdf(x, b = 1, x0 = 0) {
   return b/(M_PI * ((x-x0)*(x-x0) + b*b));
}

/** @summary gaussian_pdf function
  * @memberof Math */
function gaussian_pdf(x, sigma = 1, x0 = 0) {
   let tmp = (x-x0)/sigma;
   return (1.0/(Math.sqrt(2 * M_PI) * Math.abs(sigma))) * Math.exp(-tmp*tmp/2);
}

/** @summary gamma_pdf function
  * @memberof Math */
function gamma_pdf(x, alpha, theta, x0 = 0) {
   if ((x - x0) < 0) {
      return 0.0;
   } else if ((x - x0) == 0) {
      return (alpha == 1) ? 1.0 / theta : 0;
   } else if (alpha == 1) {
      return Math.exp(-(x - x0) / theta) / theta;
   }
   return Math.exp((alpha - 1) * Math.log((x - x0) / theta) - (x - x0) / theta - lgamma(alpha)) / theta;
}

/** @summary tdistribution_cdf_c function
  * @memberof Math */
function tdistribution_cdf_c(x, r, x0 = 0) {
   let p    = x - x0,
       sign = (p > 0) ? 1. : -1;
   return .5 - .5*inc_beta(p*p/(r + p*p), .5, .5*r)*sign;
}

/** @summary tdistribution_cdf function
  * @memberof Math */
function tdistribution_cdf(x, r, x0 = 0) {
   let p    = x - x0,
       sign = (p > 0) ? 1. : -1;
   return  .5 + .5*inc_beta(p*p/(r + p*p), .5, .5*r)*sign;
}

/** @summary tdistribution_pdf function
  * @memberof Math */
function tdistribution_pdf(x, r, x0 = 0) {
   return (Math.exp (lgamma((r + 1.0)/2.0) - lgamma(r/2.0)) / Math.sqrt (M_PI * r))
          * Math.pow ((1.0 + (x-x0)*(x-x0)/r), -(r + 1.0)/2.0);
}

/** @summary exponential_cdf_c function
  * @memberof Math */
function exponential_cdf_c(x, lambda, x0 = 0) {
   return ((x-x0) < 0) ? 1.0 : Math.exp(-lambda * (x-x0));
}

/** @summary exponential_cdf function
  * @memberof Math */
function exponential_cdf(x, lambda, x0 = 0) {
   return ((x-x0) < 0) ? 0.0 : -Math.expm1(-lambda * (x-x0));
}

/** @summary chisquared_pdf
  * @memberof Math */
function chisquared_pdf(x,r,x0) {
   if (x0===undefined) x0 = 0;
   if ((x-x0) < 0) return 0.0;
   const a = r/2 -1.;
   // let return inf for case x  = x0 and treat special case of r = 2 otherwise will return nan
   if (x == x0 && a == 0) return 0.5;

   return Math.exp ((r/2 - 1) * Math.log((x-x0)/2) - (x-x0)/2 - lgamma(r/2))/2;
}

/** @summary Probability density function of the F-distribution.
  * @memberof Math */
function fdistribution_pdf(x, n, m, x0 = 0) {
   if (n < 0 || m < 0)
      return Number.NaN;
   if ((x-x0) < 0)
      return 0.0;

   return Math.exp((n/2) * Math.log(n) + (m/2) * Math.log(m) + lgamma((n+m)/2) - lgamma(n/2) - lgamma(m/2)
                 + (n/2 -1) * Math.log(x-x0) - ((n+m)/2) * Math.log(m +  n*(x-x0)) );
}

/** @summary fdistribution_cdf_c function
  * @memberof Math */
function fdistribution_cdf_c(x, n, m, x0 = 0) {
   if (n < 0 || m < 0) return Number.NaN;

   let z = m / (m + n * (x - x0));
   // fox z->1 and large a and b IB looses precision use complement function
   if (z > 0.9 && n > 1 && m > 1) return 1. - fdistribution_cdf(x, n, m, x0);

   // for the complement use the fact that IB(x,a,b) = 1. - IB(1-x,b,a)
   return inc_beta(m / (m + n * (x - x0)), .5 * m, .5 * n);
}

/** @summary fdistribution_cdf function
  * @memberof Math */
function fdistribution_cdf(x, n, m, x0 = 0) {
   if (n < 0 || m < 0) return Number.NaN;

   let z = n * (x - x0) / (m + n * (x - x0));
   // fox z->1 and large a and b IB looses precision use complement function
   if (z > 0.9 && n > 1 && m > 1)
      return 1. - fdistribution_cdf_c(x, n, m, x0);

   return inc_beta(z, .5 * n, .5 * m);
}

/** @summary Prob function
  * @memberof Math */
function Prob(chi2, ndf) {
   if (ndf <= 0) return 0; // Set CL to zero in case ndf<=0

   if (chi2 <= 0) {
      if (chi2 < 0) return 0;
      else          return 1;
   }

   return chisquared_cdf_c(chi2,ndf,0);
}

/** @summary Gaus function
  * @memberof Math */
function Gaus(x, mean, sigma) {
   return Math.exp(-0.5 * Math.pow((x-mean) / sigma, 2));
}

/** @summary BreitWigner function
  * @memberof Math */
function BreitWigner(x, mean, gamma) {
   return gamma/((x-mean)*(x-mean) + gamma*gamma/4) / 2 / Math.PI;
}

/** @summary Calculates Beta-function Gamma(p)*Gamma(q)/Gamma(p+q).
  * @memberof Math */
function Beta(x,y) {
   return Math.exp(lgamma(x) + lgamma(y) - lgamma(x+y));
}

/** @summary GammaDist function
  * @memberof Math */
function GammaDist(x, gamma, mu = 0, beta = 1) {
   if ((x < mu) || (gamma <= 0) || (beta <= 0)) return 0;
   return gamma_pdf(x, gamma, beta, mu);
}

/** @summary probability density function of Laplace distribution
  * @memberof Math */
function LaplaceDist(x, alpha = 0, beta = 1) {
   return Math.exp(-Math.abs((x-alpha)/beta)) / (2.*beta);
}

/** @summary distribution function of Laplace distribution
  * @memberof Math */
function LaplaceDistI(x, alpha = 0, beta = 1) {
   return (x <= alpha) ? 0.5*Math.exp(-Math.abs((x-alpha)/beta)) : 1 - 0.5*Math.exp(-Math.abs((x-alpha)/beta));
}

/** @summary density function for Student's t- distribution
  * @memberof Math */
function Student(T, ndf) {
   if (ndf < 1) return 0;

   let r   = ndf,
       rh  = 0.5*r,
       rh1 = rh + 0.5,
       denom = Math.sqrt(r*Math.PI)*gamma(rh)*Math.pow(1+T*T/r, rh1);
   return gamma(rh1)/denom;
}

/** @summary cumulative distribution function of Student's
  * @memberof Math */
function StudentI(T, ndf) {
   let r = ndf;

   return (T > 0) ? (1 - 0.5*BetaIncomplete((r/(r + T*T)), r*0.5, 0.5))
                  :  0.5*BetaIncomplete((r/(r + T*T)), r*0.5, 0.5);
}

/** @summary LogNormal function
  * @memberof Math */
function LogNormal(x, sigma, theta = 0, m = 1) {
   if ((x < theta) || (sigma <= 0) || (m <= 0)) return 0;
   return lognormal_pdf(x, Math.log(m), sigma, theta);
}

/** @summary Computes the probability density function of the Beta distribution
  * @memberof Math */
function BetaDist(x, p, q) {
   if ((x < 0) || (x > 1) || (p <= 0) || (q <= 0))
     return 0;
   let beta = Beta(p, q);
   return Math.pow(x, p-1) * Math.pow(1-x, q-1) / beta;
}

/** @summary Computes the distribution function of the Beta distribution.
  * @memberof Math */
function BetaDistI(x, p, q) {
   if ((x<0) || (x>1) || (p<=0) || (q<=0)) return 0;
   return BetaIncomplete(x, p, q);
}

/** @summary gaus function for TFormula
  * @memberof Math */
function gaus(f, x, i) {
   return f.GetParValue(i+0) * Math.exp(-0.5 * Math.pow((x-f.GetParValue(i+1)) / f.GetParValue(i+2), 2));
}

/** @summary gausn function for TFormula
  * @memberof Math */
function gausn(f, x, i) {
   return gaus(f, x, i)/(Math.sqrt(2 * Math.PI) * f.GetParValue(i+2));
}

/** @summary gausxy function for TFormula
  * @memberof Math */
function gausxy(f, x, y, i) {
   return f.GetParValue(i+0) * Math.exp(-0.5 * Math.pow((x-f.GetParValue(i+1)) / f.GetParValue(i+2), 2))
                             * Math.exp(-0.5 * Math.pow((y-f.GetParValue(i+3)) / f.GetParValue(i+4), 2));
}

/** @summary expo function for TFormula
  * @memberof Math */
function expo(f, x, i) {
   return Math.exp(f.GetParValue(i+0) + f.GetParValue(i+1) * x);
}

/** @summary landau function for TFormula
  * @memberof Math */
function landau(f, x, i) {
   return Landau(x, f.GetParValue(i+1),f.GetParValue(i+2), false);
}

/** @summary landaun function for TFormula
  * @memberof Math */
function landaun(f, x, i) {
   return Landau(x, f.GetParValue(i+1),f.GetParValue(i+2), true);
}

/** @summary Crystal ball function
  * @memberof Math */
function crystalball_function(x, alpha, n, sigma, mean = 0) {
   if (sigma < 0.)     return 0.;
   let z = (x - mean)/sigma;
   if (alpha < 0) z = -z;
   let abs_alpha = Math.abs(alpha);
   if (z  > - abs_alpha)
      return Math.exp(- 0.5 * z * z);
   let nDivAlpha = n/abs_alpha,
       AA =  Math.exp(-0.5*abs_alpha*abs_alpha),
       B = nDivAlpha - abs_alpha,
       arg = nDivAlpha/(B-z);
  return AA * Math.pow(arg,n);
}

/** @summary pdf definition of the crystal_ball which is defined only for n > 1 otherwise integral is diverging
  * @memberof Math */
function crystalball_pdf(x, alpha, n, sigma, mean = 0) {
   if (sigma < 0.) return 0.;
   if (n <= 1) return Number.NaN;  // pdf is not normalized for n <=1
   let abs_alpha = Math.abs(alpha),
       C = n/abs_alpha * 1./(n-1.) * Math.exp(-alpha*alpha/2.),
       D = Math.sqrt(M_PI/2.)*(1.+erf(abs_alpha/Math.sqrt(2.))),
       N = 1./(sigma*(C+D));
   return N * crystalball_function(x,alpha,n,sigma,mean);
}

/** @summary compute the integral of the crystal ball function
  * @memberof Math */
function crystalball_integral(x, alpha, n, sigma, mean = 0) {
   if (sigma == 0) return 0;
   if (alpha==0) return 0.;
   let useLog = (n == 1.0),
       z = (x-mean)/sigma;
   if (alpha < 0 ) z = -z;

   let abs_alpha = Math.abs(alpha),
       intgaus = 0., intpow  = 0.;

   const sqrtpiover2 = Math.sqrt(M_PI/2.),
         sqrt2pi = Math.sqrt( 2.*M_PI),
         oneoversqrt2 = 1./Math.sqrt(2.);
   if (z <= -abs_alpha) {
      let A = Math.pow(n/abs_alpha,n) * Math.exp(-0.5 * alpha*alpha),
          B = n/abs_alpha - abs_alpha;

      if (!useLog) {
         let C = (n/abs_alpha) * (1./(n-1)) * Math.exp(-alpha*alpha/2.);
         intpow  = C - A /(n-1.) * Math.pow(B-z,-n+1) ;
      }
      else {
         // for n=1 the primitive of 1/x is log(x)
         intpow = -A * Math.log( n / abs_alpha ) + A * Math.log( B -z );
      }
      intgaus =  sqrtpiover2*(1. + erf(abs_alpha*oneoversqrt2));
   }
   else
   {
      intgaus = normal_cdf_c(z, 1);
      intgaus *= sqrt2pi;
      intpow  =  0;
   }
   return sigma * (intgaus + intpow);
}

/** @summary crystalball_cdf function
  * @memberof Math */
function crystalball_cdf(x, alpha, n, sigma, mean = 0) {
   if (n <= 1.)
      return Number.NaN;

   let abs_alpha = Math.abs(alpha),
       C = n/abs_alpha * 1./(n-1.) * Math.exp(-alpha*alpha/2.),
       D = Math.sqrt(M_PI/2.)*(1. + erf(abs_alpha/Math.sqrt(2.))),
       totIntegral = sigma*(C+D),
       integral = crystalball_integral(x,alpha,n,sigma,mean);

   return (alpha > 0) ? 1. - integral/totIntegral : integral/totIntegral;
}

/** @summary crystalball_cdf_c function
  * @memberof Math */
function crystalball_cdf_c(x, alpha, n, sigma, mean = 0) {
   if (n <= 1.)
      return Number.NaN;

   let abs_alpha = Math.abs(alpha),
       C = n/abs_alpha * 1./(n-1.) * Math.exp(-alpha*alpha/2.),
       D = Math.sqrt(M_PI/2.)*(1. + erf(abs_alpha/Math.sqrt(2.))),
       totIntegral = sigma*(C+D),
       integral = crystalball_integral(x,alpha,n,sigma,mean);

   return (alpha > 0) ? integral/totIntegral : 1. - (integral/totIntegral);
}

/** @summary ChebyshevN function
  * @memberof Math */
function ChebyshevN(n, x, c) {
   let d1 = 0.0, d2 = 0.0, y2 = 2.0 * x;

   for (let i = n; i >= 1; i--) {
      let temp = d1;
      d1 = y2 * d1 - d2 + c[i];
      d2 = temp;
   }

   return x * d1 - d2 + c[0];
}

/** @summary Chebyshev1 function
  * @memberof Math */
function Chebyshev1(x, c0, c1) {
   return c0 + c1*x;
}

/** @summary Chebyshev2 function
  * @memberof Math */
function Chebyshev2(x, c0, c1, c2) {
   return c0 + c1*x + c2*(2.0*x*x - 1.0);
}

/** @summary Chebyshev3 function
  * @memberof Math */
function Chebyshev3(x, ...args) {
   return ChebyshevN(3, x, args);
}

/** @summary Chebyshev4 function
  * @memberof Math */
function Chebyshev4(x, ...args) {
   return ChebyshevN(4, x, args);
}

/** @summary Chebyshev5 function
  * @memberof Math */
function Chebyshev5(x, ...args) {
   return ChebyshevN(5, x, args);
}

/** @summary Chebyshev6 function
  * @memberof Math */
function Chebyshev6(x, ...args) {
   return ChebyshevN(6, x, args);
}

/** @summary Chebyshev7 function
  * @memberof Math */
function Chebyshev7(x, ...args) {
   return ChebyshevN(7, x, args);
}

/** @summary Chebyshev8 function
  * @memberof Math */
function Chebyshev8(x, ...args) {
   return ChebyshevN(8, x, args);
}

/** @summary Chebyshev9 function
  * @memberof Math */
function Chebyshev9(x, ...args) {
   return ChebyshevN(9, x, args);
}

/** @summary Chebyshev10 function
  * @memberof Math */
function Chebyshev10(x, ...args) {
   return ChebyshevN(10, x, args);
}

// =========================================================================

/** @summary Caluclate ClopperPearson
  * @memberof Math */
function eff_ClopperPearson(total,passed,level,bUpper) {
   let alpha = (1.0 - level) / 2;
   if(bUpper)
      return ((passed == total) ? 1.0 : beta_quantile(1 - alpha,passed + 1,total-passed));

   return ((passed == 0) ? 0.0 : beta_quantile(alpha,passed,total-passed+1.0));
}

/** @summary Caluclate normal
  * @memberof Math */
function eff_Normal(total,passed,level,bUpper) {
   if (total == 0) return bUpper ? 1 : 0;

   let alpha = (1.0 - level)/2,
       average = passed / total,
       sigma = Math.sqrt(average * (1 - average) / total),
       delta = normal_quantile(1 - alpha, sigma);

   if(bUpper)
      return ((average + delta) > 1) ? 1.0 : (average + delta);

   return ((average - delta) < 0) ? 0.0 : (average - delta);
}

/** @summary Calculates the boundaries for the frequentist Wilson interval
  * @memberof Math */
function eff_Wilson(total,passed,level,bUpper) {
   let alpha = (1.0 - level)/2;
   if (total == 0) return bUpper ? 1 : 0;
   let average = passed / total,
       kappa = normal_quantile(1 - alpha,1),
       mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa),
       delta = kappa / (total + kappa*kappa) * Math.sqrt(total * average * (1 - average) + kappa * kappa / 4);

   if(bUpper)
      return ((mode + delta) > 1) ? 1.0 : (mode + delta);

   return ((mode - delta) < 0) ? 0.0 : (mode - delta);
}

/** @summary Calculates the boundaries for the frequentist Agresti-Coull interval
  * @memberof Math */
function eff_AgrestiCoull(total,passed,level,bUpper) {
   let alpha = (1.0 - level)/2,
       kappa = normal_quantile(1 - alpha,1),
       mode = (passed + 0.5 * kappa * kappa) / (total + kappa * kappa),
       delta = kappa * Math.sqrt(mode * (1 - mode) / (total + kappa * kappa));

  if(bUpper)
     return ((mode + delta) > 1) ? 1.0 : (mode + delta);

  return ((mode - delta) < 0) ? 0.0 : (mode - delta);
}

/** @summary Calculates the boundaries using the  mid-P binomial
  * @memberof Math */
function eff_MidPInterval(total,passed,level,bUpper) {
   const alpha = 1. - level, equal_tailed = true, alpha_min = equal_tailed ? alpha/2 : alpha, tol = 1e-9; // tolerance
   let pmin = 0, pmax = 1, p = 0;

   // treat special case for 0<passed<1
   // do a linear interpolation of the upper limit values
   if ( passed > 0 && passed < 1) {
      let p0 =  eff_MidPInterval(total,0.0,level,bUpper);
      let p1 =  eff_MidPInterval(total,1.0,level,bUpper);
      p = (p1 - p0) * passed + p0;
      return p;
   }

   while (Math.abs(pmax - pmin) > tol) {
      p = (pmin + pmax)/2;
      //double v = 0.5 * ROOT::Math::binomial_pdf(int(passed), p, int(total));
      // make it work for non integer using the binomial - beta relationship
      let v = 0.5 * beta_pdf(p, passed+1., total-passed+1)/(total+1);
      //if (passed > 0) v += ROOT::Math::binomial_cdf(int(passed - 1), p, int(total));
      // compute the binomial cdf at passed -1
      if ( (passed-1) >= 0) v += beta_cdf_c(p, passed, total-passed+1);

      let vmin = bUpper ? alpha_min : 1.- alpha_min;
      if (v > vmin)
         pmin = p;
      else
         pmax = p;
   }

   return p;
}

/** @summary for a central confidence interval for a Beta distribution
  * @memberof Math */
function eff_Bayesian(total,passed,level,bUpper,alpha,beta) {
   let  a = passed + alpha,
        b = total - passed + beta;
   if(bUpper) {
      if((a > 0) && (b > 0))
         return beta_quantile((1+level)/2,a,b);
      else
         return 1;
   } else {
      if((a > 0) && (b > 0))
         return beta_quantile((1-level)/2,a,b);
      else
         return 0;
   }
}

/** @summary Return function to calculate boundary of TEfficiency
  * @memberof Math */
function getTEfficiencyBoundaryFunc(option, isbayessian) {
   const  kFCP = 0,       ///< Clopper-Pearson interval (recommended by PDG)
          kFNormal = 1,   ///< Normal approximation
          kFWilson = 2,   ///< Wilson interval
          kFAC = 3,       ///< Agresti-Coull interval
          kFFC = 4,       ///< Feldman-Cousins interval, too complicated for JavaScript
          // kBJeffrey = 5,  ///< Jeffrey interval (Prior ~ Beta(0.5,0.5)
          // kBUniform = 6,  ///< Prior ~ Uniform = Beta(1,1)
          // kBBayesian = 7, ///< User specified Prior ~ Beta(fBeta_alpha,fBeta_beta)
          kMidP = 8;      ///< Mid-P Lancaster interval

   if (isbayessian)
      return eff_Bayesian;

   switch (option) {
      case kFCP: return eff_ClopperPearson;
      case kFNormal: return eff_Normal;
      case kFWilson: return eff_Wilson;
      case kFAC: return eff_AgrestiCoull;
      case kFFC: console.log("Feldman-Cousins interval kFFC not supported; using kFCP"); return eff_ClopperPearson;
      case kMidP: return eff_MidPInterval;
      // case kBJeffrey:
      // case kBUniform:
      // case kBBayesian: return eff_ClopperPearson;
   }
   console.log(`Not recognized stat option ${option}, using kFCP`);
   return eff_ClopperPearson;
}

export {
   gamma, gamma as tgamma, gamma as Gamma,
   Polynomialeval, Polynomial1eval, stirf,
   gamma_pdf, ndtri, normal_quantile, normal_quantile_c, lognormal_cdf_c, lognormal_cdf,
   igami, igamc, igam, lgam, lgamma, erfc, erf,
   beta_pdf, inc_beta, BetaIncomplete,
   pseries, incbet, incbi, beta_quantile,  chisquared_cdf_c,
   beta, inc_gamma, inc_gamma_c, landau_pdf, beta_cdf_c, Landau,
   fdistribution_pdf, fdistribution_pdf as FDist,
   fdistribution_cdf, fdistribution_cdf as FDistI,
   fdistribution_cdf_c,
   normal_cdf_c, normal_cdf_c as gaussian_cdf_c,
   normal_cdf, normal_cdf as gaussian_cdf,
   lognormal_pdf, normal_pdf, crystalball_function, crystalball_pdf,  crystalball_cdf, crystalball_cdf_c,

   chisquared_cdf, gamma_quantile_c, gamma_quantile, breitwigner_cdf_c, breitwigner_cdf,
   cauchy_cdf_c, cauchy_cdf, cauchy_pdf, gaussian_pdf,
   tdistribution_cdf_c, tdistribution_cdf, tdistribution_pdf, exponential_cdf_c, exponential_cdf, chisquared_pdf,
   Beta, GammaDist, LaplaceDist, LaplaceDistI, LogNormal, Student, StudentI,
   gaus, gausn, gausxy, expo,
   Prob, Gaus, BreitWigner, BetaDist, BetaDistI, landau, landaun,

   ChebyshevN, Chebyshev1, Chebyshev2, Chebyshev3, Chebyshev4, Chebyshev5,
   Chebyshev6, Chebyshev7, Chebyshev8, Chebyshev9, Chebyshev10,


   getTEfficiencyBoundaryFunc
};
