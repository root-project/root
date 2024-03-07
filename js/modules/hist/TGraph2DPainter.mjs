import { settings, createHistogram, setHistogramTitle, kNoZoom,
         clTH2F, clTGraph2DErrors, clTGraph2DAsymmErrors, clTPaletteAxis, kNoStats } from '../core.mjs';
import { Color, DoubleSide, LineBasicMaterial, MeshBasicMaterial, Mesh } from '../three.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TH2Painter } from './TH2Painter.mjs';
import { Triangles3DHandler } from '../hist2d/TH2Painter.mjs';
import { createLineSegments, PointsCreator, getMaterialArgs } from '../base/base3d.mjs';
import { convertLegoBuf, createLegoGeom } from './hist3d.mjs';

function getMax(arr) {
   let v = arr[0];
   for (let i = 1; i < arr.length; ++i)
      v = Math.max(v, arr[i]);
   return v;
}

function getMin(arr) {
   let v = arr[0];
   for (let i = 1; i < arr.length; ++i)
      v = Math.min(v, arr[i]);
   return v;
}

function TMath_Sort(np, values, indicies /*, down */) {
   const arr = new Array(np);
   for (let i = 0; i < np; ++i)
      arr[i] = { v: values[i], i };

   arr.sort((a, b) => { return a.v < b.v ? -1 : (a.v > b.v ? 1 : 0); });

   for (let i = 0; i < np; ++i)
      indicies[i] = arr[i].i;
}

class TGraphDelaunay {

   constructor(g) {
      this.fGraph2D = g;
      this.fX = g.fX;
      this.fY = g.fY;
      this.fZ = g.fZ;
      this.fNpoints = g.fNpoints;
      this.fZout = 0.0;
      this.fNdt = 0;
      this.fNhull = 0;
      this.fHullPoints = null;
      this.fXN = null;
      this.fYN = null;
      this.fOrder = null;
      this.fDist = null;
      this.fPTried = null;
      this.fNTried = null;
      this.fMTried = null;
      this.fInit = false;
      this.fXNmin = 0.0;
      this.fXNmax = 0.0;
      this.fYNmin = 0.0;
      this.fYNmax = 0.0;
      this.fXoffset = 0.0;
      this.fYoffset = 0.0;
      this.fXScaleFactor = 0.0;
      this.fYScaleFactor = 0.0;

      this.SetMaxIter();
   }


   Initialize() {
      if (!this.fInit) {
         this.CreateTrianglesDataStructure();
         this.FindHull();
         this.fInit = true;
      }
   }

   ComputeZ(x, y) {
      // Initialise the Delaunay algorithm if needed.
      // CreateTrianglesDataStructure computes fXoffset, fYoffset,
      // fXScaleFactor and fYScaleFactor;
      // needed in this function.
      this.Initialize();

      // Find the z value corresponding to the point (x,y).
      const xx = (x+this.fXoffset)*this.fXScaleFactor,
            yy = (y+this.fYoffset)*this.fYScaleFactor;
      let zz = this.Interpolate(xx, yy);

      // Wrong zeros may appear when points sit on a regular grid.
      // The following line try to avoid this problem.
      if (zz === 0) zz = this.Interpolate(xx+0.0001, yy);

      return zz;
   }


   CreateTrianglesDataStructure() {
      // Offset fX and fY so they average zero, and scale so the average
      // of the X and Y ranges is one. The normalized version of fX and fY used
      // in Interpolate.
      const xmax = getMax(this.fGraph2D.fX),
            ymax = getMax(this.fGraph2D.fY),
            xmin = getMin(this.fGraph2D.fX),
            ymin = getMin(this.fGraph2D.fY);
      this.fXoffset = -(xmax+xmin)/2;
      this.fYoffset = -(ymax+ymin)/2;
      this.fXScaleFactor = 1/(xmax-xmin);
      this.fYScaleFactor = 1/(ymax-ymin);
      this.fXNmax = (xmax+this.fXoffset)*this.fXScaleFactor;
      this.fXNmin = (xmin+this.fXoffset)*this.fXScaleFactor;
      this.fYNmax = (ymax+this.fYoffset)*this.fYScaleFactor;
      this.fYNmin = (ymin+this.fYoffset)*this.fYScaleFactor;
      this.fXN = new Array(this.fNpoints+1);
      this.fYN = new Array(this.fNpoints+1);
      for (let n = 0; n < this.fNpoints; n++) {
         this.fXN[n+1] = (this.fX[n]+this.fXoffset)*this.fXScaleFactor;
         this.fYN[n+1] = (this.fY[n]+this.fYoffset)*this.fYScaleFactor;
      }

      // If needed, creates the arrays to hold the Delaunay triangles.
      // A maximum number of 2*fNpoints is guessed. If more triangles will be
      // find, FillIt will automatically enlarge these arrays.
      this.fPTried = [];
      this.fNTried = [];
      this.fMTried = [];
   }


   /// Is point e inside the triangle t1-t2-t3 ?

   Enclose(t1, t2, t3, e) {
      const x = [this.fXN[t1], this.fXN[t2], this.fXN[t3], this.fXN[t1]],
            y = [this.fYN[t1], this.fYN[t2], this.fYN[t3], this.fYN[t1]],
            xp = this.fXN[e],
            yp = this.fYN[e];
      let i = 0, j = x.length - 1, oddNodes = false;

      for (; i < x.length; ++i) {
         if ((y[i]<yp && y[j]>=yp) || (y[j]<yp && y[i]>=yp)) {
            if (x[i]+(yp-y[i])/(y[j]-y[i])*(x[j]-x[i])<xp)
               oddNodes = !oddNodes;
         }
         j = i;
      }

      return oddNodes;
   }


   /// Files the triangle defined by the 3 vertices p, n and m into the
   /// fxTried arrays. If these arrays are to small they are automatically
   /// expanded.

   FileIt(p, n, m) {
      let swap, tmp, ps = p, ns = n, ms = m;

      // order the vertices before storing them
      do {
         swap = false;
         if (ns > ps) { tmp = ps; ps = ns; ns = tmp; swap = true; }
         if (ms > ns) { tmp = ns; ns = ms; ms = tmp; swap = true; }
      } while (swap);

      // store a new Delaunay triangle
      this.fNdt++;
      this.fPTried.push(ps);
      this.fNTried.push(ns);
      this.fMTried.push(ms);
   }


   /// Attempt to find all the Delaunay triangles of the point set. It is not
   /// guaranteed that it will fully succeed, and no check is made that it has
   /// fully succeeded (such a check would be possible by referencing the points
   /// that make up the convex hull). The method is to check if each triangle
   /// shares all three of its sides with other triangles. If not, a point is
   /// generated just outside the triangle on the side(s) not shared, and a new
   /// triangle is found for that point. If this method is not working properly
   /// (many triangles are not being found) it's probably because the new points
   /// are too far beyond or too close to the non-shared sides. Fiddling with
   /// the size of the `alittlebit' parameter may help.

   FindAllTriangles() {
      if (this.fAllTri) return;

      this.fAllTri = true;

      let xcntr, ycntr, xm, ym, xx, yy,
          sx, sy, nx, ny, mx, my, mdotn, nn, a,
          t1, t2, pa, na, ma, pb, nb, mb, p1=0, p2=0, m, n, p3=0;
      const s = [false, false, false],
            alittlebit = 0.0001;

      this.Initialize();

      // start with a point that is guaranteed to be inside the hull (the
      // centre of the hull). The starting point is shifted "a little bit"
      // otherwise, in case of triangles aligned on a regular grid, we may
      // found none of them.
      xcntr = 0;
      ycntr = 0;
      for (n = 1; n <= this.fNhull; n++) {
         xcntr += this.fXN[this.fHullPoints[n-1]];
         ycntr += this.fYN[this.fHullPoints[n-1]];
      }
      xcntr = xcntr/this.fNhull+alittlebit;
      ycntr = ycntr/this.fNhull+alittlebit;
      // and calculate it's triangle
      this.Interpolate(xcntr, ycntr);

      // loop over all Delaunay triangles (including those constantly being
      // produced within the loop) and check to see if their 3 sides also
      // correspond to the sides of other Delaunay triangles, i.e. that they
      // have all their neighbours.
      t1 = 1;
      while (t1 <= this.fNdt) {
         // get the three points that make up this triangle
         pa = this.fPTried[t1-1];
         na = this.fNTried[t1-1];
         ma = this.fMTried[t1-1];

         // produce three integers which will represent the three sides
         s[0] = false;
         s[1] = false;
         s[2] = false;
         // loop over all other Delaunay triangles
         for (t2=1; t2<=this.fNdt; t2++) {
            if (t2 !== t1) {
               // get the points that make up this triangle
               pb = this.fPTried[t2-1];
               nb = this.fNTried[t2-1];
               mb = this.fMTried[t2-1];
               // do triangles t1 and t2 share a side?
               if ((pa === pb && na === nb) || (pa === pb && na === mb) || (pa === nb && na === mb)) {
                  // they share side 1
                  s[0] = true;
               } else if ((pa === pb && ma === nb) || (pa === pb && ma === mb) || (pa === nb && ma === mb)) {
                  // they share side 2
                  s[1] = true;
               } else if ((na === pb && ma === nb) || (na === pb && ma === mb) || (na === nb && ma === mb)) {
                  // they share side 3
                  s[2] = true;
               }
            }
            // if t1 shares all its sides with other Delaunay triangles then
            // forget about it
            if (s[0] && s[1] && s[2]) continue;
         }
         // Looks like t1 is missing a neighbour on at least one side.
         // For each side, take a point a little bit beyond it and calculate
         // the Delaunay triangle for that point, this should be the triangle
         // which shares the side.
         for (m=1; m<=3; m++) {
            if (!s[m-1]) {
               // get the two points that make up this side
               if (m === 1) {
                  p1 = pa;
                  p2 = na;
                  p3 = ma;
               } else if (m === 2) {
                  p1 = pa;
                  p2 = ma;
                  p3 = na;
               } else if (m === 3) {
                  p1 = na;
                  p2 = ma;
                  p3 = pa;
               }
               // get the coordinates of the centre of this side
               xm = (this.fXN[p1]+this.fXN[p2])/2.0;
               ym = (this.fYN[p1]+this.fYN[p2])/2.0;
               // we want to add a little to these coordinates to get a point just
               // outside the triangle; (sx,sy) will be the vector that represents
               // the side
               sx = this.fXN[p1]-this.fXN[p2];
               sy = this.fYN[p1]-this.fYN[p2];
               // (nx,ny) will be the normal to the side, but don't know if it's
               // pointing in or out yet
               nx = sy;
               ny = -sx;
               nn = Math.sqrt(nx*nx+ny*ny);
               nx = nx/nn;
               ny = ny/nn;
               mx = this.fXN[p3]-xm;
               my = this.fYN[p3]-ym;
               mdotn = mx*nx+my*ny;
               if (mdotn > 0) {
                  // (nx,ny) is pointing in, we want it pointing out
                  nx = -nx;
                  ny = -ny;
               }
               // increase/decrease xm and ym a little to produce a point
               // just outside the triangle (ensuring that the amount added will
               // be large enough such that it won't be lost in rounding errors)
               a = Math.abs(Math.max(alittlebit*xm, alittlebit*ym));
               xx = xm+nx*a;
               yy = ym+ny*a;
               // try and find a new Delaunay triangle for this point
               this.Interpolate(xx, yy);

               // this side of t1 should now, hopefully, if it's not part of the
               // hull, be shared with a new Delaunay triangle just calculated by Interpolate
            }
         }
         t1++;
      }
   }

   /// Finds those points which make up the convex hull of the set. If the xy
   /// plane were a sheet of wood, and the points were nails hammered into it
   /// at the respective coordinates, then if an elastic band were stretched
   /// over all the nails it would form the shape of the convex hull. Those
   /// nails in contact with it are the points that make up the hull.

   FindHull() {
      if (!this.fHullPoints)
         this.fHullPoints = new Array(this.fNpoints);

      let nhull_tmp = 0;
      for (let n=1; n<=this.fNpoints; n++) {
         // if the point is not inside the hull of the set of all points
         // bar it, then it is part of the hull of the set of all points
         // including it
         const is_in = this.InHull(n, n);
         if (!is_in) {
            // cannot increment fNhull directly - InHull needs to know that
            // the hull has not yet been completely found
            nhull_tmp++;
            this.fHullPoints[nhull_tmp-1] = n;
         }
      }
      this.fNhull = nhull_tmp;
   }


   /// Is point e inside the hull defined by all points apart from x ?

   InHull(e, x) {
      let n1, n2, n, m, ntry,
          lastdphi, dd1, dd2, dx1, dx2, dx3, dy1, dy2, dy3,
          u, v, vNv1, vNv2, phi1, phi2, dphi,
          deTinhull = false;

      const xx = this.fXN[e],
            yy = this.fYN[e];

      if (this.fNhull > 0) {
         //  The hull has been found - no need to use any points other than
         //  those that make up the hull
         ntry = this.fNhull;
      } else {
         //  The hull has not yet been found, will have to try every point
         ntry = this.fNpoints;
      }

      //  n1 and n2 will represent the two points most separated by angle
      //  from point e. Initially the angle between them will be <180 degs.
      //  But subsequent points will increase the n1-e-n2 angle. If it
      //  increases above 180 degrees then point e must be surrounded by
      //  points - it is not part of the hull.
      n1 = 1;
      n2 = 2;
      if (n1 === x) {
         n1 = n2;
         n2++;
      } else if (n2 === x)
         n2++;


      //  Get the angle n1-e-n2 and set it to lastdphi
      dx1 = xx-this.fXN[n1];
      dy1 = yy-this.fYN[n1];
      dx2 = xx-this.fXN[n2];
      dy2 = yy-this.fYN[n2];
      phi1 = Math.atan2(dy1, dx1);
      phi2 = Math.atan2(dy2, dx2);
      dphi = (phi1-phi2)-(Math.floor((phi1-phi2)/(Math.PI*2))*Math.PI*2);
      if (dphi < 0) dphi += Math.PI*2;
      lastdphi = dphi;
      for (n=1; n<=ntry; n++) {
         if (this.fNhull > 0) {
            // Try hull point n
            m = this.fHullPoints[n-1];
         } else
            m = n;

         if ((m !== n1) && (m !== n2) && (m !== x)) {
            // Can the vector e->m be represented as a sum with positive
            // coefficients of vectors e->n1 and e->n2?
            dx1 = xx-this.fXN[n1];
            dy1 = yy-this.fYN[n1];
            dx2 = xx-this.fXN[n2];
            dy2 = yy-this.fYN[n2];
            dx3 = xx-this.fXN[m];
            dy3 = yy-this.fYN[m];

            dd1 = (dx2*dy1-dx1*dy2);
            dd2 = (dx1*dy2-dx2*dy1);

            if (dd1*dd2 !== 0) {
               u = (dx2*dy3-dx3*dy2)/dd1;
               v = (dx1*dy3-dx3*dy1)/dd2;
               if ((u < 0) || (v < 0)) {
                  // No, it cannot - point m does not lie in-between n1 and n2 as
                  // viewed from e. Replace either n1 or n2 to increase the
                  // n1-e-n2 angle. The one to replace is the one which makes the
                  // smallest angle with e->m
                  vNv1 = (dx1*dx3+dy1*dy3)/Math.sqrt(dx1*dx1+dy1*dy1);
                  vNv2 = (dx2*dx3+dy2*dy3)/Math.sqrt(dx2*dx2+dy2*dy2);
                  if (vNv1 > vNv2) {
                     n1 = m;
                     phi1 = Math.atan2(dy3, dx3);
                     phi2 = Math.atan2(dy2, dx2);
                  } else {
                     n2 = m;
                     phi1 = Math.atan2(dy1, dx1);
                     phi2 = Math.atan2(dy3, dx3);
                  }
                  dphi = (phi1-phi2)-(Math.floor((phi1-phi2)/(Math.PI*2))*Math.PI*2);
                  if (dphi < 0) dphi += Math.PI*2;
                  if ((dphi - Math.PI)*(lastdphi - Math.PI) < 0) {
                     // The addition of point m means the angle n1-e-n2 has risen
                     // above 180 degs, the point is in the hull.
                     deTinhull = true;
                     return deTinhull;
                  }
                  lastdphi = dphi;
               }
            }
         }
      }
      // Point e is not surrounded by points - it is not in the hull.
      return deTinhull;
   }


   /// Finds the z-value at point e given that it lies
   /// on the plane defined by t1,t2,t3

   InterpolateOnPlane(TI1, TI2, TI3, e) {
      let tmp, swap, t1 = TI1, t2 = TI2, t3 = TI3;

      // order the vertices
      do {
         swap = false;
         if (t2 > t1) { tmp = t1; t1 = t2; t2 = tmp; swap = true; }
         if (t3 > t2) { tmp = t2; t2 = t3; t3 = tmp; swap = true; }
      } while (swap);

      const x1 = this.fXN[t1],
            x2 = this.fXN[t2],
            x3 = this.fXN[t3],
            y1 = this.fYN[t1],
            y2 = this.fYN[t2],
            y3 = this.fYN[t3],
            f1 = this.fZ[t1-1],
            f2 = this.fZ[t2-1],
            f3 = this.fZ[t3-1],
            u = (f1*(y2-y3)+f2*(y3-y1)+f3*(y1-y2))/(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)),
            v = (f1*(x2-x3)+f2*(x3-x1)+f3*(x1-x2))/(y1*(x2-x3)+y2*(x3-x1)+y3*(x1-x2)),
            w = f1-u*x1-v*y1;

      return u*this.fXN[e] + v*this.fYN[e] + w;
   }

   /// Finds the Delaunay triangle that the point (xi,yi) sits in (if any) and
   /// calculate a z-value for it by linearly interpolating the z-values that
   /// make up that triangle.

   Interpolate(xx, yy) {
      let thevalue,
          it, ntris_tried, p, n, m,
          i, j, k, l, z, f, d, o1, o2, a, b, t1, t2, t3,
          ndegen = 0, degen = 0, fdegen = 0, o1degen = 0, o2degen = 0,
          vxN, vyN,
          d1, d2, d3, c1, c2, dko1, dko2, dfo1,
          dfo2, sin_sum, cfo1k, co2o1k, co2o1f,
          dx1, dx2, dx3, dy1, dy2, dy3, u, v;
      const dxz = [0, 0, 0], dyz = [0, 0, 0];

      // initialise the Delaunay algorithm if needed
      this.Initialize();

      // create vectors needed for sorting
      if (!this.fOrder) {
         this.fOrder = new Array(this.fNpoints);
         this.fDist = new Array(this.fNpoints);
      }

      // the input point will be point zero.
      this.fXN[0] = xx;
      this.fYN[0] = yy;

      // set the output value to the default value for now
      thevalue = this.fZout;

      // some counting
      ntris_tried = 0;

      // no point in proceeding if xx or yy are silly
      if ((xx > this.fXNmax) || (xx < this.fXNmin) || (yy > this.fYNmax) || (yy < this.fYNmin))
         return thevalue;

      // check existing Delaunay triangles for a good one
      for (it=1; it<=this.fNdt; it++) {
         p = this.fPTried[it-1];
         n = this.fNTried[it-1];
         m = this.fMTried[it-1];
         // p, n and m form a previously found Delaunay triangle, does it
         // enclose the point?
         if (this.Enclose(p, n, m, 0)) {
            // yes, we have the triangle
            thevalue = this.InterpolateOnPlane(p, n, m, 0);
            return thevalue;
         }
      }

      // is this point inside the convex hull?
      const shouldbein = this.InHull(0, -1);
      if (!shouldbein)
         return thevalue;

      // it must be in a Delaunay triangle - find it...

      // order mass points by distance in mass plane from desired point
      for (it=1; it<=this.fNpoints; it++) {
         vxN = this.fXN[it];
         vyN = this.fYN[it];
         this.fDist[it-1] = Math.sqrt((xx-vxN)*(xx-vxN)+(yy-vyN)*(yy-vyN));
      }

      // sort array 'fDist' to find closest points
      TMath_Sort(this.fNpoints, this.fDist, this.fOrder /*, false */);
      for (it=0; it<this.fNpoints; it++) this.fOrder[it]++;

      // loop over triplets of close points to try to find a triangle that
      // encloses the point.
      for (k=3; k<=this.fNpoints; k++) {
         m = this.fOrder[k-1];
         for (j=2; j<=k-1; j++) {
            n = this.fOrder[j-1];
            for (i=1; i<=j-1; i++) {
               let skip_this_triangle = false; // used instead of goto L90
               p = this.fOrder[i-1];
               if (ntris_tried > this.fMaxIter) {
                  // perhaps this point isn't in the hull after all
   ///            Warning("Interpolate",
   ///                    "Abandoning the effort to find a Delaunay triangle (and thus interpolated z-value) for point %g %g"
   ///                    ,xx,yy);
                  return thevalue;
               }
               ntris_tried++;
               // check the points aren't colinear
               d1 = Math.sqrt((this.fXN[p]-this.fXN[n])**2+(this.fYN[p]-this.fYN[n])**2);
               d2 = Math.sqrt((this.fXN[p]-this.fXN[m])**2+(this.fYN[p]-this.fYN[m])**2);
               d3 = Math.sqrt((this.fXN[n]-this.fXN[m])**2+(this.fYN[n]-this.fYN[m])**2);
               if ((d1+d2 <= d3) || (d1+d3 <= d2) || (d2+d3 <= d1))
                  continue;

               // does the triangle enclose the point?
               if (!this.Enclose(p, n, m, 0))
                  continue;

               // is it a Delaunay triangle? (ie. are there any other points
               // inside the circle that is defined by its vertices?)

               // test the triangle for Delaunay'ness

               // loop over all other points testing each to see if it's
               // inside the triangle's circle
               ndegen = 0;
               for (z = 1; z <= this.fNpoints; z++) {
                  if ((z === p) || (z === n) || (z === m))
                     continue; // goto L50;
                  // An easy first check is to see if point z is inside the triangle
                  // (if it's in the triangle it's also in the circle)

                  // point z cannot be inside the triangle if it's further from (xx,yy)
                  // than the furthest pointing making up the triangle - test this
                  for (l=1; l<=this.fNpoints; l++) {
                     if (this.fOrder[l-1] === z) {
                        if ((l<i) || (l<j) || (l<k)) {
                           // point z is nearer to (xx,yy) than m, n or p - it could be in the
                           // triangle so call enclose to find out

                           // if it is inside the triangle this can't be a Delaunay triangle
                           if (this.Enclose(p, n, m, z)) { skip_this_triangle = true; break; } // goto L90;
                        } else {
                           // there's no way it could be in the triangle so there's no point
                           // calling enclose
                           break; // goto L1;
                        }
                     }
                  }

                  if (skip_this_triangle) break;

                  // is point z colinear with any pair of the triangle points?
   // L1:
                  if (((this.fXN[p]-this.fXN[z])*(this.fYN[p]-this.fYN[n])) === ((this.fYN[p]-this.fYN[z])*(this.fXN[p]-this.fXN[n]))) {
                     // z is colinear with p and n
                     a = p;
                     b = n;
                  } else if (((this.fXN[p]-this.fXN[z])*(this.fYN[p]-this.fYN[m])) === ((this.fYN[p]-this.fYN[z])*(this.fXN[p]-this.fXN[m]))) {
                     // z is colinear with p and m
                     a = p;
                     b = m;
                  } else if (((this.fXN[n]-this.fXN[z])*(this.fYN[n]-this.fYN[m])) === ((this.fYN[n]-this.fYN[z])*(this.fXN[n]-this.fXN[m]))) {
                     // z is colinear with n and m
                     a = n;
                     b = m;
                  } else {
                     a = 0;
                     b = 0;
                  }
                  if (a !== 0) {
                     // point z is colinear with 2 of the triangle points, if it lies
                     // between them it's in the circle otherwise it's outside
                     if (this.fXN[a] !== this.fXN[b]) {
                        if (((this.fXN[z]-this.fXN[a])*(this.fXN[z]-this.fXN[b])) < 0) {
                           skip_this_triangle = true;
                           break;
                           // goto L90;
                        } else if (((this.fXN[z]-this.fXN[a])*(this.fXN[z]-this.fXN[b])) === 0) {
                           // At least two points are sitting on top of each other, we will
                           // treat these as one and not consider this a 'multiple points lying
                           // on a common circle' situation. It is a sign something could be
                           // wrong though, especially if the two coincident points have
                           // different fZ's. If they don't then this is harmless.
                           console.warn(`Interpolate Two of these three points are coincident ${a} ${b} ${z}`);
                        }
                     } else {
                        if (((this.fYN[z]-this.fYN[a])*(this.fYN[z]-this.fYN[b])) < 0) {
                           skip_this_triangle = true;
                           break;
                           // goto L90;
                        } else if (((this.fYN[z]-this.fYN[a])*(this.fYN[z]-this.fYN[b])) === 0) {
                           // At least two points are sitting on top of each other - see above.
                           console.warn(`Interpolate Two of these three points are coincident ${a} ${b} ${z}`);
                        }
                     }
                     // point is outside the circle, move to next point
                     continue; // goto L50;
                  }

                  if (skip_this_triangle) break; // deepscan-disable-line

   ///            Error("Interpolate", "Should not get to here");
                  // may as well soldier on
                  // SL: initialize before try to find better values
                  f = m;
                  o1 = p;
                  o2 = n;

                  // if point z were to look at the triangle, which point would it see
                  // lying between the other two? (we're going to form a quadrilateral
                  // from the points, and then demand certain properties of that
                  // quadrilateral)
                  dxz[0] = this.fXN[p]-this.fXN[z];
                  dyz[0] = this.fYN[p]-this.fYN[z];
                  dxz[1] = this.fXN[n]-this.fXN[z];
                  dyz[1] = this.fYN[n]-this.fYN[z];
                  dxz[2] = this.fXN[m]-this.fXN[z];
                  dyz[2] = this.fYN[m]-this.fYN[z];
                  for (l=1; l<=3; l++) {
                     dx1 = dxz[l-1];
                     dx2 = dxz[l%3];
                     dx3 = dxz[(l+1)%3];
                     dy1 = dyz[l-1];
                     dy2 = dyz[l%3];
                     dy3 = dyz[(l+1)%3];

                     // u et v are used only to know their sign. The previous
                     // code computed them with a division which was long and
                     // might be a division by 0. It is now replaced by a
                     // multiplication.
                     u = (dy3*dx2-dx3*dy2)*(dy1*dx2-dx1*dy2);
                     v = (dy3*dx1-dx3*dy1)*(dy2*dx1-dx2*dy1);

                     if ((u >= 0) && (v >= 0)) {
                        // vector (dx3,dy3) is expressible as a sum of the other two vectors
                        // with positive coefficients -> i.e. it lies between the other two vectors
                        if (l === 1) {
                           f = m; o1 = p; o2 = n; // deepscan-disable-line
                        } else if (l === 2) {
                           f = p; o1 = n; o2 = m;
                        } else {
                           f = n; o1 = m; o2 = p;
                        }
                        break; // goto L2;
                     }
                  }
   // L2:
                  // this is not a valid quadrilateral if the diagonals don't cross,
                  // check that points f and z lie on opposite side of the line o1-o2,
                  // this is true if the angle f-o1-z is greater than o2-o1-z and o2-o1-f
                  cfo1k = ((this.fXN[f]-this.fXN[o1])*(this.fXN[z]-this.fXN[o1])+(this.fYN[f]-this.fYN[o1])*(this.fYN[z]-this.fYN[o1]))/
                           Math.sqrt(((this.fXN[f]-this.fXN[o1])*(this.fXN[f]-this.fXN[o1])+(this.fYN[f]-this.fYN[o1])*(this.fYN[f]-this.fYN[o1]))*
                           ((this.fXN[z]-this.fXN[o1])*(this.fXN[z]-this.fXN[o1])+(this.fYN[z]-this.fYN[o1])*(this.fYN[z]-this.fYN[o1])));
                  co2o1k = ((this.fXN[o2]-this.fXN[o1])*(this.fXN[z]-this.fXN[o1])+(this.fYN[o2]-this.fYN[o1])*(this.fYN[z]-this.fYN[o1]))/
                           Math.sqrt(((this.fXN[o2]-this.fXN[o1])*(this.fXN[o2]-this.fXN[o1])+(this.fYN[o2]-this.fYN[o1])*(this.fYN[o2]-this.fYN[o1]))*
                           ((this.fXN[z]-this.fXN[o1])*(this.fXN[z]-this.fXN[o1]) + (this.fYN[z]-this.fYN[o1])*(this.fYN[z]-this.fYN[o1])));
                  co2o1f = ((this.fXN[o2]-this.fXN[o1])*(this.fXN[f]-this.fXN[o1])+(this.fYN[o2]-this.fYN[o1])*(this.fYN[f]-this.fYN[o1]))/
                           Math.sqrt(((this.fXN[o2]-this.fXN[o1])*(this.fXN[o2]-this.fXN[o1])+(this.fYN[o2]-this.fYN[o1])*(this.fYN[o2]-this.fYN[o1]))*
                           ((this.fXN[f]-this.fXN[o1])*(this.fXN[f]-this.fXN[o1]) + (this.fYN[f]-this.fYN[o1])*(this.fYN[f]-this.fYN[o1])));
                  if ((cfo1k > co2o1k) || (cfo1k > co2o1f)) {
                     // not a valid quadrilateral - point z is definitely outside the circle
                     continue; // goto L50;
                  }
                  // calculate the 2 internal angles of the quadrangle formed by joining
                  // points z and f to points o1 and o2, at z and f. If they sum to less
                  // than 180 degrees then z lies outside the circle
                  dko1 = Math.sqrt((this.fXN[z]-this.fXN[o1])*(this.fXN[z]-this.fXN[o1])+(this.fYN[z]-this.fYN[o1])*(this.fYN[z]-this.fYN[o1]));
                  dko2 = Math.sqrt((this.fXN[z]-this.fXN[o2])*(this.fXN[z]-this.fXN[o2])+(this.fYN[z]-this.fYN[o2])*(this.fYN[z]-this.fYN[o2]));
                  dfo1 = Math.sqrt((this.fXN[f]-this.fXN[o1])*(this.fXN[f]-this.fXN[o1])+(this.fYN[f]-this.fYN[o1])*(this.fYN[f]-this.fYN[o1]));
                  dfo2 = Math.sqrt((this.fXN[f]-this.fXN[o2])*(this.fXN[f]-this.fXN[o2])+(this.fYN[f]-this.fYN[o2])*(this.fYN[f]-this.fYN[o2]));
                  c1 = ((this.fXN[z]-this.fXN[o1])*(this.fXN[z]-this.fXN[o2])+(this.fYN[z]-this.fYN[o1])*(this.fYN[z]-this.fYN[o2]))/dko1/dko2;
                  c2 = ((this.fXN[f]-this.fXN[o1])*(this.fXN[f]-this.fXN[o2])+(this.fYN[f]-this.fYN[o1])*(this.fYN[f]-this.fYN[o2]))/dfo1/dfo2;
                  sin_sum = c1*Math.sqrt(1-c2*c2)+c2*Math.sqrt(1-c1*c1);

                  // sin_sum doesn't always come out as zero when it should do.
                  if (sin_sum < -1.e-6) {
                     // z is inside the circle, this is not a Delaunay triangle
                     skip_this_triangle = true;
                     break;
                     // goto L90;
                  } else if (Math.abs(sin_sum) <= 1.e-6) {
                     // point z lies on the circumference of the circle (within rounding errors)
                     // defined by the triangle, so there is potential for degeneracy in the
                     // triangle set (Delaunay triangulation does not give a unique way to split
                     // a polygon whose points lie on a circle into constituent triangles). Make
                     // a note of the additional point number.
                     ndegen++;
                     degen = z;
                     fdegen = f;
                     o1degen = o1;
                     o2degen = o2;
                  }

                  // L50: continue;
               } // end of for ( z = 1 ...) loop

               if (skip_this_triangle) continue;

               // This is a good triangle
               if (ndegen > 0) {
                  // but is degenerate with at least one other,
                  // haven't figured out what to do if more than 4 points are involved
   ///            if (ndegen > 1) {
   ///               Error("Interpolate",
   ///                     "More than 4 points lying on a circle. No decision making process formulated for triangulating this region in a non-arbitrary way %d %d %d %d",
   ///                     p,n,m,degen);
   ///               return thevalue;
   ///            }

                  // we have a quadrilateral which can be split down either diagonal
                  // (d<->f or o1<->o2) to form valid Delaunay triangles. Choose diagonal
                  // with highest average z-value. Whichever we choose we will have
                  // verified two triangles as good and two as bad, only note the good ones
                  d = degen;
                  f = fdegen;
                  o1 = o1degen;
                  o2 = o2degen;
                  if ((this.fZ[o1-1] + this.fZ[o2-1]) > (this.fZ[d-1] + this.fZ[f-1])) {
                     // best diagonalisation of quadrilateral is current one, we have
                     // the triangle
                     t1 = p;
                     t2 = n;
                     t3 = m;
                     // file the good triangles
                     this.FileIt(p, n, m);
                     this.FileIt(d, o1, o2);
                  } else {
                     // use other diagonal to split quadrilateral, use triangle formed by
                     // point f, the degnerate point d and whichever of o1 and o2 create
                     // an enclosing triangle
                     t1 = f;
                     t2 = d;
                     if (this.Enclose(f, d, o1, 0))
                        t3 = o1;
                      else
                        t3 = o2;

                     // file the good triangles
                     this.FileIt(f, d, o1);
                     this.FileIt(f, d, o2);
                  }
               } else {
                  // this is a Delaunay triangle, file it
                  this.FileIt(p, n, m);
                  t1 = p;
                  t2 = n;
                  t3 = m;
               }
               // do the interpolation
               thevalue = this.InterpolateOnPlane(t1, t2, t3, 0);
               return thevalue;

   // L90:      continue;
            }
         }
      }
      if (shouldbein) // deepscan-disable-line
         console.error(`Interpolate Point outside hull when expected inside: this point could be dodgy ${xx}  ${yy} ${ntris_tried}`);
      return thevalue;
   }

   /// Defines the number of triangles tested for a Delaunay triangle
   /// (number of iterations) before abandoning the search

   SetMaxIter(n = 100000) {
      this.fAllTri = false;
      this.fMaxIter = n;
   }

   /// Sets the histogram bin height for points lying outside the convex hull ie:
   /// the bins in the margin.

   SetMarginBinsContent(z) {
      this.fZout = z;
   }

}

   /** @summary Function handles tooltips in the mesh */
function graph2DTooltip(intersect) {
   let indx = Math.floor(intersect.index / this.nvertex);
   if ((indx < 0) || (indx >= this.index.length)) return null;
   const sqr = v => v*v;

   indx = this.index[indx];

   const fp = this.fp, gr = this.graph;
   let grx = fp.grx(gr.fX[indx]),
       gry = fp.gry(gr.fY[indx]),
       grz = fp.grz(gr.fZ[indx]);

   if (this.check_next && indx+1<gr.fX.length) {
      const d = intersect.point,
          grx1 = fp.grx(gr.fX[indx+1]),
          gry1 = fp.gry(gr.fY[indx+1]),
          grz1 = fp.grz(gr.fZ[indx+1]);
      if (sqr(d.x-grx1)+sqr(d.y-gry1)+sqr(d.z-grz1) < sqr(d.x-grx)+sqr(d.y-gry)+sqr(d.z-grz)) {
         grx = grx1; gry = gry1; grz = grz1; indx++;
      }
   }

   return {
      x1: grx - this.scale0,
      x2: grx + this.scale0,
      y1: gry - this.scale0,
      y2: gry + this.scale0,
      z1: grz - this.scale0,
      z2: grz + this.scale0,
      color: this.tip_color,
      lines: [this.tip_name,
               'pnt: ' + indx,
               'x: ' + fp.axisAsText('x', gr.fX[indx]),
               'y: ' + fp.axisAsText('y', gr.fY[indx]),
               'z: ' + fp.axisAsText('z', gr.fZ[indx])
             ]
   };
}



/**
 * @summary Painter for TGraph2D classes
 * @private
 */

class TGraph2DPainter extends ObjectPainter {

   /** @summary Decode options string  */
   decodeOptions(opt, _gr) {
      const d = new DrawOptions(opt);

      if (!this.options)
         this.options = {};

      const res = this.options, gr2d = this.getObject();

      if (d.check('FILL_', 'color') && gr2d)
         gr2d.fFillColor = d.color;

      if (d.check('LINE_', 'color') && gr2d)
         gr2d.fLineColor = d.color;

      d.check('SAME');
      if (d.check('TRI1'))
         res.Triangles = 11; // wireframe and colors
      else if (d.check('TRI2'))
         res.Triangles = 10; // only color triangles
      else if (d.check('TRIW'))
         res.Triangles = 1;
      else if (d.check('TRI'))
         res.Triangles = 2;
      else
         res.Triangles = 0;
      res.Line = d.check('LINE');
      res.Error = d.check('ERR') && (this.matchObjectType(clTGraph2DErrors) || this.matchObjectType(clTGraph2DAsymmErrors));

      if (d.check('P0COL'))
         res.Color = res.Circles = res.Markers = true;
       else {
         res.Color = d.check('COL');
         res.Circles = d.check('P0');
         res.Markers = d.check('P');
      }

      if (!res.Markers) res.Color = false;

      if (res.Color || res.Triangles >= 10)
         res.Zscale = d.check('Z');

      res.isAny = function() {
         return this.Markers || this.Error || this.Circles || this.Line || this.Triangles;
      };

      if (res.isAny()) {
         res.Axis = 'lego2';
         if (res.Zscale) res.Axis += 'z';
      } else
         res.Axis = opt;

      this.storeDrawOpt(opt);
   }

   /** @summary Create histogram for axes drawing */
   createHistogram() {
      const gr = this.getObject(),
            asymm = this.matchObjectType(clTGraph2DAsymmErrors);
      let xmin = gr.fX[0], xmax = xmin,
          ymin = gr.fY[0], ymax = ymin,
          zmin = gr.fZ[0], zmax = zmin;

      for (let p = 0; p < gr.fNpoints; ++p) {
         const x = gr.fX[p], y = gr.fY[p], z = gr.fZ[p];

         if (this.options.Error) {
            xmin = Math.min(xmin, x - (asymm ? gr.fEXlow[p] : gr.fEX[p]));
            xmax = Math.max(xmax, x + (asymm ? gr.fEXhigh[p] : gr.fEX[p]));
            ymin = Math.min(ymin, y - (asymm ? gr.fEYlow[p] : gr.fEY[p]));
            ymax = Math.max(ymax, y + (asymm ? gr.fEYhigh[p] : gr.fEY[p]));
            zmin = Math.min(zmin, z - (asymm ? gr.fEZlow[p] : gr.fEZ[p]));
            zmax = Math.max(zmax, z + (asymm ? gr.fEZhigh[p] : gr.fEZ[p]));
         } else {
            xmin = Math.min(xmin, x);
            xmax = Math.max(xmax, x);
            ymin = Math.min(ymin, y);
            ymax = Math.max(ymax, y);
            zmin = Math.min(zmin, z);
            zmax = Math.max(zmax, z);
         }
      }

      function calc_delta(min, max, margin) {
         if (min < max) return margin * (max - min);
         return Math.abs(min) < 1e5 ? 0.02 : 0.02 * Math.abs(min);
      }
      const dx = calc_delta(xmin, xmax, gr.fMargin),
            dy = calc_delta(ymin, ymax, gr.fMargin),
            dz = calc_delta(zmin, zmax, 0);
      let uxmin = xmin - dx, uxmax = xmax + dx,
          uymin = ymin - dy, uymax = ymax + dy,
          uzmin = zmin - dz, uzmax = zmax + dz;

      if ((uxmin < 0) && (xmin >= 0)) uxmin = xmin*0.98;
      if ((uxmax > 0) && (xmax <= 0)) uxmax = 0;

      if ((uymin < 0) && (ymin >= 0)) uymin = ymin*0.98;
      if ((uymax > 0) && (ymax <= 0)) uymax = 0;

      if ((uzmin < 0) && (zmin >= 0)) uzmin = zmin*0.98;
      if ((uzmax > 0) && (zmax <= 0)) uzmax = 0;

      const graph = this.getObject();

      if (graph.fMinimum !== kNoZoom) uzmin = graph.fMinimum;
      if (graph.fMaximum !== kNoZoom) uzmax = graph.fMaximum;

      this._own_histogram = true; // when histogram created on client side

      const histo = createHistogram(clTH2F, graph.fNpx, graph.fNpy);
      histo.fName = graph.fName + '_h';
      setHistogramTitle(histo, graph.fTitle);
      histo.fXaxis.fXmin = uxmin;
      histo.fXaxis.fXmax = uxmax;
      histo.fYaxis.fXmin = uymin;
      histo.fYaxis.fXmax = uymax;
      histo.fZaxis.fXmin = uzmin;
      histo.fZaxis.fXmax = uzmax;
      histo.fMinimum = uzmin;
      histo.fMaximum = uzmax;
      histo.fBits |= kNoStats;

      if (!this.options.isAny()) {
         const dulaunay = this.buildDelaunay(graph);
         if (dulaunay) {
            for (let i = 0; i < graph.fNpx; ++i) {
               const xx = uxmin + (i + 0.5) / graph.fNpx * (uxmax - uxmin);
               for (let j = 0; j < graph.fNpy; ++j) {
                  const yy = uymin + (j + 0.5) / graph.fNpy * (uymax - uymin),
                        zz = dulaunay.ComputeZ(xx, yy);
                  histo.fArray[histo.getBin(i+1, j+1)] = zz;
               }
            }
         }
      }

      return histo;
   }

   buildDelaunay(graph) {
      if (!this._delaunay) {
         this._delaunay = new TGraphDelaunay(graph);
         this._delaunay.FindAllTriangles();
         if (!this._delaunay.fNdt)
            delete this._delaunay;
      }
      return this._delaunay;
   }

   drawTriangles(fp, graph, levels, palette) {
      const dulaunay = this.buildDelaunay(graph);
      if (!dulaunay) return;

      const main_grz = !fp.logz ? fp.grz : value => (value < fp.scale_zmin) ? -0.1 : fp.grz(value),
            plain_mode = this.options.Triangles === 2,
            do_faces = (this.options.Triangles >= 10) || plain_mode,
            do_lines = (this.options.Triangles % 10 === 1) || (plain_mode && (graph.fLineColor !== graph.fFillColor)),
            triangles = new Triangles3DHandler(levels, main_grz, 0, 2*fp.size_z3d, do_lines);

      for (triangles.loop = 0; triangles.loop < 2; ++triangles.loop) {
         triangles.createBuffers();

         for (let t = 0; t < dulaunay.fNdt; ++t) {
            const points = [dulaunay.fPTried[t], dulaunay.fNTried[t], dulaunay.fMTried[t]],
                  coord = [];
            let use_triangle = true;
            for (let i = 0; i < 3; ++i) {
               const pnt = points[i] - 1;
               coord.push(fp.grx(graph.fX[pnt]), fp.gry(graph.fY[pnt]), main_grz(graph.fZ[pnt]));

                if ((graph.fX[pnt] < fp.scale_xmin) || (graph.fX[pnt] > fp.scale_xmax) ||
                    (graph.fY[pnt] < fp.scale_ymin) || (graph.fY[pnt] > fp.scale_ymax))
                  use_triangle = false;
            }

            if (do_faces && use_triangle)
               triangles.addMainTriangle(...coord);

            if (do_lines && use_triangle) {
               triangles.addLineSegment(coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]);

               triangles.addLineSegment(coord[3], coord[4], coord[5], coord[6], coord[7], coord[8]);

               triangles.addLineSegment(coord[6], coord[7], coord[8], coord[0], coord[1], coord[2]);
            }
         }
      }

      triangles.callFuncs((lvl, pos) => {
         const geometry = createLegoGeom(this.getMainPainter(), pos, null, 100, 100),
               color = plain_mode ? this.getColor(graph.fFillColor) : palette.calcColor(lvl, levels.length),
               material = new MeshBasicMaterial(getMaterialArgs(color, { side: DoubleSide, vertexColors: false })),

          mesh = new Mesh(geometry, material);

         fp.add3DMesh(mesh, this);

         mesh.painter = this; // to let use it with context menu
      }, (_isgrid, lpos) => {
         const lcolor = this.getColor(graph.fLineColor),
              material = new LineBasicMaterial({ color: new Color(lcolor), linewidth: graph.fLineWidth }),
              linemesh = createLineSegments(convertLegoBuf(this.getMainPainter(), lpos, 100, 100), material);
         fp.add3DMesh(linemesh, this);
      });
   }

   /** @summary Update TGraph2D object */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj)) return false;

      if (opt && (opt !== this.options.original))
         this.decodeOptions(opt, obj);

      Object.assign(this.getObject(), obj);

      delete this._delaunay; // rebuild triangles

      delete this.$redraw_hist;

      // if our own histogram was used as axis drawing, we need update histogram as well
      if (this.axes_draw) {
         const hist_painter = this.getMainPainter();
         hist_painter?.updateObject(this.createHistogram(), this.options.Axis);
         this.$redraw_hist = hist_painter;
      }

      return true;
   }

   /** @summary Redraw TGraph2D object
     * @desc Update histogram drawing if necessary
     * @return {Promise} for drawing ready */
   async redraw() {
      let promise = Promise.resolve(true);

      if (this.$redraw_hist) {
         promise = this.$redraw_hist.redraw();
         delete this.$redraw_hist;
      }

      return promise.then(() => this.drawGraph2D());
   }

   /** @summary Actual drawing of TGraph2D object
     * @return {Promise} for drawing ready */
   async drawGraph2D() {
      const main = this.getMainPainter(),
            fp = this.getFramePainter(),
            graph = this.getObject();

      if (!graph || !main || !fp || !fp.mode3d)
         return this;

      fp.remove3DMeshes(this);

      if (!this.options.isAny()) {
         // no need to draw somthing if histogram content was drawn
         if (main.draw_content)
            return this;
         if ((graph.fMarkerSize === 1) && (graph.fMarkerStyle === 1))
            this.options.Circles = true;
         else
            this.options.Markers = true;
      }

      const countSelected = (zmin, zmax) => {
         let cnt = 0;
         for (let i = 0; i < graph.fNpoints; ++i) {
            if ((graph.fX[i] < fp.scale_xmin) || (graph.fX[i] > fp.scale_xmax) ||
                (graph.fY[i] < fp.scale_ymin) || (graph.fY[i] > fp.scale_ymax) ||
                (graph.fZ[i] < zmin) || (graph.fZ[i] >= zmax)) continue;

            ++cnt;
         }
         return cnt;
      };

      // try to define scale-down factor
      let step = 1;
      if ((settings.OptimizeDraw > 0) && !fp.webgl) {
         const numselected = countSelected(fp.scale_zmin, fp.scale_zmax),
             sizelimit = 50000;

         if (numselected > sizelimit) {
            step = Math.floor(numselected / sizelimit);
            if (step <= 2) step = 2;
         }
      }

      const markeratt = this.createAttMarker({ attr: graph, std: false }),
            promises = [];
      let palette = null,
          levels = [fp.scale_zmin, fp.scale_zmax],
          scale = fp.size_x3d / 100 * markeratt.getFullSize();

      if (this.options.Circles)
         scale = 0.06 * fp.size_x3d;

      if (fp.usesvg) scale *= 0.3;

      scale *= 7 * Math.max(fp.size_x3d / fp.getFrameWidth(), fp.size_z3d / fp.getFrameHeight());

      if (this.options.Color || (this.options.Triangles >= 10)) {
         levels = main.getContourLevels(true);
         palette = main.getHistPalette();
      }

      if (this.options.Triangles)
         this.drawTriangles(fp, graph, levels, palette);

      for (let lvl = 0; lvl < levels.length-1; ++lvl) {
         const lvl_zmin = Math.max(levels[lvl], fp.scale_zmin),
               lvl_zmax = Math.min(levels[lvl+1], fp.scale_zmax);

         if (lvl_zmin >= lvl_zmax) continue;

         const size = Math.floor(countSelected(lvl_zmin, lvl_zmax) / step),
               index = new Int32Array(size);
         let pnts = null, select = 0, icnt = 0,
             err = null, asymm = false, line = null, ierr = 0, iline = 0;

         if (this.options.Markers || this.options.Circles)
            pnts = new PointsCreator(size, fp.webgl, scale/3);

         if (this.options.Error) {
            err = new Float32Array(size*6*3);
            asymm = this.matchObjectType(clTGraph2DAsymmErrors);
          }

         if (this.options.Line)
            line = new Float32Array((size-1)*6);

         for (let i = 0; i < graph.fNpoints; ++i) {
            if ((graph.fX[i] < fp.scale_xmin) || (graph.fX[i] > fp.scale_xmax) ||
                (graph.fY[i] < fp.scale_ymin) || (graph.fY[i] > fp.scale_ymax) ||
                (graph.fZ[i] < lvl_zmin) || (graph.fZ[i] >= lvl_zmax)) continue;

            if (step > 1) {
               select = (select+1) % step;
               if (select !== 0) continue;
            }

            index[icnt++] = i; // remember point index for tooltip

            const x = fp.grx(graph.fX[i]),
                y = fp.gry(graph.fY[i]),
                z = fp.grz(graph.fZ[i]);

            if (pnts) pnts.addPoint(x, y, z);

            if (err) {
               err[ierr] = fp.grx(graph.fX[i] - (asymm ? graph.fEXlow[i] : graph.fEX[i]));
               err[ierr+1] = y;
               err[ierr+2] = z;
               err[ierr+3] = fp.grx(graph.fX[i] + (asymm ? graph.fEXhigh[i] : graph.fEX[i]));
               err[ierr+4] = y;
               err[ierr+5] = z;
               ierr+=6;
               err[ierr] = x;
               err[ierr+1] = fp.gry(graph.fY[i] - (asymm ? graph.fEYlow[i] : graph.fEY[i]));
               err[ierr+2] = z;
               err[ierr+3] = x;
               err[ierr+4] = fp.gry(graph.fY[i] + (asymm ? graph.fEYhigh[i] : graph.fEY[i]));
               err[ierr+5] = z;
               ierr+=6;
               err[ierr] = x;
               err[ierr+1] = y;
               err[ierr+2] = fp.grz(graph.fZ[i] - (asymm ? graph.fEZlow[i] : graph.fEZ[i]));
               err[ierr+3] = x;
               err[ierr+4] = y;
               err[ierr+5] = fp.grz(graph.fZ[i] + (asymm ? graph.fEZhigh[i] : graph.fEZ[i]));
               ierr+=6;
            }

            if (line) {
               if (iline>=6) {
                  line[iline] = line[iline-3];
                  line[iline+1] = line[iline-2];
                  line[iline+2] = line[iline-1];
                  iline+=3;
               }
               line[iline] = x;
               line[iline+1] = y;
               line[iline+2] = z;
               iline+=3;
            }
         }

         if (line && (iline > 3) && (line.length === iline)) {
            const lcolor = this.getColor(graph.fLineColor),
                  material = new LineBasicMaterial({ color: new Color(lcolor), linewidth: graph.fLineWidth }),
                  linemesh = createLineSegments(line, material);
            fp.add3DMesh(linemesh, this);

            linemesh.graph = graph;
            linemesh.index = index;
            linemesh.fp = fp;
            linemesh.scale0 = 0.7*scale;
            linemesh.tip_name = this.getObjectHint();
            linemesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
            linemesh.nvertex = 2;
            linemesh.check_next = true;

            linemesh.tooltip = graph2DTooltip;
         }

         if (err) {
            const lcolor = this.getColor(graph.fLineColor),
                  material = new LineBasicMaterial({ color: new Color(lcolor), linewidth: graph.fLineWidth }),
                  errmesh = createLineSegments(err, material);
            fp.add3DMesh(errmesh, this);

            errmesh.graph = graph;
            errmesh.index = index;
            errmesh.fp = fp;
            errmesh.scale0 = 0.7*scale;
            errmesh.tip_name = this.getObjectHint();
            errmesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
            errmesh.nvertex = 6;

            errmesh.tooltip = graph2DTooltip;
         }

         if (pnts) {
            let color = 'blue';

            if (!this.options.Circles || this.options.Color)
               color = palette?.calcColor(lvl, levels.length) ?? this.getColor(graph.fMarkerColor);

            const pr = pnts.createPoints({ color, style: this.options.Circles ? 4 : graph.fMarkerStyle }).then(mesh => {
               mesh.graph = graph;
               mesh.fp = fp;
               mesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
               mesh.scale0 = 0.3*scale;
               mesh.index = index;

               mesh.tip_name = this.getObjectHint();
               mesh.tooltip = graph2DTooltip;
               fp.add3DMesh(mesh, this);
            });

            promises.push(pr);
         }
      }

      return Promise.all(promises).then(() => {
         if (this.options.Zscale && this.axes_draw) {
            const pal = this.getMainPainter()?.findFunction(clTPaletteAxis),
                  pal_painter = this.getPadPainter()?.findPainterFor(pal);
            return pal_painter?.drawPave();
         }
      }).then(() => {
         fp.render3D(100);
         return this;
      });
   }

   /** @summary draw TGraph2D object */
   static async draw(dom, gr, opt) {
      const painter = new TGraph2DPainter(dom, gr);
      painter.decodeOptions(opt, gr);

      let promise = Promise.resolve(null);

      if (!painter.getMainPainter()) {
         // histogram is not preserved in TGraph2D
         promise = TH2Painter.draw(dom, painter.createHistogram(), painter.options.Axis);
         painter.axes_draw = true;
      }

      return promise.then(() => {
         painter.addToPadPrimitives();
         return painter.drawGraph2D();
      });
   }

} // class TGraph2DPainter

export { TGraph2DPainter };
