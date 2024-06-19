import { gStyle, createHistogram, createTPolyLine, isFunc, isStr,
         clTMultiGraph, clTH1D, clTF2, clTProfile2D, kInspect } from '../core.mjs';
import { rgb as d3_rgb, chord as d3_chord, arc as d3_arc, ribbon as d3_ribbon } from '../d3.mjs';
import { kBlack } from '../base/colors.mjs';
import { TRandom, floatToString, makeTranslate, addHighlightStyle } from '../base/BasePainter.mjs';
import { EAxisBits } from '../base/ObjectPainter.mjs';
import { THistPainter } from './THistPainter.mjs';


/** @summary Build histogram contour lines
  * @private */
function buildHist2dContour(histo, handle, levels, palette, contour_func) {
   const kMAXCONTOUR = 2004,
         kMAXCOUNT = 2000,
         // arguments used in the PaintContourLine
         xarr = new Float32Array(2*kMAXCONTOUR),
         yarr = new Float32Array(2*kMAXCONTOUR),
         itarr = new Int32Array(2*kMAXCONTOUR),
         nlevels = levels.length,
         first_level = levels[0], last_level = levels[nlevels - 1],
         polys = [],
         x = [0, 0, 0, 0], y = [0, 0, 0, 0], zc = [0, 0, 0, 0], ir = [0, 0, 0, 0],
         arrx = handle.grx,
         arry = handle.gry;

   let lj = 0, ipoly, poly, np, npmax = 0,
       i, j, k, n, m, ljfill, count,
       xsave, ysave, itars, ix, jx;

   const LinearSearch = zc => {
      if (zc >= last_level)
         return nlevels-1;

      for (let kk = 0; kk < nlevels; ++kk) {
         if (zc < levels[kk])
            return kk-1;
       }
      return nlevels-1;
   }, BinarySearch = zc => {
      if (zc < first_level)
         return -1;
      if (zc >= last_level)
         return nlevels - 1;

      let l = 0, r = nlevels - 1, m;
      while (r - l > 1) {
        m = Math.round((r + l) / 2);
        if (zc < levels[m])
           r = m;
        else
           l = m;
      }
      return l;
   },
   LevelSearch = nlevels < 10 ? LinearSearch : BinarySearch,
   PaintContourLine = (elev1, icont1, x1, y1, elev2, icont2, x2, y2) => {
      /* Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels */
      const vert = (x1 === x2),
            tlen = vert ? (y2 - y1) : (x2 - x1),
            tdif = elev2 - elev1;
      let n = icont1 + 1, ii = lj-1, icount = 0,
          xlen, pdif, diff, elev;
      const maxii = ii + kMAXCONTOUR/2 -3;

      while (n <= icont2 && ii <= maxii) {
         // elev = fH->GetContourLevel(n);
         elev = levels[n];
         diff = elev - elev1;
         pdif = diff/tdif;
         xlen = tlen*pdif;
         if (vert) {
            xarr[ii] = x1;
            yarr[ii] = y1 + xlen;
         } else {
            xarr[ii] = x1 + xlen;
            yarr[ii] = y1;
         }
         itarr[ii] = n;
         icount++;
         ii += 2;
         n++;
      }
      return icount;
   };

   for (j = handle.j1; j < handle.j2-1; ++j) {
      y[1] = y[0] = (arry[j] + arry[j+1])/2;
      y[3] = y[2] = (arry[j+1] + arry[j+2])/2;

      for (i = handle.i1; i < handle.i2-1; ++i) {
         zc[0] = histo.getBinContent(i+1, j+1);
         zc[1] = histo.getBinContent(i+2, j+1);
         zc[2] = histo.getBinContent(i+2, j+2);
         zc[3] = histo.getBinContent(i+1, j+2);

         for (k = 0; k < 4; k++)
            ir[k] = LevelSearch(zc[k]);

         if ((ir[0] !== ir[1]) || (ir[1] !== ir[2]) || (ir[2] !== ir[3]) || (ir[3] !== ir[0])) { // deepscan-disable-line
            x[3] = x[0] = (arrx[i] + arrx[i+1])/2;
            x[2] = x[1] = (arrx[i+1] + arrx[i+2])/2;

            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=1;
            for (ix=1; ix<=4; ix++) {
               m = n%4 + 1;
               ljfill = PaintContourLine(zc[n-1], ir[n-1], x[n-1], y[n-1], zc[m-1], ir[m-1], x[m-1], y[m-1]);
               lj += 2*ljfill;
               n = m;
            }

            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=2;
            for (ix=1; ix<=4; ix++) {
               m = (n === 1) ? 4 : n-1;
               ljfill = PaintContourLine(zc[n-1], ir[n-1], x[n-1], y[n-1], zc[m-1], ir[m-1], x[m-1], y[m-1]);
               lj += 2*ljfill;
               n = m;
            }
            //     Re-order endpoints

            count = 0;
            for (ix = 1; ix <= lj - 5; ix += 2) {
               // count = 0;
               while (itarr[ix-1] !== itarr[ix]) {
                  xsave = xarr[ix];
                  ysave = yarr[ix];
                  itars = itarr[ix];
                  for (jx=ix; jx<=lj-5; jx +=2) {
                     xarr[jx] = xarr[jx+2];
                     yarr[jx] = yarr[jx+2];
                     itarr[jx] = itarr[jx+2];
                  }
                  xarr[lj-3] = xsave;
                  yarr[lj-3] = ysave;
                  itarr[lj-3] = itars;
                  if (count > kMAXCOUNT) break;
                  count++;
               }
            }

            if (count > 100) continue;

            for (ix = 1; ix <= lj - 2; ix += 2) {
               ipoly = itarr[ix-1];

               if ((ipoly >= 0) && (ipoly < levels.length)) {
                  poly = polys[ipoly];
                  if (!poly)
                     poly = polys[ipoly] = createTPolyLine(kMAXCONTOUR*4, true);

                  np = poly.fLastPoint;
                  if (np < poly.fN-2) {
                     poly.fX[np+1] = Math.round(xarr[ix-1]);
                     poly.fY[np+1] = Math.round(yarr[ix-1]);
                     poly.fX[np+2] = Math.round(xarr[ix]);
                     poly.fY[np+2] = Math.round(yarr[ix]);
                     poly.fLastPoint = np+2;
                     npmax = Math.max(npmax, poly.fLastPoint+1);
                  } else {
                     // console.log(`reject point ${poly.fLastPoint}`);
                  }
               }
            }
         } // end of if (ir[0]
      } // end of j
   } // end of i

   const polysort = new Int32Array(levels.length);
   let first = 0;
   // find first positive contour
   for (ipoly = 0; ipoly < levels.length; ipoly++)
      if (levels[ipoly] >= 0) { first = ipoly; break; }

   // store negative contours from 0 to minimum, then all positive contours
   k = 0;
   for (ipoly = first-1; ipoly >= 0; ipoly--) { polysort[k] = ipoly; k++; }
   for (ipoly = first; ipoly < levels.length; ipoly++) { polysort[k] = ipoly; k++; }

   const xp = new Float32Array(2*npmax),
         yp = new Float32Array(2*npmax),
         has_func = isFunc(palette.calcColorIndex); // rcanvas for v7

   for (k = 0; k < levels.length; ++k) {
      ipoly = polysort[k];
      poly = polys[ipoly];
      if (!poly) continue;

      const colindx = has_func ? palette.calcColorIndex(ipoly, levels.length) : ipoly,
            xx = poly.fX, yy = poly.fY, np = poly.fLastPoint+1,
            xmin = 0, ymin = 0;
      let istart = 0, iminus, iplus, nadd;

      while (true) {
         iminus = npmax;
         iplus = iminus+1;
         xp[iminus]= xx[istart]; yp[iminus] = yy[istart];
         xp[iplus] = xx[istart+1]; yp[iplus] = yy[istart+1];
         xx[istart] = xx[istart+1] = xmin;
         yy[istart] = yy[istart+1] = ymin;
         while (true) {
            nadd = 0;
            for (i = 2; i < np; i += 2) {
               if ((iplus < 2*npmax-1) && (xx[i] === xp[iplus]) && (yy[i] === yp[iplus])) {
                  iplus++;
                  xp[iplus] = xx[i+1]; yp[iplus] = yy[i+1];
                  xx[i] = xx[i+1] = xmin;
                  yy[i] = yy[i+1] = ymin;
                  nadd++;
               }
               if ((iminus > 0) && (xx[i+1] === xp[iminus]) && (yy[i+1] === yp[iminus])) {
                  iminus--;
                  xp[iminus] = xx[i]; yp[iminus] = yy[i];
                  xx[i] = xx[i+1] = xmin;
                  yy[i] = yy[i+1] = ymin;
                  nadd++;
               }
            }
            if (nadd === 0) break;
         }

         if ((iminus+1 < iplus) && (iminus >= 0))
            contour_func(colindx, xp, yp, iminus, iplus, ipoly);

         istart = 0;
         for (i = 2; i < np; i += 2) {
            if (xx[i] !== xmin && yy[i] !== ymin) {
               istart = i;
               break;
            }
         }

         if (istart === 0) break;
      }
   }
}

/** @summary Handle 3D triangles with color levels */

class Triangles3DHandler {

   constructor(ilevels, grz, grz_min, grz_max, dolines, donormals, dogrid) {
      let levels = [grz_min, grz_max]; // just cut top/bottom parts

      if (ilevels) {
         // recalculate levels into graphical coordinates
         levels = new Float32Array(ilevels.length);
         for (let ll = 0; ll < ilevels.length; ++ll)
            levels[ll] = grz(ilevels[ll]);
      }

      Object.assign(this, { grz_min, grz_max, dolines, donormals, dogrid });

      this.loop = 0;

      const nfaces = [], posbuf = [], posbufindx = [],    // buffers for faces
            pntbuf = new Float32Array(6*3), // maximal 6 points
            gridpnts = new Float32Array(2*3),
            levels_eps = (levels[levels.length-1] - levels[0]) / levels.length / 1e2;
      let nsegments = 0, lpos = null, lindx = 0,  // buffer for lines
          ngridsegments = 0, grid = null, gindx = 0, // buffer for grid lines segments
          normindx = [],                             // buffer to remember place of vertex for each bin
          pntindx = 0, lastpart = 0, gridcnt = 0;

      function checkSide(z, level1, level2, eps) {
         return (z < level1 - eps) ? -1 : (z > level2 + eps ? 1 : 0);
      }

      this.createNormIndex = function(handle) {
          // for each bin maximal 8 points reserved
         if (handle.donormals)
            normindx = new Int32Array((handle.i2-handle.i1)*(handle.j2-handle.j1)*8).fill(-1);
      };

      this.createBuffers = function() {
         if (!this.loop) return;

         for (let lvl = 1; lvl < levels.length; ++lvl) {
            if (nfaces[lvl]) {
               posbuf[lvl] = new Float32Array(nfaces[lvl] * 9);
               posbufindx[lvl] = 0;
            }
         }
         if (this.dolines && (nsegments > 0))
            lpos = new Float32Array(nsegments * 6);
         if (this.dogrid && (ngridsegments > 0))
            grid = new Float32Array(ngridsegments * 6);
      };

      this.addLineSegment = function(x1, y1, z1, x2, y2, z2) {
         if (!this.dolines) return;
         const side1 = checkSide(z1, this.grz_min, this.grz_max, 0),
             side2 = checkSide(z2, this.grz_min, this.grz_max, 0);
         if ((side1 === side2) && (side1 !== 0))
            return;
         if (!this.loop)
            return ++nsegments;

         if (side1 !== 0) {
            const diff = z2 - z1;
            z1 = (side1 < 0) ? this.grz_min : this.grz_max;
            x1 = x2 - (x2 - x1) / diff * (z2 - z1);
            y1 = y2 - (y2 - y1) / diff * (z2 - z1);
         }
         if (side2 !== 0) {
            const diff = z1 - z2;
            z2 = (side2 < 0) ? this.grz_min : this.grz_max;
            x2 = x1 - (x1 - x2) / diff * (z1 - z2);
            y2 = y1 - (y1 - y2) / diff * (z1 - z2);
         }

         lpos[lindx] = x1; lpos[lindx+1] = y1; lpos[lindx+2] = z1; lindx+=3;
         lpos[lindx] = x2; lpos[lindx+1] = y2; lpos[lindx+2] = z2; lindx+=3;
      };

      function addCrossingPoint(xx1, yy1, zz1, xx2, yy2, zz2, crossz, with_grid) {
         if (pntindx >= pntbuf.length)
            console.log('more than 6 points???');

         const part = (crossz - zz1) / (zz2 - zz1);
         let shift = 3;
         if ((lastpart !== 0) && (Math.abs(part) < Math.abs(lastpart))) {
            // while second crossing point closer than first to original, move it in memory
            pntbuf[pntindx] = pntbuf[pntindx-3];
            pntbuf[pntindx+1] = pntbuf[pntindx-2];
            pntbuf[pntindx+2] = pntbuf[pntindx-1];
            pntindx-=3; shift = 6;
         }

         pntbuf[pntindx] = xx1 + part*(xx2-xx1);
         pntbuf[pntindx+1] = yy1 + part*(yy2-yy1);
         pntbuf[pntindx+2] = crossz;

         if (with_grid && grid) {
            gridpnts[gridcnt] = pntbuf[pntindx];
            gridpnts[gridcnt+1] = pntbuf[pntindx+1];
            gridpnts[gridcnt+2] = pntbuf[pntindx+2];
            gridcnt += 3;
         }

         pntindx += shift;
         lastpart = part;
      }

      function rememberVertex(indx, handle, ii, jj) {
         const bin = ((ii-handle.i1) * (handle.j2-handle.j1) + (jj-handle.j1))*8;

         if (normindx[bin] >= 0)
            return console.error('More than 8 vertexes for the bin');

         const pos = bin + 8 + normindx[bin]; // position where write index
         normindx[bin]--;
         normindx[pos] = indx; // at this moment index can be overwritten, means all 8 position are there
      }

      this.addMainTriangle = function(x1, y1, z1, x2, y2, z2, x3, y3, z3, is_first, handle, i, j) {
         for (let lvl = 1; lvl < levels.length; ++lvl) {
            let side1 = checkSide(z1, levels[lvl-1], levels[lvl], levels_eps),
                side2 = checkSide(z2, levels[lvl-1], levels[lvl], levels_eps),
                side3 = checkSide(z3, levels[lvl-1], levels[lvl], levels_eps),
                side_sum = side1 + side2 + side3;

            // always show top segments
            if ((lvl > 1) && (lvl === levels.length - 1) && (side_sum === 3) && (z1 <= this.grz_max))
               side1 = side2 = side3 = side_sum = 0;


            if (side_sum === 3) continue;
            if (side_sum === -3) return;

            if (!this.loop) {
               let npnts = Math.abs(side2-side1) + Math.abs(side3-side2) + Math.abs(side1-side3);
               if (side1 === 0) ++npnts;
               if (side2 === 0) ++npnts;
               if (side3 === 0) ++npnts;

               if ((npnts === 1) || (npnts === 2)) console.error(`FOUND npnts = ${npnts}`);

               if (npnts > 2) {
                  if (nfaces[lvl] === undefined)
                     nfaces[lvl] = 0;
                  nfaces[lvl] += npnts-2;
               }

               // check if any(contours for given level exists
               if (((side1 > 0) || (side2 > 0) || (side3 > 0)) &&
                   ((side1 !== side2) || (side2 !== side3) || (side3 !== side1))) // deepscan-disable-line
                      ++ngridsegments;

               continue;
            }

            gridcnt = 0;

            pntindx = 0;
            if (side1 === 0) { pntbuf[pntindx] = x1; pntbuf[pntindx+1] = y1; pntbuf[pntindx+2] = z1; pntindx += 3; }

            if (side1 !== side2) {
               // order is important, should move from 1->2 point, checked via lastpart
               lastpart = 0;
               if ((side1 < 0) || (side2 < 0)) addCrossingPoint(x1, y1, z1, x2, y2, z2, levels[lvl-1]);
               if ((side1 > 0) || (side2 > 0)) addCrossingPoint(x1, y1, z1, x2, y2, z2, levels[lvl], true);
            }

            if (side2 === 0) { pntbuf[pntindx] = x2; pntbuf[pntindx+1] = y2; pntbuf[pntindx+2] = z2; pntindx += 3; }

            if (side2 !== side3) {
               // order is important, should move from 2->3 point, checked via lastpart
               lastpart = 0;
               if ((side2 < 0) || (side3 < 0)) addCrossingPoint(x2, y2, z2, x3, y3, z3, levels[lvl-1]);
               if ((side2 > 0) || (side3 > 0)) addCrossingPoint(x2, y2, z2, x3, y3, z3, levels[lvl], true);
            }

            if (side3 === 0) { pntbuf[pntindx] = x3; pntbuf[pntindx+1] = y3; pntbuf[pntindx+2] = z3; pntindx += 3; }

            if (side3 !== side1) {
               // order is important, should move from 3->1 point, checked via lastpart
               lastpart = 0;
               if ((side3 < 0) || (side1 < 0)) addCrossingPoint(x3, y3, z3, x1, y1, z1, levels[lvl-1]);
               if ((side3 > 0) || (side1 > 0)) addCrossingPoint(x3, y3, z3, x1, y1, z1, levels[lvl], true);
            }

            if (pntindx === 0) continue;
            if (pntindx < 9) { console.log(`found ${pntindx/3} points, must be at least 3`); continue; }

            if (grid && (gridcnt === 6)) {
               for (let jj = 0; jj < 6; ++jj)
                  grid[gindx+jj] = gridpnts[jj];
               gindx += 6;
            }

            // if three points and surf === 14, remember vertex for each point

            const buf = posbuf[lvl];
            let s = posbufindx[lvl];
            if (this.donormals && (pntindx === 9)) {
               rememberVertex(s, handle, i, j);
               rememberVertex(s+3, handle, i+1, is_first ? j+1 : j);
               rememberVertex(s+6, handle, is_first ? i : i+1, j+1);
            }

            for (let k1 = 3; k1 < pntindx - 3; k1 += 3) {
               buf[s] = pntbuf[0]; buf[s+1] = pntbuf[1]; buf[s+2] = pntbuf[2]; s+=3;
               buf[s] = pntbuf[k1]; buf[s+1] = pntbuf[k1+1]; buf[s+2] = pntbuf[k1+2]; s+=3;
               buf[s] = pntbuf[k1+3]; buf[s+1] = pntbuf[k1+4]; buf[s+2] = pntbuf[k1+5]; s+=3;
            }
            posbufindx[lvl] = s;
         }
      };

      this.callFuncs = function(meshFunc, linesFunc) {
         for (let lvl = 1; lvl < levels.length; ++lvl) {
            if (posbuf[lvl] && meshFunc)
               meshFunc(lvl, posbuf[lvl], normindx);
         }

         if (lpos && linesFunc) {
            if (nsegments*6 !== lindx)
               console.error(`SURF lines mismmatch nsegm=${nsegments} lindx=${lindx} diff=${nsegments*6 - lindx}`);
            linesFunc(false, lpos);
         }

         if (grid && linesFunc) {
            if (ngridsegments*6 !== gindx)
               console.error(`SURF grid draw mismatch ngridsegm=${ngridsegments} gindx=${gindx} diff=${ngridsegments*6 - gindx}`);
            linesFunc(true, grid);
         }
      };
    }

}


/** @summary Build 3d surface
  * @desc Make it indepependent from three.js to be able reuse it for 2d case
  * @private */
function buildSurf3D(histo, handle, ilevels, meshFunc, linesFunc) {
   const main_grz = handle.grz,
        arrx = handle.original ? handle.origx : handle.grx,
        arry = handle.original ? handle.origy : handle.gry,
        triangles = new Triangles3DHandler(ilevels, handle.grz, handle.grz_min, handle.grz_max, handle.dolines, handle.donormals, handle.dogrid);
   let i, j, x1, x2, y1, y2, z11, z12, z21, z22;

   triangles.createNormIndex(handle);

   for (triangles.loop = 0; triangles.loop < 2; ++triangles.loop) {
      triangles.createBuffers();

      for (i = handle.i1; i < handle.i2-1; ++i) {
         x1 = handle.original ? 0.5 * (arrx[i] + arrx[i+1]) : arrx[i];
         x2 = handle.original ? 0.5 * (arrx[i+1] + arrx[i+2]) : arrx[i+1];
         for (j = handle.j1; j < handle.j2-1; ++j) {
            y1 = handle.original ? 0.5 * (arry[j] + arry[j+1]) : arry[j];
            y2 = handle.original ? 0.5 * (arry[j+1] + arry[j+2]) : arry[j+1];

            z11 = main_grz(histo.getBinContent(i+1, j+1));
            z12 = main_grz(histo.getBinContent(i+1, j+2));
            z21 = main_grz(histo.getBinContent(i+2, j+1));
            z22 = main_grz(histo.getBinContent(i+2, j+2));

            triangles.addMainTriangle(x1, y1, z11, x2, y2, z22, x1, y2, z12, true, handle, i, j);

            triangles.addMainTriangle(x1, y1, z11, x2, y1, z21, x2, y2, z22, false, handle, i, j);

            triangles.addLineSegment(x1, y2, z12, x1, y1, z11);
            triangles.addLineSegment(x1, y1, z11, x2, y1, z21);

            if (i === handle.i2 - 2) triangles.addLineSegment(x2, y1, z21, x2, y2, z22);
            if (j === handle.j2 - 2) triangles.addLineSegment(x1, y2, z12, x2, y2, z22);
         }
      }
   }

   triangles.callFuncs(meshFunc, linesFunc);
}


/**
 * @summary Painter for TH2 classes
 * @private
 */

class TH2Painter extends THistPainter {

   /** @summary constructor
     * @param {object} histo - histogram object */
   constructor(dom, histo) {
      super(dom, histo);
      this.wheel_zoomy = true;
      this._show_empty_bins = false;
   }

   /** @summary cleanup painter */
   cleanup() {
      delete this.tt_handle;

      super.cleanup();
   }

   /** @summary Toggle projection */
   toggleProjection(kind, width) {
      if ((kind === 'Projections') || (kind === 'Off'))
         kind = '';

      let widthX = width, widthY = width;

      if (isStr(kind) && (kind.indexOf('XY') === 0)) {
         const ws = (kind.length > 2) ? kind.slice(2) : '';
         kind = 'XY';
         widthX = widthY = parseInt(ws) || 1;
      } else if (isStr(kind) && (kind.length > 1)) {
         const ps = kind.indexOf('_');
         if ((ps > 0) && (kind[0] === 'X') && (kind[ps+1] === 'Y')) {
            widthX = parseInt(kind.slice(1, ps)) || 1;
            widthY = parseInt(kind.slice(ps+2)) || 1;
            kind = 'XY';
         } else if ((ps > 0) && (kind[0] === 'Y') && (kind[ps+1] === 'X')) {
            widthY = parseInt(kind.slice(1, ps)) || 1;
            widthX = parseInt(kind.slice(ps+2)) || 1;
            kind = 'XY';
         } else {
            widthX = widthY = parseInt(kind.slice(1)) || 1;
            kind = kind[0];
         }
      }

      if (!widthX && !widthY)
         widthX = widthY = 1;

      if (kind && (this.is_projection === kind)) {
         if ((this.projection_widthX === widthX) && (this.projection_widthY === widthY))
            kind = '';
          else {
            this.projection_widthX = widthX;
            this.projection_widthY = widthY;
            return;
         }
      }

      delete this.proj_hist;

      const new_proj = (this.is_projection === kind) ? '' : kind;
      this.projection_widthX = widthX;
      this.projection_widthY = widthY;
      this.is_projection = ''; // avoid projection handling until area is created

      return this.provideSpecialDrawArea(new_proj).then(() => { this.is_projection = new_proj; return this.redrawProjection(); });
   }

   /** @summary Redraw projection */
   async redrawProjection(ii1, ii2, jj1, jj2) {
      if (!this.is_projection)
         return false;

      if (jj2 === undefined) {
         if (!this.tt_handle) return;
         ii1 = Math.round((this.tt_handle.i1 + this.tt_handle.i2)/2); ii2 = ii1+1;
         jj1 = Math.round((this.tt_handle.j1 + this.tt_handle.j2)/2); jj2 = jj1+1;
      }

      const canp = this.getCanvPainter();

      if (canp && !canp._readonly && (this.snapid !== undefined)) {
         // this is when projection should be created on the server side
         if (((this.is_projection === 'X') || (this.is_projection === 'XY')) && !canp.websocketTimeout('projX')) {
            if (canp.sendWebsocket(`EXECANDSEND:DXPROJ:${this.snapid}:ProjectionX("_projx",${jj1+1},${jj2},"")`))
               canp.websocketTimeout('projX', 1000);
         }
         if (((this.is_projection === 'Y') || (this.is_projection === 'XY')) && !canp.websocketTimeout('projY')) {
            if (canp.sendWebsocket(`EXECANDSEND:DYPROJ:${this.snapid}:ProjectionY("_projy",${ii1+1},${ii2},"")`))
               canp.websocketTimeout('projY', 1000);
         }
         return true;
      }

      if (this.doing_projection)
         return false;

      this.doing_projection = true;

      const histo = this.getHisto(),
      createXProject = () => {
        const p = createHistogram(clTH1D, this.nbinsx);
        Object.assign(p.fXaxis, histo.fXaxis);
        p.fName = 'xproj';
        p.fTitle = 'X projection';
        return p;
      },
      createYProject = () => {
        const p = createHistogram(clTH1D, this.nbinsy);
        Object.assign(p.fXaxis, histo.fYaxis);
        p.fName = 'yproj';
        p.fTitle = 'Y projection';
        return p;
      },
      fillProjectHist = (kind, p) => {
         let first = 0, last = -1;
         if (kind === 'X') {
            for (let i = 0; i < this.nbinsx; ++i) {
               let sum = 0;
               for (let j = jj1; j < jj2; ++j)
                  sum += histo.getBinContent(i+1, j+1);
               p.setBinContent(i+1, sum);
            }
            p.fTitle = 'X projection ' + (jj1+1 === jj2 ? `bin ${jj2}` : `bins [${jj1+1} .. ${jj2}]`);
            if (this.tt_handle) { first = this.tt_handle.i1+1; last = this.tt_handle.i2; }
         } else {
            for (let j = 0; j < this.nbinsy; ++j) {
               let sum = 0;
               for (let i = ii1; i < ii2; ++i)
                  sum += histo.getBinContent(i+1, j+1);
               p.setBinContent(j+1, sum);
            }
            p.fTitle = 'Y projection ' + (ii1+1 === ii2 ? `bin ${ii2}` : `bins [${ii1+1} .. ${ii2}]`);
            if (this.tt_handle) { first = this.tt_handle.j1+1; last = this.tt_handle.j2; }
         }

         if (first < last) {
            const axis = p.fXaxis;
            axis.fFirst = first;
            axis.fLast = last;

            if (((axis.fFirst === 1) && (axis.fLast === axis.fNbins)) === axis.TestBit(EAxisBits.kAxisRange))
               axis.InvertBit(EAxisBits.kAxisRange);
         }

         // reset statistic before display
         p.fEntries = 0;
         p.fTsumw = 0;
      };

      if (!this.proj_hist) {
         switch (this.is_projection) {
            case 'X':
               this.proj_hist = createXProject();
               break;
            case 'XY':
               this.proj_hist = createXProject();
               this.proj_hist2 = createYProject();
               break;
            default:
               this.proj_hist = createYProject();
         }
      }

      if (this.is_projection === 'XY') {
         fillProjectHist('X', this.proj_hist);
         fillProjectHist('Y', this.proj_hist2);
         return this.drawInSpecialArea(this.proj_hist, '', 'X')
                    .then(() => this.drawInSpecialArea(this.proj_hist2, '', 'Y'))
                    .then(res => { delete this.doing_projection; return res; });
      }

      fillProjectHist(this.is_projection, this.proj_hist);

      return this.drawInSpecialArea(this.proj_hist).then(res => { delete this.doing_projection; return res; });
   }

   /** @summary Execute TH2 menu command
     * @desc Used to catch standard menu items and provide local implementation */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method, args))
         return true;

      if ((method.fName === 'SetShowProjectionX') || (method.fName === 'SetShowProjectionY')) {
         this.toggleProjection(method.fName[17], args && parseInt(args) ? parseInt(args) : 1);
         return true;
      }

      if (method.fName === 'SetShowProjectionXY') {
         this.toggleProjection('X' + args.replaceAll(',', '_Y'));
         return true;
      }

      return false;
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {
      if (!this.isTH2Poly() && this.getPadPainter()?.iscan) {
         let kind = this.is_projection || '';
         if (kind) kind += this.projection_widthX;
         if ((this.projection_widthX !== this.projection_widthY) && (this.is_projection === 'XY'))
            kind = `X${this.projection_widthX}_Y${this.projection_widthY}`;

         const kinds = ['X1', 'X2', 'X3', 'X5', 'X10', 'Y1', 'Y2', 'Y3', 'Y5', 'Y10', 'XY1', 'XY2', 'XY3', 'XY5', 'XY10'];
         if (kind) kinds.unshift('Off');

         menu.add('sub:Projections', () => menu.input('Input projection kind X1 or XY2 or X3_Y4', kind, 'string').then(val => this.toggleProjection(val)));
         for (let k = 0; k < kinds.length; ++k)
            menu.addchk(kind === kinds[k], kinds[k], kinds[k], arg => this.toggleProjection(arg));
         menu.add('endsub:');
      }

      if (!this.isTH2Poly())
         menu.add('Auto zoom-in', () => this.autoZoom());

      const opts = this.getSupportedDrawOptions();

      menu.addDrawMenu('Draw with', opts, arg => {
         if (arg.indexOf(kInspect) === 0)
            return this.showInspector(arg);
         const oldProject = this.options.Project;
         this.decodeOptions(arg);
         if ((oldProject === this.options.Project) || this.mode3d)
            this.interactiveRedraw('pad', 'drawopt');
         else
            this.toggleProjection(this.options.Project);
      });

      if (this.options.Color || this.options.Contour || this.options.Hist || this.options.Surf || this.options.Lego === 12 || this.options.Lego === 14)
         this.fillPaletteMenu(menu, true);
   }

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      const res = super.clickButton(funcname);
      if (res) return res;

      if (this.isMainPainter()) {
         switch (funcname) {
            case 'ToggleColor': return this.toggleColor();
            case 'ToggleColorZ': return this.toggleColz();
            case 'Toggle3D': return this.toggleMode3D();
         }
      }

      // all methods here should not be processed further
      return false;
   }

   /** @summary Fill pad toolbar with histogram-related functions */
   fillToolbar() {
      super.fillToolbar(true);

      const pp = this.getPadPainter();
      if (!pp) return;

      if (!this.isTH2Poly() && !this.options.Axis)
         pp.addPadButton('th2color', 'Toggle color', 'ToggleColor');
      if (!this.options.Axis)
         pp.addPadButton('th2colorz', 'Toggle color palette', 'ToggleColorZ');
      pp.addPadButton('th2draw3d', 'Toggle 3D mode', 'Toggle3D');
      pp.showPadButtons();
   }

   /** @summary Toggle color drawing mode */
   toggleColor() {
      if (this.options.Mode3D) {
         this.options.Mode3D = false;
         this.options.Color = true;
      } else {
         this.options.Color = !this.options.Color;
         this.options.Scat = !this.options.Color;
      }

      this._can_move_colz = true; // indicate that next redraw can move Z scale

      this.copyOptionsToOthers();

      return this.interactiveRedraw('pad', 'drawopt');
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      if (this.isTH2Poly()) return; // not implemented

      const i1 = this.getSelectIndex('x', 'left', -1),
          i2 = this.getSelectIndex('x', 'right', 1),
          j1 = this.getSelectIndex('y', 'left', -1),
          j2 = this.getSelectIndex('y', 'right', 1),
          histo = this.getObject();

      if ((i1 === i2) || (j1 === j2)) return;

      // first find minimum
      let min = histo.getBinContent(i1 + 1, j1 + 1);
      for (let i = i1; i < i2; ++i) {
         for (let j = j1; j < j2; ++j)
            min = Math.min(min, histo.getBinContent(i + 1, j + 1));
      }
      if (min > 0) return; // if all points positive, no chance for autoscale

      let ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (let i = i1; i < i2; ++i) {
         for (let j = j1; j < j2; ++j) {
            if (histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }
         }
      }

      let xmin, xmax, ymin, ymax, isany = false;

      if ((ileft === iright-1) && (ileft > i1+1) && (iright < i2-1)) { ileft--; iright++; }
      if ((jleft === jright-1) && (jleft > j1+1) && (jright < j2-1)) { jleft--; jright++; }

      if ((ileft > i1 || iright < i2) && (ileft < iright - 1)) {
         xmin = histo.fXaxis.GetBinLowEdge(ileft+1);
         xmax = histo.fXaxis.GetBinLowEdge(iright+1);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = histo.fYaxis.GetBinLowEdge(jleft+1);
         ymax = histo.fYaxis.GetBinLowEdge(jright+1);
         isany = true;
      }

      if (isany)
         return this.getFramePainter().zoom(xmin, xmax, ymin, ymax);
   }

   /** @summary Scan TH2 histogram content */
   scanContent(when_axis_changed) {
      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy) return;

      const histo = this.getObject();
      let i, j;

      this.extractAxesProperties(2);

      if (this.isTH2Poly()) {
         this.gminposbin = null;
         this.gminbin = this.gmaxbin = 0;

         for (let n = 0, len = histo.fBins.arr.length; n < len; ++n) {
            const bin_content = histo.fBins.arr[n].fContent;
            if (n === 0) this.gminbin = this.gmaxbin = bin_content;

            if (bin_content < this.gminbin)
               this.gminbin = bin_content;
            else if (bin_content > this.gmaxbin)
               this.gmaxbin = bin_content;

            if ((bin_content > 0) && ((this.gminposbin === null) || (this.gminposbin > bin_content)))
               this.gminposbin = bin_content;
         }
      } else {
         // global min/max, used at the moment in 3D drawing
         this.gminbin = this.gmaxbin = histo.getBinContent(1, 1);
         this.gminposbin = null;
         for (i = 0; i < this.nbinsx; ++i) {
            for (j = 0; j < this.nbinsy; ++j) {
               const bin_content = histo.getBinContent(i+1, j+1);
               if (bin_content < this.gminbin)
                  this.gminbin = bin_content;
               else if (bin_content > this.gmaxbin)
                  this.gmaxbin = bin_content;
               if (bin_content > 0) {
                  if ((this.gminposbin === null) || (this.gminposbin > bin_content))
                     this.gminposbin = bin_content;
               }
            }
         }
      }

      // this value used for logz scale drawing
      if ((this.gminposbin === null) && (this.gmaxbin > 0))
         this.gminposbin = this.gmaxbin*1e-4;

      const is_content = (this.gmaxbin !== 0) || (this.gminbin !== 0);

      if (this.options.Axis > 0) {
         // Paint histogram axis only
         this.draw_content = false;
      } else if (this.isTH2Poly()) {
         this.draw_content = is_content || this.options.Line || this.options.Fill || this.options.Mark;
         if (!this.draw_content && this.options.Zero) {
            this.draw_content = true;
            this.options.Line = 1;
         }
      } else
         this.draw_content = is_content;
   }

   /** @summary Count TH2 histogram statistic
     * @desc Optionally one could provide condition function to select special range */
   countStat(cond, count_skew) {
      if (!isFunc(cond))
         cond = this.options.cutg ? (x, y) => this.options.cutg.IsInside(x, y) : null;

      const histo = this.getHisto(), xaxis = histo.fXaxis, yaxis = histo.fYaxis,
            fp = this.getFramePainter(),
            funcs = fp.getGrFuncs(this.options.second_x, this.options.second_y),
            res = { name: histo.fName, entries: 0, eff_entries: 0, integral: 0,
                    meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    xmax: 0, ymax: 0, wmax: null, skewx: 0, skewy: 0, skewd: 0, kurtx: 0, kurty: 0, kurtd: 0 },
            has_counted_stat = !fp.isAxisZoomed('x') && !fp.isAxisZoomed('y') && (Math.abs(histo.fTsumw) > 1e-300) && !cond;
      let stat_sum0 = 0, stat_sumw2 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0,
          xside, yside, xx, yy, zz, xleft, xright, yleft, yright;

      if (this.isTH2Poly()) {
         const len = histo.fBins.arr.length;
         let i, bin, n, gr, ngr, numgraphs, numpoints;

         for (i = 0; i < len; ++i) {
            bin = histo.fBins.arr[i];

            xside = (bin.fXmin > funcs.scale_xmax) ? 2 : (bin.fXmax < funcs.scale_xmin ? 0 : 1);
            yside = (bin.fYmin > funcs.scale_ymax) ? 2 : (bin.fYmax < funcs.scale_ymin ? 0 : 1);

            xx = yy = numpoints = 0;
            gr = bin.fPoly; numgraphs = 1;
            if (gr._typename === clTMultiGraph) { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

            for (ngr = 0; ngr < numgraphs; ++ngr) {
               if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];

               for (n = 0; n < gr.fNpoints; ++n) {
                  ++numpoints;
                  xx += gr.fX[n];
                  yy += gr.fY[n];
               }
            }

            if (numpoints > 1) {
               xx = xx / numpoints;
               yy = yy / numpoints;
            }

            zz = bin.fContent;

            res.entries += zz;

            res.matrix[yside * 3 + xside] += zz;

            if ((xside !== 1) || (yside !== 1) || (cond && !cond(xx, yy))) continue;

            if ((res.wmax === null) || (zz > res.wmax)) {
               res.wmax = zz;
               res.xmax = xx;
               res.ymax = yy;
            }

            if (!has_counted_stat) {
               stat_sum0 += zz;
               stat_sumw2 += zz * zz;
               stat_sumx1 += xx * zz;
               stat_sumy1 += yy * zz;
               stat_sumx2 += xx * xx * zz;
               stat_sumy2 += yy * yy * zz;
            }
         }
      } else {
         xleft = this.getSelectIndex('x', 'left');
         xright = this.getSelectIndex('x', 'right');
         yleft = this.getSelectIndex('y', 'left');
         yright = this.getSelectIndex('y', 'right');

         for (let xi = 0; xi <= this.nbinsx + 1; ++xi) {
            xside = (xi <= xleft) ? 0 : (xi > xright ? 2 : 1);
            xx = xaxis.GetBinCoord(xi - 0.5);

            for (let yi = 0; yi <= this.nbinsy + 1; ++yi) {
               yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
               yy = yaxis.GetBinCoord(yi - 0.5);

               zz = histo.getBinContent(xi, yi);

               res.entries += zz;

               res.matrix[yside * 3 + xside] += zz;

               if ((xside !== 1) || (yside !== 1) || (cond && !cond(xx, yy))) continue;

               if ((res.wmax === null) || (zz > res.wmax)) {
                  res.wmax = zz;
                  res.xmax = xx;
                  res.ymax = yy;
               }

               if (!has_counted_stat) {
                  stat_sum0 += zz;
                  stat_sumw2 += zz * zz;
                  stat_sumx1 += xx * zz;
                  stat_sumy1 += yy * zz;
                  stat_sumx2 += xx**2 * zz;
                  stat_sumy2 += yy**2 * zz;
                  // stat_sumxy += xx * yy * zz;
               }
            }
         }
      }

      if (has_counted_stat) {
         stat_sum0 = histo.fTsumw;
         stat_sumw2 = histo.fTsumw2;
         stat_sumx1 = histo.fTsumwx;
         stat_sumx2 = histo.fTsumwx2;
         stat_sumy1 = histo.fTsumwy;
         stat_sumy2 = histo.fTsumwy2;
         // stat_sumxy = histo.fTsumwxy;
      }

      if (Math.abs(stat_sum0) > 1e-300) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx**2));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany**2));
      }

      if (res.wmax === null)
         res.wmax = 0;
      res.integral = stat_sum0;

      if (histo.fEntries > 1)
         res.entries = histo.fEntries;

      res.eff_entries = stat_sumw2 ? stat_sum0*stat_sum0/stat_sumw2 : Math.abs(stat_sum0);

      if (count_skew && !this.isTH2Poly()) {
         let sumx3 = 0, sumy3 = 0, sumx4 = 0, sumy4 = 0, np = 0, w = 0;
         for (let xi = xleft; xi < xright; ++xi) {
            xx = xaxis.GetBinCoord(xi + 0.5);
            for (let yi = yleft; yi < yright; ++yi) {
               yy = yaxis.GetBinCoord(yi + 0.5);
               if (cond && !cond(xx, yy)) continue;
               w = histo.getBinContent(xi + 1, yi + 1);
               np += w;
               sumx3 += w * Math.pow(xx - res.meanx, 3);
               sumy3 += w * Math.pow(yy - res.meany, 3);
               sumx4 += w * Math.pow(xx - res.meanx, 4);
               sumy4 += w * Math.pow(yy - res.meany, 4);
            }
         }

         const stddev3x = Math.pow(res.rmsx, 3),
               stddev3y = Math.pow(res.rmsy, 3),
               stddev4x = Math.pow(res.rmsx, 4),
               stddev4y = Math.pow(res.rmsy, 4);
         if (np * stddev3x !== 0)
            res.skewx = sumx3 / (np * stddev3x);
         if (np * stddev3y !== 0)
            res.skewy = sumy3 / (np * stddev3y);
         res.skewd = res.eff_entries > 0 ? Math.sqrt(6/res.eff_entries) : 0;
         if (np * stddev4x !== 0)
            res.kurtx = sumx4 / (np * stddev4x) - 3;
         if (np * stddev4y !== 0)
            res.kurty = sumy4 / (np * stddev4y) - 3;
         res.kurtd = res.eff_entries > 0 ? Math.sqrt(24/res.eff_entries) : 0;
      }

      return res;
   }

   /** @summary Fill TH2 statistic in stat box */
   fillStatistic(stat, dostat, dofit) {
      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill()) return false;

      if (dostat === 1) dostat = 1111;

      const print_name = Math.floor(dostat % 10),
            print_entries = Math.floor(dostat / 10) % 10,
            print_mean = Math.floor(dostat / 100) % 10,
            print_rms = Math.floor(dostat / 1000) % 10,
            print_under = Math.floor(dostat / 10000) % 10,
            print_over = Math.floor(dostat / 100000) % 10,
            print_integral = Math.floor(dostat / 1000000) % 10,
            print_skew = Math.floor(dostat / 10000000) % 10,
            print_kurt = Math.floor(dostat / 100000000) % 10,
            data = this.countStat(undefined, (print_skew > 0) || (print_kurt > 0));

      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (print_entries > 0)
         stat.addText('Entries = ' + stat.format(data.entries, 'entries'));

      if (print_mean > 0) {
         stat.addText('Mean x = ' + stat.format(data.meanx));
         stat.addText('Mean y = ' + stat.format(data.meany));
      }

      if (print_rms > 0) {
         stat.addText('Std Dev x = ' + stat.format(data.rmsx));
         stat.addText('Std Dev y = ' + stat.format(data.rmsy));
      }

      if (print_integral > 0)
         stat.addText('Integral = ' + stat.format(data.matrix[4], 'entries'));

      if (print_skew === 2) {
         stat.addText(`Skewness x = ${stat.format(data.skewx)} #pm ${stat.format(data.skewd)}`);
         stat.addText(`Skewness y = ${stat.format(data.skewy)} #pm ${stat.format(data.skewd)}`);
      } else if (print_skew > 0) {
         stat.addText(`Skewness x = ${stat.format(data.skewx)}`);
         stat.addText(`Skewness y = ${stat.format(data.skewy)}`);
      }

      if (print_kurt === 2) {
         stat.addText(`Kurtosis x = ${stat.format(data.kurtx)} #pm ${stat.format(data.kurtd)}`);
         stat.addText(`Kurtosis y = ${stat.format(data.kurty)} #pm ${stat.format(data.kurtd)}`);
      } else if (print_kurt > 0) {
         stat.addText(`Kurtosis x = ${stat.format(data.kurtx)}`);
         stat.addText(`Kurtosis y = ${stat.format(data.kurty)}`);
      }

      if ((print_under > 0) || (print_over > 0)) {
         const get = i => data.matrix[i].toFixed(0);

         stat.addText(`${get(6)} | ${get(7)} | ${get(7)}`);
         stat.addText(`${get(3)} | ${get(4)} | ${get(5)}`);
         stat.addText(`${get(0)} | ${get(1)} | ${get(2)}`);
      }

      if (dofit) stat.fillFunctionStat(this.findFunction(clTF2), dofit, 2);

      return true;
   }

   /** @summary Draw TH2 bins as colors */
   drawBinsColor() {
      const histo = this.getHisto(),
            handle = this.prepareDraw(),
            cntr = this.getContour(),
            palette = this.getHistPalette(),
            entries = [],
            show_empty = this._show_empty_bins,
            can_merge_x = (handle.xbar2 === 1) && (handle.xbar1 === 0),
            can_merge_y = (handle.ybar2 === 1) && (handle.ybar1 === 0);

      let dx, dy, x1, y2, binz, is_zero, colindx, last_entry = null,
          skip_zero = !this.options.Zero;

      const test_cutg = this.options.cutg,
            flush_last_entry = () => {
         last_entry.path += `h${dx}v${last_entry.y1-last_entry.y2}h${-dx}z`;
         last_entry = null;
      };

      // check in the beginning if zero can be skipped
      if (!skip_zero && !show_empty && (cntr.getPaletteIndex(palette, 0) === null)) skip_zero = true;

      // now start build
      for (let i = handle.i1; i < handle.i2; ++i) {
         dx = (handle.grx[i+1] - handle.grx[i]) || 1;
         if (can_merge_x)
            x1 = handle.grx[i];
         else {
            x1 = Math.round(handle.grx[i] + dx*handle.xbar1);
            dx = Math.round(dx*(handle.xbar2 - handle.xbar1)) || 1;
         }

         for (let j = handle.j2 - 1; j >= handle.j1; --j) {
            binz = histo.getBinContent(i + 1, j + 1);
            is_zero = (binz === 0);

            if ((is_zero && skip_zero) || (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5), histo.fYaxis.GetBinCoord(j + 0.5)))) {
               if (last_entry) flush_last_entry();
               continue;
            }

            colindx = cntr.getPaletteIndex(palette, binz);
            if (colindx === null) {
               if (is_zero && show_empty)
                  colindx = 0;
                 else {
                   if (last_entry) flush_last_entry();
                   continue;
                }
            }

            dy = (handle.gry[j] - handle.gry[j+1]) || 1;
            if (can_merge_y)
               y2 = handle.gry[j+1];
            else {
               y2 = Math.round(handle.gry[j] - dy*handle.ybar2);
               dy = Math.round(dy*(handle.ybar2 - handle.ybar1)) || 1;
            }

            const cmd1 = `M${x1},${y2}`;
            let entry = entries[colindx];
            if (!entry)
               entry = entries[colindx] = { path: cmd1 };
             else if (can_merge_y && (entry === last_entry)) {
               entry.y1 = y2 + dy;
               continue;
            } else {
               const ddx = x1 - entry.x1, ddy = y2 - entry.y2;
               if (ddx || ddy) {
                  const cmd2 = `m${ddx},${ddy}`;
                  entry.path += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               }
            }
            if (last_entry) flush_last_entry();

            entry.x1 = x1;
            entry.y2 = y2;

            if (can_merge_y) {
               entry.y1 = y2 + dy;
               last_entry = entry;
            } else
               entry.path += `h${dx}v${dy}h${-dx}z`;
         }
         if (last_entry) flush_last_entry();
      }

      entries.forEach((entry, colindx) => {
         if (entry) {
            this.draw_g
                .append('svg:path')
                .attr('fill', palette.getColor(colindx))
                .attr('d', entry.path);
         }
      });

      return handle;
   }

   /** @summary Draw histogram bins with projection function */
   drawBinsProjected() {
      const handle = this.prepareDraw({ rounding: false, nozoom: true, extra: 100, original: true }),
            main = this.getFramePainter(),
            funcs = main.getGrFuncs(this.options.second_x, this.options.second_y),
            ilevels = this.getContourLevels(),
            palette = this.getHistPalette(),
            func = main.getProjectionFunc();

      handle.grz = z => z;
      handle.grz_min = ilevels[0];
      handle.grz_max = ilevels[ilevels.length - 1];

      buildSurf3D(this.getHisto(), handle, ilevels, (lvl, pos) => {
         let dd = '', lastx, lasty;

         for (let i = 0; i < pos.length; i += 3) {
            const pnt = func(pos[i], pos[i + 1]),
                x = Math.round(funcs.grx(pnt.x)),
                y = Math.round(funcs.gry(pnt.y));

            if (i === 0)
               dd = `M${x},${y}`;
              else {
               if ((x === lastx) && (y === lasty))
                  continue;
               if (i % 9 === 0)
                  dd += `m${x-lastx},${y-lasty}`;
               else if (y === lasty)
                  dd += `h${x-lastx}`;
               else if (x === lastx)
                  dd += `v${y-lasty}`;
               else
                  dd += `l${x-lastx},${y-lasty}`;
            }

            lastx = x; lasty = y;
         }

         this.draw_g
             .append('svg:path')
             .attr('d', dd)
             .style('fill', palette.calcColor(lvl, ilevels.length));
      });

      return handle;
   }

   /** @summary Draw histogram bins as contour */
   drawBinsContour() {
      const handle = this.prepareDraw({ rounding: false, extra: 100 }),
            main = this.getFramePainter(),
            frame_w = main.getFrameWidth(),
            frame_h = main.getFrameHeight(),
            levels = this.getContourLevels(),
            palette = this.getHistPalette(),

       get_segm_intersection = (segm1, segm2) => {
          const s10_x = segm1.x2 - segm1.x1,
                s10_y = segm1.y2 - segm1.y1,
                s32_x = segm2.x2 - segm2.x1,
                s32_y = segm2.y2 - segm2.y1,
                denom = s10_x * s32_y - s32_x * s10_y;

          if (denom === 0)
              return 0; // Collinear
          const denomPositive = denom > 0,
              s02_x = segm1.x1 - segm2.x1,
              s02_y = segm1.y1 - segm2.y1,
              s_numer = s10_x * s02_y - s10_y * s02_x;
          if ((s_numer < 0) === denomPositive)
              return null; // No collision

          const t_numer = s32_x * s02_y - s32_y * s02_x;
          if ((t_numer < 0) === denomPositive)
              return null; // No collision

          if (((s_numer > denom) === denomPositive) || ((t_numer > denom) === denomPositive))
              return null; // No collision
          // Collision detected
          const t = t_numer / denom;
          return { x: Math.round(segm1.x1 + (t * s10_x)), y: Math.round(segm1.y1 + (t * s10_y)) };
      }, buildPath = (xp, yp, iminus, iplus, do_close, check_rapair) => {
         let cmd = '', lastx, lasty, x0, y0, isany = false, matched, x, y;
         for (let i = iminus; i <= iplus; ++i) {
            x = Math.round(xp[i]);
            y = Math.round(yp[i]);
            if (!cmd) {
               cmd = `M${x},${y}`; x0 = x; y0 = y;
            } else if ((i === iplus) && (iminus !== iplus) && (x === x0) && (y === y0)) {
               if (!isany) return ''; // all same points
               cmd += 'z'; do_close = false; matched = true;
            } else {
               const dx = x - lastx, dy = y - lasty;
               if (dx) {
                  isany = true;
                  cmd += dy ? `l${dx},${dy}` : `h${dx}`;
               } else if (dy) {
                  isany = true;
                  cmd += `v${dy}`;
               }
            }

            lastx = x; lasty = y;
         }

         if (!do_close || matched || !check_rapair)
            return do_close ? cmd + 'z' : cmd;

         // try to build path which fills area to outside borders

         const points = [{ x: 0, y: 0 }, { x: frame_w, y: 0 }, { x: frame_w, y: frame_h }, { x: 0, y: frame_h }],

          get_intersect = (i, di) => {
            const segm = { x1: xp[i], y1: yp[i], x2: 2*xp[i] - xp[i+di], y2: 2*yp[i] - yp[i+di] };
            for (let i = 0; i < 4; ++i) {
               const res = get_segm_intersection(segm, { x1: points[i].x, y1: points[i].y, x2: points[(i+1)%4].x, y2: points[(i+1)%4].y });
               if (res) {
                  res.indx = i + 0.5;
                  return res;
               }
            }
            return null;
         };

         let pnt1, pnt2;
         iminus--;
         while ((iminus < iplus - 1) && !pnt1)
            pnt1 = get_intersect(++iminus, 1);
         if (!pnt1) return '';
         iplus++;
         while ((iminus < iplus - 1) && !pnt2)
            pnt2 = get_intersect(--iplus, -1);
         if (!pnt2) return '';

         // TODO: now side is always same direction, could be that side should be checked more precise

         let dd = buildPath(xp, yp, iminus, iplus),
             indx = pnt2.indx;
         const side = 1, step = side*0.5;

         dd += `L${pnt2.x},${pnt2.y}`;

         while (Math.abs(indx - pnt1.indx) > 0.1) {
            indx = Math.round(indx + step) % 4;
            dd += `L${points[indx].x},${points[indx].y}`;
            indx += step;
         }
         return dd + `L${pnt1.x},${pnt1.y}z`;
      };

      if (this.options.Contour === 14) {
         this.draw_g
             .append('svg:path')
             .attr('d', `M0,0h${frame_w}v${frame_h}h${-frame_w}z`)
             .style('fill', palette.calcColor(0, levels.length));
      }

      buildHist2dContour(this.getHisto(), handle, levels, palette, (colindx, xp, yp, iminus, iplus, ipoly) => {
         const icol = palette.getColor(colindx);
         let fillcolor = icol, lineatt;

         switch (this.options.Contour) {
            case 1: break;
            case 11: fillcolor = 'none'; lineatt = this.createAttLine({ color: icol, std: false }); break;
            case 12: fillcolor = 'none'; lineatt = this.createAttLine({ color: 1, style: (ipoly%5 + 1), width: 1, std: false }); break;
            case 13: fillcolor = 'none'; lineatt = this.lineatt; break;
            case 14: break;
         }

         const dd = buildPath(xp, yp, iminus, iplus, fillcolor !== 'none', true);
         if (!dd) return;

         const elem = this.draw_g
                        .append('svg:path')
                        .attr('d', dd)
                        .style('fill', fillcolor);

         if (lineatt)
            elem.call(lineatt.func);
      });

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   getGrNPoints(gr) {
      const x = gr.fX, y = gr.fY;
      let npnts = gr.fNpoints;
      if ((npnts > 2) && (x[0] === x[npnts-1]) && (y[0] === y[npnts-1]))
         npnts--;
      return npnts;
   }

   /** @summary Create single graph path from TH2PolyBin */
   createPolyGr(funcs, gr, textbin) {
      let grcmd = '', acc_x = 0, acc_y = 0;

      const x = gr.fX, y = gr.fY,
         flush = () => {
            if (acc_x) { grcmd += 'h' + acc_x; acc_x = 0; }
            if (acc_y) { grcmd += 'v' + acc_y; acc_y = 0; }
         }, addPoint = (x1, y1, x2, y2) => {
            const len = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
            textbin.sumx += (x1 + x2) * len / 2;
            textbin.sumy += (y1 + y2) * len / 2;
            textbin.sum += len;
         }, npnts = this.getGrNPoints(gr);

      if (npnts < 2)
         return '';

      const grx0 = Math.round(funcs.grx(x[0])),
            gry0 = Math.round(funcs.gry(y[0]));
      let grx = grx0, gry = gry0;

      for (let n = 1; n < npnts; ++n) {
         const nextx = Math.round(funcs.grx(x[n])),
               nexty = Math.round(funcs.gry(y[n])),
               dx = nextx - grx,
               dy = nexty - gry;

         if (textbin) addPoint(grx, gry, nextx, nexty);
         if (dx || dy) {
            if (dx === 0) {
               if ((acc_y === 0) || ((dy < 0) !== (acc_y < 0))) flush();
               acc_y += dy;
            } else if (dy === 0) {
               if ((acc_x === 0) || ((dx < 0) !== (acc_x < 0))) flush();
               acc_x += dx;
            } else {
               flush();
               grcmd += `l${dx},${dy}`;
            }

            grx = nextx; gry = nexty;
         }
      }

      if (textbin) addPoint(grx, gry, grx0, gry0);
      flush();

      return grcmd ? `M${grx0},${gry0}` + grcmd + 'z' : '';
   }

   /** @summary Create path for complete TH2PolyBin */
   createPolyBin(funcs, bin) {
      const arr = (bin.fPoly._typename === clTMultiGraph) ? bin.fPoly.fGraphs.arr : [bin.fPoly];
      let cmd = '';
      for (let k = 0; k < arr.length; ++k)
         cmd += this.createPolyGr(funcs, arr[k]);
      return cmd;
   }

   /** @summary draw TH2Poly bins */
   async drawPolyBins() {
      const histo = this.getObject(),
            pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            draw_colors = this.options.Color || (!this.options.Line && !this.options.Fill && !this.options.Text && !this.options.Mark),
            draw_lines = this.options.Line || (this.options.Text && !draw_colors),
            draw_fill = this.options.Fill && !draw_colors,
            draw_mark = this.options.Mark,
            h = pmain.getFrameHeight(),
            textbins = [],
            len = histo.fBins.arr.length;
       let colindx, cmd,
           full_cmd = '', allmarkers_cmd = '',
           bin, item, i, gr0 = null,
           lineatt_match = draw_lines,
           fillatt_match = draw_fill,
           markatt_match = draw_mark;

      // force recalculations of contours
      // use global coordinates
      this.maxbin = this.gmaxbin;
      this.minbin = this.gminbin;
      this.minposbin = this.gminposbin;

      const cntr = draw_colors ? this.getContour(true) : null,
            palette = cntr ? this.getHistPalette() : null,
            rejectBin = bin => {
               // check if bin outside visible range
               return ((bin.fXmin > funcs.scale_xmax) || (bin.fXmax < funcs.scale_xmin) ||
                       (bin.fYmin > funcs.scale_ymax) || (bin.fYmax < funcs.scale_ymin));
            };

      // check if similar fill attributes
      for (i = 0; i < len; ++i) {
         bin = histo.fBins.arr[i];
         if (rejectBin(bin)) continue;

         const arr = (bin.fPoly._typename === clTMultiGraph) ? bin.fPoly.fGraphs.arr : [bin.fPoly];
         for (let k = 0; k < arr.length; ++k) {
            const gr = arr[k];
            if (!gr0) { gr0 = gr; continue; }
            if (lineatt_match && ((gr0.fLineColor !== gr.fLineColor) || (gr0.fLineWidth !== gr.fLineWidth) || (gr0.fLineStyle !== gr.fLineStyle)))
               lineatt_match = false;
            if (fillatt_match && ((gr0.fFillColor !== gr.fFillColor) || (gr0.fFillStyle !== gr.fFillStyle)))
               fillatt_match = false;
            if (markatt_match && ((gr0.fMarkerColor !== gr.fMarkerColor) || (gr0.fMarkerStyle !== gr.fMarkerStyle) || (gr0.fMarkerSize !== gr.fMarkerSize)))
               markatt_match = false;
         }
         if (!lineatt_match && !fillatt_match && !markatt_match)
            break;
      }

      // do not try color draw optimization as with plain th2 while
      // bins are not rectangular and drawings artefacts are nasty
      // therefore draw each bin separately when doing color draw
      const lineatt0 = lineatt_match && gr0 ? this.createAttLine(gr0) : null,
            fillatt0 = fillatt_match && gr0 ? this.createAttFill(gr0) : null,
            markeratt0 = markatt_match && gr0 ? this.createAttMarker({ attr: gr0, style: this.options.MarkStyle, std: false }) : null,
            optimize_draw = !draw_colors && (draw_lines ? lineatt_match : true) && (draw_fill ? fillatt_match : true);

      // draw bins
      for (i = 0; i < len; ++i) {
         bin = histo.fBins.arr[i];
         if (rejectBin(bin)) continue;

         const draw_bin = bin.fContent || this.options.Zero,
               arr = (bin.fPoly._typename === clTMultiGraph) ? bin.fPoly.fGraphs.arr : [bin.fPoly];

         colindx = draw_colors && draw_bin ? cntr.getPaletteIndex(palette, bin.fContent) : null;

         const textbin = this.options.Text && draw_bin ? { bin, sumx: 0, sumy: 0, sum: 0 } : null;

         for (let k = 0; k < arr.length; ++k) {
            const gr = arr[k];
            if (markeratt0) {
               const npnts = this.getGrNPoints(gr);
               for (let n = 0; n < npnts; ++n)
                  allmarkers_cmd += markeratt0.create(funcs.grx(gr.fX[n]), funcs.gry(gr.fY[n]));
            }

            cmd = this.createPolyGr(funcs, gr, textbin);
            if (!cmd) continue;

            if (optimize_draw)
               full_cmd += cmd;
            else if ((colindx !== null) || draw_fill || draw_lines) {
               item = this.draw_g.append('svg:path').attr('d', cmd);
               if (draw_colors && (colindx !== null))
                  item.style('fill', this._color_palette.getColor(colindx));
               else if (draw_fill)
                  item.call('fill', this.createAttFill(gr).func);
               else
                  item.style('fill', 'none');
               if (draw_lines)
                  item.call(this.createAttLine(gr).func);
            }
         } // loop over graphs

         if (textbin?.sum)
            textbins.push(textbin);
      } // loop over bins

      if (optimize_draw) {
         item = this.draw_g.append('svg:path').attr('d', full_cmd);
         if (draw_fill && fillatt0)
            item.call(fillatt0.func);
         else
            item.style('fill', 'none');
         if (draw_lines && lineatt0)
            item.call(lineatt0.func);
      }

      if (markeratt0 && !markeratt0.empty() && allmarkers_cmd) {
         this.draw_g.append('svg:path')
                    .attr('d', allmarkers_cmd)
                    .call(markeratt0.func);
      } else if (draw_mark) {
         for (i = 0; i < len; ++i) {
            bin = histo.fBins.arr[i];
            if (rejectBin(bin)) continue;

            const arr = (bin.fPoly._typename === clTMultiGraph) ? bin.fPoly.fGraphs.arr : [bin.fPoly];

            for (let k = 0; k < arr.length; ++k) {
               const gr = arr[k], npnts = this.getGrNPoints(gr),
                     markeratt = this.createAttMarker({ attr: gr, style: this.options.MarkStyle, std: false });
               if (!npnts || markeratt.empty())
                  continue;
               let cmd = '';

               for (let n = 0; n < npnts; ++n)
                  cmd += markeratt.create(funcs.grx(gr.fX[n]), funcs.gry(gr.fY[n]));

               this.draw_g.append('svg:path')
                   .attr('d', cmd)
                   .call(markeratt.func);
            } // loop over graphs
         } // loop over bins
      }

      let pr = Promise.resolve(true);

      if (textbins.length > 0) {
         const color = this.getColor(histo.fMarkerColor),
               rotate = -1*this.options.TextAngle,
               text_g = this.draw_g.append('svg:g').attr('class', 'th2poly_text'),
               text_size = ((histo.fMarkerSize !== 1) && rotate) ? Math.round(0.02*h*histo.fMarkerSize) : 12;

         this.startTextDrawing(42, text_size, text_g, text_size);

         for (i = 0; i < textbins.length; ++i) {
            const textbin = textbins[i];

            bin = textbin.bin;

            if (textbin.sum > 0) {
               textbin.midx = Math.round(textbin.sumx / textbin.sum);
               textbin.midy = Math.round(textbin.sumy / textbin.sum);
            } else {
               textbin.midx = Math.round(funcs.grx((bin.fXmin + bin.fXmax)/2));
               textbin.midy = Math.round(funcs.gry((bin.fYmin + bin.fYmax)/2));
            }

            let text;

            if (!this.options.TextKind)
               text = (Math.round(bin.fContent) === bin.fContent) ? bin.fContent.toString() : floatToString(bin.fContent, gStyle.fPaintTextFormat);
             else {
               text = bin.fPoly?.fName;
               if (!text || (text === 'Graph'))
                  text = bin.fNumber.toString();
            }

            this.drawText({ align: 22, x: textbin.midx, y: textbin.midy, rotate, text, color, latex: 0, draw_g: text_g });
         }

         pr = this.finishTextDrawing(text_g, true);
      }

      return pr.then(() => { return { poly: true }; });
   }

   /** @summary Draw TH2 bins as text */
   async drawBinsText(handle) {
      const histo = this.getObject(),
            test_cutg = this.options.cutg,
            color = this.getColor(histo.fMarkerColor),
            rotate = -1*this.options.TextAngle,
            draw_g = this.draw_g.append('svg:g').attr('class', 'th2_text'),
            show_err = (this.options.TextKind === 'E'),
            latex = (show_err && !this.options.TextLine) ? 1 : 0;
      let x, y, width, height,
          text_size = 20, text_offset = 0;

      if (!handle) handle = this.prepareDraw({ rounding: false });

      if ((histo.fMarkerSize !== 1) && rotate)
         text_size = Math.round(0.02*histo.fMarkerSize*this.getFramePainter().getFrameHeight());

      if (histo.fBarOffset !== 0) text_offset = histo.fBarOffset*1e-3;

      this.startTextDrawing(42, text_size, draw_g, text_size);

      for (let i = handle.i1; i < handle.i2; ++i) {
         const binw = handle.grx[i+1] - handle.grx[i];
         for (let j = handle.j1; j < handle.j2; ++j) {
            const binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;

            if (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5),
                     histo.fYaxis.GetBinCoord(j + 0.5))) continue;

            const binh = handle.gry[j] - handle.gry[j+1];

            let text = (binz === Math.round(binz)) ? binz.toString() : floatToString(binz, gStyle.fPaintTextFormat);

            if (show_err) {
               const errz = histo.getBinError(histo.getBin(i+1, j+1)),
                   lble = (errz === Math.round(errz)) ? errz.toString() : floatToString(errz, gStyle.fPaintTextFormat);
               if (this.options.TextLine)
                  text += '\xB1' + lble;
               else
                  text = `#splitline{${text}}{#pm${lble}}`;
            }

            if (rotate /* || (histo.fMarkerSize !== 1) */) {
               x = Math.round(handle.grx[i] + binw*0.5);
               y = Math.round(handle.gry[j+1] + binh*(0.5 + text_offset));
               width = height = 0;
            } else {
               x = Math.round(handle.grx[i] + binw*0.1);
               y = Math.round(handle.gry[j+1] + binh*(0.1 + text_offset));
               width = Math.round(binw*0.8);
               height = Math.round(binh*0.8);
            }

            this.drawText({ align: 22, x, y, width, height, rotate, text, color, latex, draw_g });
         }
      }

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return this.finishTextDrawing(draw_g, true).then(() => handle);
   }

   /** @summary Draw TH2 bins as arrows */
   drawBinsArrow() {
      const histo = this.getObject(),
            test_cutg = this.options.cutg,
            handle = this.prepareDraw({ rounding: false }),
            scale_x = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1)/2,
            scale_y = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1)/2,
            makeLine = (dx, dy) => dx ? (dy ? `l${dx},${dy}` : `h${dx}`) : (dy ? `v${dy}` : '');
      let i, j, dn = 1e-30, dx, dy, xc, yc, cmd = '',
          dxn, dyn, x1, x2, y1, y2, anr, si, co;

      for (let loop = 0; loop < 2; ++loop) {
         for (i = handle.i1; i < handle.i2; ++i) {
            for (j = handle.j1; j < handle.j2; ++j) {
               if (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5),
                     histo.fYaxis.GetBinCoord(j + 0.5))) continue;

               if (i === handle.i1)
                  dx = histo.getBinContent(i+2, j+1) - histo.getBinContent(i+1, j+1);
               else if (i === handle.i2-1)
                  dx = histo.getBinContent(i+1, j+1) - histo.getBinContent(i, j+1);
               else
                  dx = 0.5*(histo.getBinContent(i+2, j+1) - histo.getBinContent(i, j+1));

               if (j === handle.j1)
                  dy = histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j+1);
               else if (j === handle.j2-1)
                  dy = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1, j);
               else
                  dy = 0.5*(histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j));

               if (loop === 0)
                  dn = Math.max(dn, Math.abs(dx), Math.abs(dy));
                else {
                  xc = (handle.grx[i] + handle.grx[i+1])/2;
                  yc = (handle.gry[j] + handle.gry[j+1])/2;
                  dxn = scale_x*dx/dn;
                  dyn = scale_y*dy/dn;
                  x1 = xc - dxn;
                  x2 = xc + dxn;
                  y1 = yc - dyn;
                  y2 = yc + dyn;
                  dx = Math.round(x2-x1);
                  dy = Math.round(y2-y1);

                  if (dx || dy) {
                     cmd += `M${Math.round(x1)},${Math.round(y1)}${makeLine(dx, dy)}`;

                     if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        anr = Math.sqrt(9/(dx**2 + dy**2));
                        si = Math.round(anr*(dx + dy));
                        co = Math.round(anr*(dx - dy));
                        if (si || co)
                           cmd += `m${-si},${co}${makeLine(si, -co)}${makeLine(-co, -si)}`;
                     }
                  }
               }
            }
         }
      }

      this.draw_g
         .append('svg:path')
         .attr('d', cmd)
         .style('fill', 'none')
         .call(this.lineatt.func);

      return handle;
   }

   /** @summary Draw TH2 bins as boxes */
   drawBinsBox() {
      const histo = this.getObject(),
          handle = this.prepareDraw({ rounding: false }),
          main = this.getMainPainter();

      if (main === this) {
         if (main.maxbin === main.minbin) {
            main.maxbin = main.gmaxbin;
            main.minbin = main.gminbin;
            main.minposbin = main.gminposbin;
         }
         if (main.maxbin === main.minbin)
            main.minbin = Math.min(0, main.maxbin-1);
      }

      const absmax = Math.max(Math.abs(main.maxbin), Math.abs(main.minbin)),
            absmin = Math.max(0, main.minbin),
            pad = this.getPadPainter().getRootPad(true),
            test_cutg = this.options.cutg;
      let i, j, binz, absz, res = '', cross = '', btn1 = '', btn2 = '',
          zdiff, dgrx, dgry, xx, yy, ww, hh, xyfactor,
          uselogz = false, logmin = 0;

      if ((pad?.fLogv ?? pad?.fLogz) && (absmax > 0)) {
         uselogz = true;
         const logmax = Math.log(absmax);
         if (absmin > 0)
            logmin = Math.log(absmin);
         else if ((main.minposbin>=1) && (main.minposbin<100))
            logmin = Math.log(0.7);
         else
            logmin = (main.minposbin > 0) ? Math.log(0.7*main.minposbin) : logmax - 10;
         if (logmin >= logmax) logmin = logmax - 10;
         xyfactor = 1.0 / (logmax - logmin);
      } else
         xyfactor = 1.0 / (absmax - absmin);


      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            absz = Math.abs(binz);
            if ((absz === 0) || (absz < absmin)) continue;

            if (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5),
                 histo.fYaxis.GetBinCoord(j + 0.5))) continue;

            zdiff = uselogz ? ((absz > 0) ? Math.log(absz) - logmin : 0) : (absz - absmin);
            // area of the box should be proportional to absolute bin content
            zdiff = 0.5 * ((zdiff < 0) ? 1 : (1 - Math.sqrt(zdiff * xyfactor)));
            // avoid oversized bins
            if (zdiff < 0) zdiff = 0;

            ww = handle.grx[i+1] - handle.grx[i];
            hh = handle.gry[j] - handle.gry[j+1];

            dgrx = zdiff * ww;
            dgry = zdiff * hh;

            xx = Math.round(handle.grx[i] + dgrx);
            yy = Math.round(handle.gry[j+1] + dgry);

            ww = Math.max(Math.round(ww - 2*dgrx), 1);
            hh = Math.max(Math.round(hh - 2*dgry), 1);

            res += `M${xx},${yy}v${hh}h${ww}v${-hh}z`;

            if ((binz < 0) && (this.options.BoxStyle === 10))
               cross += `M${xx},${yy}l${ww},${hh}m0,${-hh}l${-ww},${hh}`;

            if ((this.options.BoxStyle === 11) && (ww > 5) && (hh > 5)) {
               const pww = Math.round(ww*0.1),
                     phh = Math.round(hh*0.1),
                     side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
                     side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;
               btn1 += (binz < 0) ? side2 : side1;
               btn2 += (binz < 0) ? side1 : side2;
            }
         }
      }

      if (res) {
         const elem = this.draw_g.append('svg:path')
                               .attr('d', res)
                               .call(this.fillatt.func);
         if ((this.options.BoxStyle !== 11) && this.fillatt.empty())
            elem.call(this.lineatt.func);
      }

      if (btn1 && this.fillatt.hasColor()) {
         this.draw_g.append('svg:path')
                    .attr('d', btn1)
                    .call(this.fillatt.func)
                    .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatHex());
      }

      if (btn2) {
         this.draw_g.append('svg:path')
                    .attr('d', btn2)
                    .call(this.fillatt.func)
                    .style('fill', !this.fillatt.hasColor() ? 'red' : d3_rgb(this.fillatt.color).darker(0.5).formatHex());
      }

      if (cross) {
         const elem = this.draw_g.append('svg:path')
                               .attr('d', cross)
                               .style('fill', 'none');
         if (!this.lineatt.empty())
            elem.call(this.lineatt.func);
         else
            elem.style('stroke', 'black');
      }

      return handle;
   }

   /** @summary Draw histogram bins as candle plot */
   drawBinsCandle() {
      const kNoOption = 0,
            kBox = 1,
            kMedianLine = 10,
            kMedianNotched = 20,
            kMedianCircle = 30,
            kMeanLine = 100,
            kMeanCircle = 300,
            kWhiskerAll = 1000,
            kWhisker15 = 2000,
            kAnchor = 10000,
            kPointsOutliers = 100000,
            kPointsAll = 200000,
            kPointsAllScat = 300000,
            kHistoLeft = 1000000,
            kHistoRight = 2000000,
            kHistoViolin = 3000000,
            kHistoZeroIndicator = 10000000,
            kHorizontal = 100000000,
            fallbackCandle = kBox + kMedianLine + kMeanCircle + kWhiskerAll + kAnchor,
            fallbackViolin = kMeanCircle + kWhiskerAll + kHistoViolin + kHistoZeroIndicator;

      let fOption = kNoOption;

      const isOption = opt => {
         let mult = 1;
         while (opt >= mult) mult *= 10;
         mult /= 10;
         return Math.floor(fOption/mult) % 10 === Math.floor(opt/mult);
      }, parseOption = (opt, is_candle) => {
         let direction = '', preset = '', res = kNoOption;
         const c0 = opt[0], c1 = opt[1];

         if (c0 >= 'A' && c0 <= 'Z') direction = c0;
         if (c0 >= '1' && c0 <= '9') preset = c0;
         if (c1 >= 'A' && c1 <= 'Z' && preset) direction = c1;
         if (c1 >= '1' && c1 <= '9' && direction) preset = c1;

         if (is_candle) {
            switch (preset) {
               case '1': res += fallbackCandle; break;
               case '2': res += kBox + kMeanLine + kMedianLine + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '3': res += kBox + kMeanCircle + kMedianLine + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '4': res += kBox + kMeanCircle + kMedianNotched + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '5': res += kBox + kMeanLine + kMedianLine + kWhisker15 + kAnchor + kPointsAll; break;
               case '6': res += kBox + kMeanCircle + kMedianLine + kWhisker15 + kAnchor + kPointsAllScat; break;
               default: res += fallbackCandle;
            }
         } else {
            switch (preset) {
               case '1': res += fallbackViolin; break;
               case '2': res += kMeanCircle + kWhisker15 + kHistoViolin + kHistoZeroIndicator + kPointsOutliers; break;
               default: res += fallbackViolin;
            }
         }

         const l = opt.indexOf('('), r = opt.lastIndexOf(')');
         if ((l >= 0) && (r > l+1))
            res = parseInt(opt.slice(l+1, r));

         fOption = res;

         if ((direction === 'Y' || direction === 'H') && !isOption(kHorizontal))
            fOption += kHorizontal;
      }, extractQuantiles = (xx, proj, prob) => {
         let integral = 0, cnt = 0, sum1 = 0;
         const res = { max: 0, first: -1, last: -1, entries: 0 };

         for (let j = 0; j < proj.length; ++j) {
            if (proj[j] > 0) {
               res.max = Math.max(res.max, proj[j]);
               if (res.first < 0) res.first = j;
               res.last = j;
            }
            integral += proj[j];
            sum1 += proj[j]*(xx[j]+xx[j+1])/2;
         }
         if (integral <= 0) return null;

         res.entries = integral;
         res.mean = sum1/integral;
         res.quantiles = new Array(prob.length);
         res.indx = new Array(prob.length);

         for (let j = 0, sum = 0, nextv = 0; j < proj.length; ++j) {
            const v = nextv;
            let x = xx[j];

            // special case - flat integral with const value
            if ((v === prob[cnt]) && (proj[j] === 0) && (v < 0.99)) {
               while ((proj[j] === 0) && (j < proj.length)) j++;
               x = (xx[j] + x) / 2; // this will be mid value
            }

            sum += proj[j];
            nextv = sum / integral;
            while ((prob[cnt] >= v) && (prob[cnt] < nextv)) {
               res.indx[cnt] = j;
               res.quantiles[cnt] = x + ((prob[cnt] - v) / (nextv - v)) * (xx[j + 1] - x);
               if (cnt++ === prob.length) return res;
               x = xx[j];
            }
         }

         while (cnt < prob.length) {
            res.indx[cnt] = proj.length-1;
            res.quantiles[cnt++] = xx[xx.length-1];
         }

         return res;
      };

      if (this.options.Candle)
         parseOption(this.options.Candle, true);
      else if (this.options.Violin)
         parseOption(this.options.Violin, false);

      const histo = this.getHisto(),
            handle = this.prepareDraw(),
            pmain = this.getFramePainter(), // used for axis values conversions
            cp = this.getCanvPainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            swapXY = isOption(kHorizontal);
      let bars = '', lines = '', dashed_lines = '',
          hists = '', hlines = '',
          markers = '', cmarkers = '', attrcmarkers = null,
          xx, proj,
          scaledViolin = gStyle.fViolinScaled,
          scaledCandle = gStyle.fCandleScaled,
          maxContent = 0, maxIntegral = 0;

      if (this.options.Scaled !== null)
         scaledViolin = scaledCandle = this.options.Scaled;
      else if (cp?.online_canvas) {
         // console.log('ignore hist title in online canvas');
      } else if (histo.fTitle.indexOf('unscaled') >= 0)
         scaledViolin = scaledCandle = false;
      else if (histo.fTitle.indexOf('scaled') >= 0)
         scaledViolin = scaledCandle = true;

      if (scaledViolin && (isOption(kHistoRight) || isOption(kHistoLeft) || isOption(kHistoViolin))) {
         for (let i = 0; i < this.nbinsx; ++i) {
            for (let j = 0; j < this.nbinsy; ++j)
               maxContent = Math.max(maxContent, histo.getBinContent(i + 1, j + 1));
          }
      }

      const make_path = (...a) => {
         if (a[1] === 'array') a = a[0];
         const l = a.length;
         let i = 2, xx = a[0], yy = a[1],
             res = swapXY ? `M${yy},${xx}` : `M${xx},${yy}`;
         while (i < l) {
            switch (a[i]) {
               case 'Z': return res + 'z';
               case 'V': if (yy !== a[i+1]) { res += (swapXY ? 'h' : 'v') + (a[i+1] - yy); yy = a[i+1]; } break;
               case 'H': if (xx !== a[i+1]) { res += (swapXY ? 'v' : 'h') + (a[i+1] - xx); xx = a[i+1]; } break;
               default: res += swapXY ? `l${a[i+1]-yy},${a[i]-xx}` : `l${a[i]-xx},${a[i+1]-yy}`; xx = a[i]; yy = a[i+1];
            }
            i += 2;
         }
         return res;
      }, make_marker = (x, y) => {
         if (!markers) {
            this.createAttMarker({ attr: histo, style: isOption(kPointsAllScat) ? 0 : 5 });
            this.markeratt.resetPos();
         }
         markers += swapXY ? this.markeratt.create(y, x) : this.markeratt.create(x, y);
      }, make_cmarker = (x, y) => {
         if (!attrcmarkers) {
            attrcmarkers = this.createAttMarker({ attr: histo, style: 24, std: false });
            attrcmarkers.resetPos();
         }
         cmarkers += swapXY ? attrcmarkers.create(y, x) : attrcmarkers.create(x, y);
      };

      // if ((histo.fFillStyle === 0) && (histo.fFillColor > 0) && (!this.fillatt || this.fillatt.empty()))
      //     this.createAttFill({ color: this.getColor(histo.fFillColor), pattern: 1001 });

      if (histo.fMarkerColor === 1) histo.fMarkerColor = histo.fLineColor;

      handle.candle = []; // array of drawn points

      // Determining the quantiles
      const wRange = gStyle.fCandleWhiskerRange, bRange = gStyle.fCandleBoxRange,
            prob = [(wRange >= 1) ? 1e-15 : 0.5 - wRange/2.0,
                     (bRange >= 1) ? 1E-14 : 0.5 - bRange/2.0,
                     0.5,
                     (bRange >= 1) ? 1-1E-14 : 0.5 + bRange/2.0,
                     (wRange >= 1) ? 1-1e-15 : 0.5 + wRange/2.0],

       produceCandlePoint = (bin_indx, grx_left, grx_right, xindx1, xindx2) => {
         const res = extractQuantiles(xx, proj, prob);
         if (!res) return;

         const pnt = { bin: bin_indx, swapXY, fBoxDown: res.quantiles[1], fMedian: res.quantiles[2], fBoxUp: res.quantiles[3] },
               iqr = pnt.fBoxUp - pnt.fBoxDown;
         let fWhiskerDown = res.quantiles[0], fWhiskerUp = res.quantiles[4];

         if (isOption(kWhisker15)) { // Improved whisker definition, with 1.5*iqr
            let pos = pnt.fBoxDown-1.5*iqr, indx = res.indx[1];
            while ((xx[indx] > pos) && (indx > 0)) indx--;
            while (!proj[indx]) indx++;
            fWhiskerDown = xx[indx]; // use lower edge here
            pos = pnt.fBoxUp+1.5*iqr; indx = res.indx[3];
            while ((xx[indx] < pos) && (indx < proj.length)) indx++;
            while (!proj[indx]) indx--;
            fWhiskerUp = xx[indx+1]; // use upper index edge here
         }

         const fMean = res.mean,
             fMedianErr = 1.57*iqr/Math.sqrt(res.entries);

         // estimate quantiles... simple function... not so nice as GetQuantiles

         // exclude points with negative y when log scale is specified
         if (fWhiskerDown <= 0)
           if ((swapXY && funcs.logx) || (!swapXY && funcs.logy)) return;

         const w = (grx_right - grx_left);
         let candleWidth, histoWidth,
             center = (grx_left + grx_right) / 2 + histo.fBarOffset/1000*w;
         if ((histo.fBarWidth > 0) && (histo.fBarWidth !== 1000))
            candleWidth = histoWidth = w * histo.fBarWidth / 1000;
          else {
            candleWidth = w*0.66;
            histoWidth = w*0.8;
         }

         if (scaledViolin && (maxContent > 0))
            histoWidth *= res.max/maxContent;
         if (scaledCandle && (maxIntegral > 0))
            candleWidth *= res.entries/maxIntegral;

         pnt.x1 = Math.round(center - candleWidth/2);
         pnt.x2 = Math.round(center + candleWidth/2);
         center = Math.round(center);

         const x1d = Math.round(center - candleWidth/3),
               x2d = Math.round(center + candleWidth/3),
               ff = swapXY ? funcs.grx : funcs.gry;

         pnt.yy1 = Math.round(ff(fWhiskerUp));
         pnt.y1 = Math.round(ff(pnt.fBoxUp));
         pnt.y0 = Math.round(ff(pnt.fMedian));
         pnt.y2 = Math.round(ff(pnt.fBoxDown));
         pnt.yy2 = Math.round(ff(fWhiskerDown));

         const y0m = Math.round(ff(fMean)),
               y01 = Math.round(ff(pnt.fMedian + fMedianErr)),
               y02 = Math.round(ff(pnt.fMedian - fMedianErr));

         if (isOption(kHistoZeroIndicator))
            hlines += make_path(center, Math.round(ff(xx[xindx1])), 'V', Math.round(ff(xx[xindx2])));

         if (isOption(kMedianLine))
            lines += make_path(pnt.x1, pnt.y0, 'H', pnt.x2);
         else if (isOption(kMedianNotched))
            lines += make_path(x1d, pnt.y0, 'H', x2d);
         else if (isOption(kMedianCircle))
            make_cmarker(center, pnt.y0);

         if (isOption(kMeanCircle))
            make_cmarker(center, y0m);
         else if (isOption(kMeanLine))
            dashed_lines += make_path(pnt.x1, y0m, 'H', pnt.x2);

         if (isOption(kBox)) {
            if (isOption(kMedianNotched))
               bars += make_path(pnt.x1, pnt.y1, 'V', y01, x1d, pnt.y0, pnt.x1, y02, 'V', pnt.y2, 'H', pnt.x2, 'V', y02, x2d, pnt.y0, pnt.x2, y01, 'V', pnt.y1, 'Z');
            else
               bars += make_path(pnt.x1, pnt.y1, 'V', pnt.y2, 'H', pnt.x2, 'V', pnt.y1, 'Z');
         }

        if (isOption(kAnchor))  // Draw the anchor line
            lines += make_path(pnt.x1, pnt.yy1, 'H', pnt.x2) + make_path(pnt.x1, pnt.yy2, 'H', pnt.x2);

         if (isOption(kWhiskerAll) && !isOption(kHistoZeroIndicator)) { // Whiskers are dashed
            dashed_lines += make_path(center, pnt.y1, 'V', pnt.yy1) + make_path(center, pnt.y2, 'V', pnt.yy2);
         } else if ((isOption(kWhiskerAll) && isOption(kHistoZeroIndicator)) || isOption(kWhisker15))
            lines += make_path(center, pnt.y1, 'V', pnt.yy1) + make_path(center, pnt.y2, 'V', pnt.yy2);


         if (isOption(kPointsOutliers) || isOption(kPointsAll) || isOption(kPointsAllScat)) {
            // reset seed for each projection to have always same pixels
            const rnd = new TRandom(bin_indx*7521 + Math.round(res.integral)),
                show_all = !isOption(kPointsOutliers),
                show_scat = isOption(kPointsAllScat);
            for (let ii = 0; ii < proj.length; ++ii) {
               const bin_content = proj[ii], binx = (xx[ii] + xx[ii+1])/2;
               let marker_x = center, marker_y = 0;

               if (!bin_content) continue;
               if (!show_all && (binx >= fWhiskerDown) && (binx <= fWhiskerUp)) continue;

               for (let k = 0; k < bin_content; k++) {
                  if (show_scat)
                     marker_x = center + Math.round(((rnd.random() - 0.5) * candleWidth));

                  if ((bin_content === 1) && !show_scat)
                     marker_y = Math.round(ff(binx));
                  else
                     marker_y = Math.round(ff(xx[ii] + rnd.random()*(xx[ii+1]-xx[ii])));

                  make_marker(marker_x, marker_y);
               }
            }
         }

         if ((isOption(kHistoRight) || isOption(kHistoLeft) || isOption(kHistoViolin)) && (res.max > 0) && (res.first >= 0)) {
            const arr = [], scale = (swapXY ? -0.5 : 0.5) *histoWidth/res.max;

            xindx1 = Math.max(xindx1, res.first);
            xindx2 = Math.min(xindx2-1, res.last);

            if (isOption(kHistoRight) || isOption(kHistoViolin)) {
               let prev_x = center, prev_y = Math.round(ff(xx[xindx1]));
               arr.push(prev_x, prev_y);
               for (let ii = xindx1; ii <= xindx2; ii++) {
                  const curr_x = Math.round(center + scale*proj[ii]),
                      curr_y = Math.round(ff(xx[ii+1]));
                  if (curr_x !== prev_x) {
                     if (ii !== xindx1) arr.push('V', prev_y);
                     arr.push('H', curr_x);
                  }
                  prev_x = curr_x;
                  prev_y = curr_y;
               }
               arr.push('V', prev_y);
            }

            if (isOption(kHistoLeft) || isOption(kHistoViolin)) {
               let prev_x = center, prev_y = Math.round(ff(xx[xindx2+1]));
               if (arr.length === 0)
                  arr.push(prev_x, prev_y);
               for (let ii = xindx2; ii >= xindx1; ii--) {
                  const curr_x = Math.round(center - scale*proj[ii]),
                      curr_y = Math.round(ff(xx[ii]));
                  if (curr_x !== prev_x) {
                     if (ii !== xindx2) arr.push('V', prev_y);
                     arr.push('H', curr_x);
                  }
                  prev_x = curr_x;
                  prev_y = curr_y;
               }
               arr.push('V', prev_y);
            }

            arr.push('H', center); // complete histogram

            hists += make_path(arr, 'array');

            if (!this.fillatt.empty()) hists += 'Z';
         }

         handle.candle.push(pnt); // keep point for the tooltip
      };

      if (swapXY) {
         xx = new Array(this.nbinsx+1);
         proj = new Array(this.nbinsx);
         for (let i = 0; i < this.nbinsx+1; ++i)
            xx[i] = histo.fXaxis.GetBinLowEdge(i+1);

         if (scaledCandle) {
            for (let j = 0; j < this.nbinsy; ++j) {
               let sum = 0;
               for (let i = 0; i < this.nbinsx; ++i)
                  sum += histo.getBinContent(i+1, j+1);
               maxIntegral = Math.max(maxIntegral, sum);
            }
         }

         for (let j = handle.j1; j < handle.j2; ++j) {
            for (let i = 0; i < this.nbinsx; ++i)
               proj[i] = histo.getBinContent(i+1, j+1);

            produceCandlePoint(j, handle.gry[j+1], handle.gry[j], handle.i1, handle.i2);
         }
      } else {
         xx = new Array(this.nbinsy+1);
         proj = new Array(this.nbinsy);

         for (let j = 0; j < this.nbinsy+1; ++j)
            xx[j] = histo.fYaxis.GetBinLowEdge(j+1);

         if (scaledCandle) {
            for (let i = 0; i < this.nbinsx; ++i) {
               let sum = 0;
               for (let j = 0; j < this.nbinsy; ++j)
                  sum += histo.getBinContent(i+1, j+1);
               maxIntegral = Math.max(maxIntegral, sum);
            }
         }

         // loop over visible x-bins
         for (let i = handle.i1; i < handle.i2; ++i) {
            for (let j = 0; j < this.nbinsy; ++j)
               proj[j] = histo.getBinContent(i+1, j+1);

            produceCandlePoint(i, handle.grx[i], handle.grx[i+1], handle.j1, handle.j2);
         }
      }

      if (hlines && (histo.fFillColor > 0)) {
         this.draw_g.append('svg:path')
             .attr('d', hlines)
             .style('stroke', this.getColor(histo.fFillColor));
      }

      const hline_color = (isOption(kHistoZeroIndicator) && (histo.fFillStyle !== 0)) ? this.fillatt.color : this.lineatt.color;
      if (hists && (!this.fillatt.empty() || (hline_color !== 'none'))) {
         this.draw_g.append('svg:path')
             .attr('d', hists)
             .style('stroke', (hline_color !== 'none') ? hline_color : null)
             .style('pointer-events', this.isBatchMode() ? null : 'visibleFill')
             .call(this.fillatt.func);
      }

      if (bars) {
         this.draw_g.append('svg:path')
             .attr('d', bars)
             .call(this.lineatt.func)
             .call(this.fillatt.func);
      }

      if (lines) {
         this.draw_g.append('svg:path')
             .attr('d', lines)
             .call(this.lineatt.func)
             .style('fill', 'none');
      }

      if (dashed_lines) {
         const dashed = this.createAttLine({ attr: histo, style: 2, std: false, color: kBlack });
         this.draw_g.append('svg:path')
             .attr('d', dashed_lines)
             .call(dashed.func)
             .style('fill', 'none');
      }

      if (cmarkers) {
         this.draw_g.append('svg:path')
             .attr('d', cmarkers)
             .call(attrcmarkers.func);
      }

      if (markers) {
         this.draw_g.append('svg:path')
             .attr('d', markers)
             .call(this.markeratt.func);
      }

      return handle;
   }

   /** @summary Draw TH2 bins as scatter plot */
   drawBinsScatter() {
      const histo = this.getObject(),
            handle = this.prepareDraw({ rounding: true, pixel_density: true }),
            test_cutg = this.options.cutg,
            colPaths = [], currx = [], curry = [], cell_w = [], cell_h = [],
            scale = this.options.ScatCoef * ((this.gmaxbin) > 2000 ? 2000 / this.gmaxbin : 1),
            rnd = new TRandom(handle.sumz);
      let colindx, cmd1, cmd2, i, j, binz, cw, ch, factor = 1.0;

      handle.ScatterPlot = true;

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         this.createAttMarker({ attr: histo });

         this.markeratt.resetPos();

         let path = '';
         for (i = handle.i1; i < handle.i2; ++i) {
            cw = handle.grx[i+1] - handle.grx[i];
            for (j = handle.j1; j < handle.j2; ++j) {
               ch = handle.gry[j] - handle.gry[j+1];
               binz = histo.getBinContent(i + 1, j + 1);

               const npix = Math.round(scale*binz);
               if (npix <= 0) continue;

               if (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5),
                     histo.fYaxis.GetBinCoord(j + 0.5))) continue;

               for (let k = 0; k < npix; ++k) {
                  path += this.markeratt.create(
                            Math.round(handle.grx[i] + cw * rnd.random()),
                            Math.round(handle.gry[j+1] + ch * rnd.random()));
               }
            }
         }

         this.draw_g
              .append('svg:path')
              .attr('d', path)
              .call(this.markeratt.func);

         return handle;
      }

      // limit filling factor, do not try to produce as many points as filled area;
      if (this.maxbin > 0.7) factor = 0.7/this.maxbin;

      const nlevels = Math.round(handle.max - handle.min),
          cntr = this.createContour((nlevels > 50) ? 50 : nlevels, this.minposbin, this.maxbin, this.minposbin);

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            if ((binz <= 0) || (binz < this.minbin)) continue;

            cw = handle.grx[i+1] - handle.grx[i];
            ch = handle.gry[j] - handle.gry[j+1];
            if (cw*ch <= 0) continue;

            colindx = cntr.getContourIndex(binz/cw/ch);
            if (colindx < 0) continue;

            if (test_cutg && !test_cutg.IsInside(histo.fXaxis.GetBinCoord(i + 0.5),
                     histo.fYaxis.GetBinCoord(j + 0.5))) continue;

            cmd1 = `M${handle.grx[i]},${handle.gry[j+1]}`;
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
               cell_w[colindx] = cw;
               cell_h[colindx] = ch;
            } else {
               cmd2 = `m${handle.grx[i]-currx[colindx]},${handle.gry[j+1]-curry[colindx]}`;
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               cell_w[colindx] = Math.max(cell_w[colindx], cw);
               cell_h[colindx] = Math.max(cell_h[colindx], ch);
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+1];

            colPaths[colindx] += `v${ch}h${cw}v${-ch}z`;
         }
      }

      const layer = this.getFrameSvg().selectChild('.main_layer');
      let defs = layer.selectChild('defs');
      if (defs.empty() && (colPaths.length > 0))
         defs = layer.insert('svg:defs', ':first-child');

      this.createAttMarker({ attr: histo });

      for (colindx = 0; colindx < colPaths.length; ++colindx) {
         if ((colPaths[colindx] !== undefined) && (colindx < cntr.arr.length)) {
           const pattern_id = (this.pad_name || 'canv') + `_scatter_${colindx}`;
           let pattern = defs.selectChild(`#${pattern_id}`);
           if (pattern.empty()) {
              pattern = defs.append('svg:pattern')
                            .attr('id', pattern_id)
                            .attr('patternUnits', 'userSpaceOnUse');
           } else
              pattern.selectAll('*').remove();

           let npix = Math.round(factor*cntr.arr[colindx]*cell_w[colindx]*cell_h[colindx]);
           if (npix < 1) npix = 1;

           const arrx = new Float32Array(npix), arry = new Float32Array(npix);

           if (npix === 1)
              arrx[0] = arry[0] = 0.5;
            else {
              for (let n = 0; n < npix; ++n) {
                 arrx[n] = rnd.random();
                 arry[n] = rnd.random();
              }
           }

           this.markeratt.resetPos();

           let path = '';

           for (let n = 0; n < npix; ++n)
              path += this.markeratt.create(arrx[n] * cell_w[colindx], arry[n] * cell_h[colindx]);

           pattern.attr('width', cell_w[colindx])
                  .attr('height', cell_h[colindx])
                  .append('svg:path')
                  .attr('d', path)
                  .call(this.markeratt.func);

           this.draw_g
               .append('svg:path')
               .attr('scatter-index', colindx)
               .style('fill', `url(#${pattern_id})`)
               .attr('d', colPaths[colindx]);
        }
      }

      return handle;
   }

   /** @summary Draw TH2 bins in 2D mode */
   draw2DBins() {
      if (this._hide_frame && this.isMainPainter()) {
         this.getFrameSvg().style('display', null);
         delete this._hide_frame;
      }

      if (!this.draw_content)
         return this.removeG();

      this.createHistDrawAttributes();

      this.createG(true);

      let handle, pr;

      if (this.isTH2Poly())
         pr = this.drawPolyBins();
       else {
         if (this.options.Scat)
            handle = this.drawBinsScatter();

         if (this.options.Color)
            handle = this.drawBinsColor();
         else if (this.options.Box)
            handle = this.drawBinsBox();
         else if (this.options.Arrow)
            handle = this.drawBinsArrow();
         else if (this.options.Proj)
            handle = this.drawBinsProjected();
         else if (this.options.Contour)
            handle = this.drawBinsContour();
         else if (this.options.Candle || this.options.Violin)
            handle = this.drawBinsCandle();

         if (this.options.Text)
            pr = this.drawBinsText(handle);

         if (!handle && !pr)
            handle = this.drawBinsColor();
      }

      if (handle)
         this.tt_handle = handle;
      else if (pr)
         return pr.then(tt => { this.tt_handle = tt; });
   }

   /** @summary Draw TH2 in circular mode */
   drawBinsCircular() {
      this.getFrameSvg().style('display', 'none');
      this._hide_frame = true;

      const rect = this.getPadPainter().getFrameRect(),
            hist = this.getHisto(),
            palette = this.options.Circular > 10 ? this.getHistPalette() : null,
            text_size = 20,
            circle_size = 16,
            axis = hist.fXaxis,
            getBinLabel = indx => {
               if (axis.fLabels) {
                  for (let i = 0; i < axis.fLabels.arr.length; ++i) {
                     const tstr = axis.fLabels.arr[i];
                     if (tstr.fUniqueID === indx+1) return tstr.fString;
                  }
               }
               return indx.toString();
            };

      this.createG();

      makeTranslate(this.draw_g, Math.round(rect.x + rect.width/2), Math.round(rect.y + rect.height/2));

      const nbins = Math.min(this.nbinsx, this.nbinsy);

      this.startTextDrawing(42, text_size, this.draw_g);

      const pnts = [];

      for (let n = 0; n < nbins; n++) {
         const a = (0.5 - n/nbins)*Math.PI*2,
               cx = Math.round((0.9*rect.width/2 - 2*circle_size) * Math.cos(a)),
               cy = Math.round((0.9*rect.height/2 - 2*circle_size) * Math.sin(a)),
               x = Math.round(0.9*rect.width/2 * Math.cos(a)),
               y = Math.round(0.9*rect.height/2 * Math.sin(a)),
               color = palette?.calcColor(n, nbins) ?? 'black';
         let rotate = Math.round(a/Math.PI*180), align = 12;

         pnts.push({ x: cx, y: cy, a, color }); // remember points coordinates

         if ((rotate < -90) || (rotate > 90)) { rotate += 180; align = 32; }

         const s2 = Math.round(text_size/2), s1 = 2*s2;

         this.draw_g.append('path')
                    .attr('d', `M${cx-s2},${cy} a${s2},${s2},0,1,0,${s1},0a${s2},${s2},0,1,0,${-s1},0z`)
                    .style('stroke', color)
                    .style('fill', 'none');

         this.drawText({ align, rotate, x, y, text: getBinLabel(n) });
      }

      const max_width = circle_size/2;
      let max_value = 0, min_value = 0;
      if (this.options.Circular > 11) {
         for (let i = 0; i < nbins - 1; ++i) {
            for (let j = i+1; j < nbins; ++j) {
              const cont = hist.getBinContent(i+1, j+1);
              if (cont > 0) {
                 max_value = Math.max(max_value, cont);
                 if (!min_value || (cont < min_value)) min_value = cont;
              }
            }
         }
      }

      for (let i = 0; i < nbins-1; ++i) {
         const pi = pnts[i];
         let path = '';

         for (let j = i+1; j < nbins; ++j) {
            const cont = hist.getBinContent(i+1, j+1);
            if (cont <= 0) continue;

            const pj = pnts[j],
                a = (pi.a + pj.a)/2,
                qr = 0.5*(1-Math.abs(pi.a - pj.a)/Math.PI), // how far Q point will be away from center
                qx = Math.round(qr*rect.width/2 * Math.cos(a)),
                qy = Math.round(qr*rect.height/2 * Math.sin(a));

            path += `M${pi.x},${pi.y}Q${qx},${qy},${pj.x},${pj.y}`;

            if ((this.options.Circular > 11) && (max_value > min_value)) {
               const width = Math.round((cont - min_value) / (max_value - min_value) * (max_width - 1) + 1);
               this.draw_g.append('path').attr('d', path).style('stroke', pi.color).style('stroke-width', width).style('fill', 'none');
               path = '';
            }
         }
         if (path)
            this.draw_g.append('path').attr('d', path).style('stroke', pi.color).style('fill', 'none');
      }

      return this.finishTextDrawing();
   }

   /** @summary Draw histogram bins as chord diagram */
   async drawBinsChord() {
      this.getFrameSvg().style('display', 'none');
      this._hide_frame = true;

      const used = [],
            nbins = Math.min(this.nbinsx, this.nbinsy),
            hist = this.getHisto();
      let fullsum = 0, isint = true;
      for (let i = 0; i < nbins; ++i) {
         let sum = 0;
         for (let j = 0; j < nbins; ++j) {
            const cont = hist.getBinContent(i+1, j+1);
            if (cont > 0) {
               sum += cont;
               if (isint && (Math.round(cont) !== cont)) isint = false;
            }
         }
         if (sum > 0) used.push(i);
         fullsum += sum;
      }

      // do not show less than 2 elements
      if (used.length < 2) return true;

      let ndig = 0, tickStep = 1;
      const rect = this.getPadPainter().getFrameRect(),
            palette = this.getHistPalette(),
            outerRadius = Math.max(10, Math.min(rect.width, rect.height) * 0.5 - 60),
            innerRadius = Math.max(2, outerRadius - 10),
            data = [], labels = [],
            getColor = indx => palette.calcColor(indx, used.length),
            formatValue = v => v.toString(),
            formatTicks = v => ndig > 3 ? v.toExponential(0) : v.toFixed(ndig),
            d3_descending = (a, b) => { return b < a ? -1 : b > a ? 1 : b >= a ? 0 : Number.NaN; };

      if (!isint && fullsum < 10) {
         const lstep = Math.round(Math.log10(fullsum) - 2.3);
         ndig = -lstep;
         tickStep = Math.pow(10, lstep);
      } else if (fullsum > 200) {
         const lstep = Math.round(Math.log10(fullsum) - 2.3);
         tickStep = Math.pow(10, lstep);
      }

      if (tickStep * 250 < fullsum)
         tickStep *= 5;
      else if (tickStep * 100 < fullsum)
         tickStep *= 2;

      for (let i = 0; i < used.length; ++i) {
         data[i] = [];
         for (let j = 0; j < used.length; ++j)
            data[i].push(hist.getBinContent(used[i]+1, used[j]+1));
         const axis = hist.fXaxis;
         let lbl = 'indx_' + used[i].toString();
         if (axis.fLabels) {
            for (let k = 0; k < axis.fLabels.arr.length; ++k) {
               const tstr = axis.fLabels.arr[k];
               if (tstr.fUniqueID === used[i]+1) { lbl = tstr.fString; break; }
            }
         }
         labels.push(lbl);
      }

      this.createG();

      makeTranslate(this.draw_g, Math.round(rect.x + rect.width/2), Math.round(rect.y + rect.height/2));

      const chord = d3_chord()
         .padAngle(10 / innerRadius)
         .sortSubgroups(d3_descending)
         .sortChords(d3_descending),

      chords = chord(data),

      group = this.draw_g.append('g')
         .attr('font-size', 10)
         .attr('font-family', 'sans-serif')
         .selectAll('g')
         .data(chords.groups)
         .join('g'),

      arc = d3_arc().innerRadius(innerRadius).outerRadius(outerRadius),

      ribbon = d3_ribbon().radius(innerRadius - 1).padAngle(1 / innerRadius);

      function ticks({ startAngle, endAngle, value }) {
         const k = (endAngle - startAngle) / value,
          arr = [];
         for (let z = 0; z <= value; z += tickStep)
            arr.push({ value: z, angle: z * k + startAngle });
         return arr;
      }

      group.append('path')
         .attr('fill', d => getColor(d.index))
         .attr('d', arc);

      group.append('title').text(d => `${labels[d.index]} ${formatValue(d.value)}`);

      const groupTick = group.append('g')
         .selectAll('g')
         .data(ticks)
         .join('g')
         .attr('transform', d => `rotate(${Math.round(d.angle*180/Math.PI-90)}) translate(${outerRadius})`);
      groupTick.append('line')
         .attr('stroke', 'currentColor')
         .attr('x2', 6);

      groupTick.append('text')
         .attr('x', 8)
         .attr('dy', '0.35em')
         .attr('transform', d => d.angle > Math.PI ? 'rotate(180) translate(-16)' : null)
         .attr('text-anchor', d => d.angle > Math.PI ? 'end' : null)
         .text(d => formatTicks(d.value));

      group.select('text')
         .attr('font-weight', 'bold')
         .text(function(d) {
            return this.getAttribute('text-anchor') === 'end' ? `↑ ${labels[d.index]}` : `${labels[d.index]} ↓`;
         });

      this.draw_g.append('g')
         .attr('fill-opacity', 0.8)
         .selectAll('path')
         .data(chords)
         .join('path')
         .style('mix-blend-mode', 'multiply')
         .attr('fill', d => getColor(d.source.index))
         .attr('d', ribbon)
         .append('title')
         .text(d => `${formatValue(d.source.value)} ${labels[d.target.index]} → ${labels[d.source.index]}${d.source.index === d.target.index ? '' : `\n${formatValue(d.target.value)} ${labels[d.source.index]} → ${labels[d.target.index]}`}`);

      return true;
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(i, j) {
      const histo = this.getHisto();
      let binz = histo.getBinContent(i+1, j+1);

      if (histo.$baseh)
         binz -= histo.$baseh.getBinContent(i+1, j+1);

      const lines = [this.getObjectHint(),
                   'x = ' + this.getAxisBinTip('x', histo.fXaxis, i),
                   'y = ' + this.getAxisBinTip('y', histo.fYaxis, j),
                   `bin = ${histo.getBin(i+1, j+1)}  x: ${i+1}  y: ${j+1}`,
                   'entries = ' + ((binz === Math.round(binz)) ? binz : floatToString(binz, gStyle.fStatFormat))];

      if ((this.options.TextKind === 'E') || this.matchObjectType(clTProfile2D)) {
         const errz = histo.getBinError(histo.getBin(i+1, j+1));
         lines.push('error = ' + ((errz === Math.round(errz)) ? errz.toString() : floatToString(errz, gStyle.fPaintTextFormat)));
      }

      return lines;
   }

   /** @summary Provide text information (tooltips) for candle bin */
   getCandleTooltips(p) {
      const pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            histo = this.getHisto();

      return [this.getObjectHint(),
              p.swapXY
                 ? 'y = ' + funcs.axisAsText('y', histo.fYaxis.GetBinLowEdge(p.bin+1))
                 : 'x = ' + funcs.axisAsText('x', histo.fXaxis.GetBinLowEdge(p.bin+1)),
              'm-25%  = ' + floatToString(p.fBoxDown, gStyle.fStatFormat),
              'median = ' + floatToString(p.fMedian, gStyle.fStatFormat),
              'm+25%  = ' + floatToString(p.fBoxUp, gStyle.fStatFormat)];
   }

   /** @summary Provide text information (tooltips) for poly bin */
   getPolyBinTooltips(binindx, realx, realy) {
      const histo = this.getHisto(),
            bin = histo.fBins.arr[binindx],
            pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            lines = [];
      let binname = bin.fPoly.fName, numpoints = 0;

      if (binname === 'Graph') binname = '';
      if (binname.length === 0) binname = bin.fNumber;

      if ((realx === undefined) && (realy === undefined)) {
         realx = realy = 0;
         let gr = bin.fPoly, numgraphs = 1;
         if (gr._typename === clTMultiGraph) { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

         for (let ngr = 0; ngr < numgraphs; ++ngr) {
            if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];

            for (let n = 0; n < gr.fNpoints; ++n) {
               ++numpoints;
               realx += gr.fX[n];
               realy += gr.fY[n];
            }
         }

         if (numpoints > 1) {
            realx = realx / numpoints;
            realy = realy / numpoints;
         }
      }

      lines.push(this.getObjectHint(),
                 'x = ' + funcs.axisAsText('x', realx),
                 'y = ' + funcs.axisAsText('y', realy));
      if (numpoints > 0)
         lines.push('npnts = ' + numpoints);
      lines.push(`bin = ${binname}`);
      if (bin.fContent === Math.round(bin.fContent))
         lines.push('content = ' + bin.fContent);
      else
         lines.push('content = ' + floatToString(bin.fContent, gStyle.fStatFormat));
      return lines;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      const histo = this.getHisto(),
            h = this.tt_handle;
      let ttrect = this.draw_g?.selectChild('.tooltip_bin');

      if (!pnt || !this.draw_content || !this.draw_g || !h || this.options.Proj) {
         ttrect?.remove();
         return null;
      }

      if (h.poly) {
         // process tooltips from TH2Poly

         const pmain = this.getFramePainter(),
               funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
               realx = funcs.revertAxis('x', pnt.x),
               realy = funcs.revertAxis('y', pnt.y);
         let foundindx = -1, bin;

         if ((realx !== undefined) && (realy !== undefined)) {
            const len = histo.fBins.arr.length;

            for (let i = 0; (i < len) && (foundindx < 0); ++i) {
               bin = histo.fBins.arr[i];

               // found potential bins candidate
               if ((realx < bin.fXmin) || (realx > bin.fXmax) ||
                   (realy < bin.fYmin) || (realy > bin.fYmax)) continue;

               // ignore empty bins with col0 option
               if (!bin.fContent && !this.options.Zero) continue;

               let gr = bin.fPoly, numgraphs = 1;
               if (gr._typename === clTMultiGraph) { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

               for (let ngr = 0; ngr < numgraphs; ++ngr) {
                  if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];
                  if (gr.IsInside(realx, realy)) {
                     foundindx = i;
                     break;
                  }
               }
            }
         }

         if (foundindx < 0) {
            ttrect.remove();
            return null;
         }

         const res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt?.color ?? 'green',
                     color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                     exact: true, menu: true,
                     lines: this.getPolyBinTooltips(foundindx, realx, realy) };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {
            if (ttrect.empty()) {
               ttrect = this.draw_g.append('svg:path')
                            .attr('class', 'tooltip_bin')
                            .style('pointer-events', 'none')
                            .call(addHighlightStyle);
            }

            res.changed = ttrect.property('current_bin') !== foundindx;

            if (res.changed) {
               ttrect.attr('d', this.createPolyBin(funcs, bin))
                     .style('opacity', '0.7')
                     .property('current_bin', foundindx);
            }
         }

         if (res.changed) {
            res.user_info = { obj: histo, name: histo.fName,
                              bin: foundindx,
                              cont: bin.fContent,
                              grx: pnt.x, gry: pnt.y };
         }
         return res;
      } else if (h.candle) {
         // process tooltips for candle

         let i, p, match;

         for (i = 0; i < h.candle.length; ++i) {
            p = h.candle[i];
            match = p.swapXY
                      ? ((p.x1 <= pnt.y) && (pnt.y <= p.x2) && (p.yy1 >= pnt.x) && (pnt.x >= p.yy2))
                      : ((p.x1 <= pnt.x) && (pnt.x <= p.x2) && (p.yy1 <= pnt.y) && (pnt.y <= p.yy2));
            if (match) break;
         }

         if (!match) {
            ttrect.remove();
            return null;
         }

         const res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt?.color ?? 'green',
                     color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                     lines: this.getCandleTooltips(p), exact: true, menu: true };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {
            if (ttrect.empty()) {
               ttrect = this.draw_g.append('svg:path')
                                   .attr('class', 'tooltip_bin')
                                   .style('pointer-events', 'none')
                                   .call(addHighlightStyle)
                                   .style('opacity', '0.7');
            }

            res.changed = ttrect.property('current_bin') !== i;

            if (res.changed) {
               ttrect.attr('d', p.swapXY ? `M${p.yy1},${p.x1}H${p.yy2}V${p.x2}H${p.yy1}Z` : `M${p.x1},${p.yy1}H${p.x2}V${p.yy2}H${p.x1}Z`)
                     .property('current_bin', i);
            }
         }

         if (res.changed) {
            res.user_info = { obj: histo, name: histo.fName,
                              bin: i+1, cont: p.fMedian, binx: i+1, biny: 1,
                              grx: pnt.x, gry: pnt.y };
         }

         return res;
      }

      const pmain = this.getFramePainter();
      let i, j, binz = 0, colindx = null,
          i1, i2, j1, j2, x1, x2, y1, y2;

      // search bins position
      if (pmain.reverse_x) {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x <= h.grx[i]) && (pnt.x >= h.grx[i+1])) break;
      } else {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x >= h.grx[i]) && (pnt.x <= h.grx[i+1])) break;
      }

      if (pmain.reverse_y) {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y <= h.gry[j+1]) && (pnt.y >= h.gry[j])) break;
      } else {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y >= h.gry[j+1]) && (pnt.y <= h.gry[j])) break;
      }

      if ((i < h.i2) && (j < h.j2)) {
         i1 = i; i2 = i+1; j1 = j; j2 = j+1;
         x1 = h.grx[i1]; x2 = h.grx[i2];
         y1 = h.gry[j2]; y2 = h.gry[j1];

         let match = true;

         if (this.options.Color) {
            // take into account bar settings
            const dx = x2 - x1, dy = y2 - y1;
            x2 = Math.round(x1 + dx*h.xbar2);
            x1 = Math.round(x1 + dx*h.xbar1);
            y2 = Math.round(y1 + dy*h.ybar2);
            y1 = Math.round(y1 + dy*h.ybar1);
            if (pmain.reverse_x) {
               if ((pnt.x > x1) || (pnt.x <= x2)) match = false;
            } else
               if ((pnt.x < x1) || (pnt.x >= x2)) match = false;

            if (pmain.reverse_y) {
               if ((pnt.y > y1) || (pnt.y <= y2)) match = false;
            } else
               if ((pnt.y < y1) || (pnt.y >= y2)) match = false;
         }

         binz = histo.getBinContent(i+1, j+1);
         if (this.is_projection)
            colindx = 0; // just to avoid hide
          else if (!match)
            colindx = null;
          else if (h.hide_only_zeros)
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
          else {
            colindx = this.getContour().getPaletteIndex(this.getHistPalette(), binz);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         return null;
      }

      const res = { name: histo.fName, title: histo.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt?.color ?? 'green',
                  color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                  lines: this.getBinTooltips(i, j), exact: true, menu: true };

      if (this.options.Color) res.color2 = this.getHistPalette().getColor(colindx);

      if (pnt.disabled && !this.is_projection) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:path')
                                .attr('class', 'tooltip_bin')
                                .style('pointer-events', 'none')
                                .call(addHighlightStyle);
         }

         let binid = i*10000 + j, path;

         if (this.is_projection) {
            const pwx = this.projection_widthX || 1, ddx = (pwx - 1) / 2;
            if ((this.is_projection.indexOf('X')) >= 0 && (pwx > 1)) {
               if (j2+ddx >= h.j2) {
                  j2 = Math.min(Math.round(j2+ddx), h.j2);
                  j1 = Math.max(j2-pwx, h.j1);
               } else {
                  j1 = Math.max(Math.round(j1-ddx), h.j1);
                  j2 = Math.min(j1+pwx, h.j2);
               }
            }
            const pwy = this.projection_widthY || 1, ddy = (pwy - 1) / 2;
            if ((this.is_projection.indexOf('Y')) >= 0 && (pwy > 1)) {
               if (i2+ddy >= h.i2) {
                  i2 = Math.min(Math.round(i2+ddy), h.i2);
                  i1 = Math.max(i2-pwy, h.i1);
               } else {
                  i1 = Math.max(Math.round(i1-ddy), h.i1);
                  i2 = Math.min(i1+pwy, h.i2);
               }
            }
         }

         if (this.is_projection === 'X') {
            x1 = 0; x2 = pmain.getFrameWidth();
            y1 = h.gry[j2]; y2 = h.gry[j1];
            binid = j1*777 + j2*333;
         } else if (this.is_projection === 'Y') {
            y1 = 0; y2 = pmain.getFrameHeight();
            x1 = h.grx[i1]; x2 = h.grx[i2];
            binid = i1*777 + i2*333;
         } else if (this.is_projection === 'XY') {
            y1 = h.gry[j2]; y2 = h.gry[j1];
            x1 = h.grx[i1]; x2 = h.grx[i2];
            binid = i1*789 + i2*653 + j1*12345 + j2*654321;
            path = `M${x1},0H${x2}V${y1}H${pmain.getFrameWidth()}V${y2}H${x2}V${pmain.getFrameHeight()}H${x1}V${y2}H0V${y1}H${x1}Z`;
         }

         res.changed = ttrect.property('current_bin') !== binid;

         if (res.changed) {
            ttrect.attr('d', path || `M${x1},${y1}H${x2}V${y2}H${x1}Z`)
                  .style('opacity', '0.7')
                  .property('current_bin', binid);
         }

         if (this.is_projection && res.changed)
            this.redrawProjection(i1, i2, j1, j2);
      }

      if (res.changed) {
         res.user_info = { obj: histo, name: histo.fName,
                           bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                           grx: pnt.x, gry: pnt.y };
      }

      return res;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      if ((axis === 'z') || this.options.Proj)
         return true;

      let obj = this.getHisto();
      if (obj) obj = (axis === 'y') ? obj.fYaxis : obj.fXaxis;

      return !obj || (obj.FindBin(max, 0.5) - obj.FindBin(min, 0) > 1);
   }

   /** @summary Complete paletted drawing */
   completePalette(pp) {
      if (!pp) return true;

      pp.$main_painter = this;
      this.options.Zvert = pp._palette_vertical;

      // redraw palette till the end when contours are available
      return pp.drawPave(this.options.Cjust ? 'cjust' : '');
   }

   /** @summary Performs 2D drawing of histogram
     * @return {Promise} when ready */
   async draw2D(/* reason */) {
      this.clear3DScene();

      const need_palette = this.options.Zscale && this.options.canHavePalette();

      // draw new palette, resize frame if required
      return this.drawColorPalette(need_palette, true).then(async pp => {
         let pr;
         if (this.options.Circular && this.isMainPainter())
            pr = this.drawBinsCircular();
          else if (this.options.Chord && this.isMainPainter())
            pr = this.drawBinsChord();
          else
            pr = this.drawAxes().then(() => this.draw2DBins());

         return pr.then(() => this.completePalette(pp));
      }).then(() => this.updateFunctions())
        .then(() => this.updateHistTitle())
        .then(() => {
            this.updateStatWebCanvas();
            return this.addInteractivity();
      });
   }

   /** @summary Should performs 3D drawing of histogram
     * @desc Disabled in 2D case. just draw default draw options
     * @return {Promise} when ready */
   async draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/TH2Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Call drawing function depending from 3D mode */
   async callDrawFunc(reason) {
      const main = this.getMainPainter(),
            fp = this.getFramePainter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.copyOptionsFrom(main);

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   async redraw(reason) {
      return this.callDrawFunc(reason);
   }

   /** @summary draw TH2 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH2Painter(dom, histo), opt);
   }

} // class TH2Painter

export { TH2Painter, buildHist2dContour, buildSurf3D, Triangles3DHandler };
