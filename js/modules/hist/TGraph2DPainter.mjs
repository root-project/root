import { BIT, settings, createHistogram } from '../core.mjs';
import { REVISION, Color, LineBasicMaterial } from '../three.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { TH2Painter } from './TH2Painter.mjs';
import { createLineSegments, PointsCreator } from '../base/base3d.mjs';

/**
 * @summary Painter for TGraph2D classes
 * @private
 */

class TGraph2DPainter extends ObjectPainter {

   /** @summary Decode options string  */
   decodeOptions(opt, gr) {
      let d = new DrawOptions(opt);

      if (!this.options)
         this.options = {};

      let res = this.options;

      res.Color = d.check("COL");
      res.Line = d.check("LINE");
      res.Error = d.check("ERR") && (this.matchObjectType("TGraph2DErrors") || this.matchObjectType("TGraph2DAsymmErrors"));
      res.Circles = d.check("P0");
      res.Markers = d.check("P");

      if (!res.Markers && !res.Error && !res.Circles && !res.Line) {
         if ((gr.fMarkerSize == 1) && (gr.fMarkerStyle == 1))
            res.Circles = true;
         else
            res.Markers = true;
      }
      if (!res.Markers) res.Color = false;

      this.storeDrawOpt(opt);
   }

   /** @summary Create histogram for axes drawing */
   createHistogram() {
      let gr = this.getObject(),
          asymm = this.matchObjectType("TGraph2DAsymmErrors"),
          xmin = gr.fX[0], xmax = xmin,
          ymin = gr.fY[0], ymax = ymin,
          zmin = gr.fZ[0], zmax = zmin;

      for (let p = 0; p < gr.fNpoints;++p) {

         let x = gr.fX[p], y = gr.fY[p], z = gr.fZ[p];

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

      if (xmin >= xmax) xmax = xmin+1;
      if (ymin >= ymax) ymax = ymin+1;
      if (zmin >= zmax) zmax = zmin+1;
      let dx = (xmax-xmin)*0.02, dy = (ymax-ymin)*0.02, dz = (zmax-zmin)*0.02,
          uxmin = xmin - dx, uxmax = xmax + dx,
          uymin = ymin - dy, uymax = ymax + dy,
          uzmin = zmin - dz, uzmax = zmax + dz;

      if ((uxmin<0) && (xmin>=0)) uxmin = xmin*0.98;
      if ((uxmax>0) && (xmax<=0)) uxmax = 0;

      if ((uymin<0) && (ymin>=0)) uymin = ymin*0.98;
      if ((uymax>0) && (ymax<=0)) uymax = 0;

      if ((uzmin<0) && (zmin>=0)) uzmin = zmin*0.98;
      if ((uzmax>0) && (zmax<=0)) uzmax = 0;

      let graph = this.getObject();

      if (graph.fMinimum != -1111) uzmin = graph.fMinimum;
      if (graph.fMaximum != -1111) uzmax = graph.fMaximum;

      let histo = createHistogram("TH2I", 10, 10);
      histo.fName = graph.fName + "_h";
      histo.fTitle = graph.fTitle;
      histo.fXaxis.fXmin = uxmin;
      histo.fXaxis.fXmax = uxmax;
      histo.fYaxis.fXmin = uymin;
      histo.fYaxis.fXmax = uymax;
      histo.fZaxis.fXmin = uzmin;
      histo.fZaxis.fXmax = uzmax;
      histo.fMinimum = uzmin;
      histo.fMaximum = uzmax;
      let kNoStats = BIT(9);
      histo.fBits = histo.fBits | kNoStats;
      return histo;
   }

   /** @summary Function handles tooltips in the mesh */
   graph2DTooltip(intersect) {
      if (!Number.isInteger(intersect.index)) {
         console.error(`intersect.index not provided, three.js version ${REVISION}`);
         return null;
      }

      let indx = Math.floor(intersect.index / this.nvertex);
      if ((indx<0) || (indx >= this.index.length)) return null;
      let sqr = v => v*v;

      indx = this.index[indx];

      let p = this.painter, gr = this.graph,
          grx = p.grx(gr.fX[indx]),
          gry = p.gry(gr.fY[indx]),
          grz = p.grz(gr.fZ[indx]);

      if (this.check_next && indx+1<gr.fX.length) {
         let d = intersect.point,
             grx1 = p.grx(gr.fX[indx+1]),
             gry1 = p.gry(gr.fY[indx+1]),
             grz1 = p.grz(gr.fZ[indx+1]);
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
         lines: [ this.tip_name,
                  "pnt: " + indx,
                  "x: " + p.axisAsText("x", gr.fX[indx]),
                  "y: " + p.axisAsText("y", gr.fY[indx]),
                  "z: " + p.axisAsText("z", gr.fZ[indx])
                ]
      };
   }

   /** @summary Actual drawing of TGraph2D object
     * @returns {Promise} for drawing ready */
   redraw() {

      let main = this.getMainPainter(),
          fp = this.getFramePainter(),
          graph = this.getObject(),
          step = 1;

      if (!graph || !main || !fp || !fp.mode3d)
         return Promise.resolve(this);

      let countSelected = (zmin, zmax) => {
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
      if ((settings.OptimizeDraw > 0) && !fp.webgl) {
         let numselected = countSelected(fp.scale_zmin, fp.scale_zmax),
             sizelimit = 50000;

         if (numselected > sizelimit) {
            step = Math.floor(numselected / sizelimit);
            if (step <= 2) step = 2;
         }
      }

      let markeratt = new TAttMarkerHandler(graph),
          palette = null,
          levels = [fp.scale_zmin, fp.scale_zmax],
          scale = fp.size_x3d / 100 * markeratt.getFullSize(),
          promises = [];

      if (this.options.Circles) scale = 0.06*fp.size_x3d;

      if (fp.usesvg) scale *= 0.3;

      if (this.options.Color) {
         levels = main.getContourLevels();
         palette = main.getHistPalette();
      }

      for (let lvl = 0; lvl < levels.length-1; ++lvl) {

         let lvl_zmin = Math.max(levels[lvl], fp.scale_zmin),
             lvl_zmax = Math.min(levels[lvl+1], fp.scale_zmax);

         if (lvl_zmin >= lvl_zmax) continue;

         let size = Math.floor(countSelected(lvl_zmin, lvl_zmax) / step),
             pnts = null, select = 0,
             index = new Int32Array(size), icnt = 0,
             err = null, asymm = false, line = null, ierr = 0, iline = 0;

         if (this.options.Markers || this.options.Circles)
            pnts = new PointsCreator(size, fp.webgl, scale/3);

         if (this.options.Error) {
            err = new Float32Array(size*6*3);
            asymm = this.matchObjectType("TGraph2DAsymmErrors");
          }

         if (this.options.Line)
            line = new Float32Array((size-1)*6);

         for (let i = 0; i < graph.fNpoints; ++i) {
            if ((graph.fX[i] < fp.scale_xmin) || (graph.fX[i] > fp.scale_xmax) ||
                (graph.fY[i] < fp.scale_ymin) || (graph.fY[i] > fp.scale_ymax) ||
                (graph.fZ[i] < lvl_zmin) || (graph.fZ[i] >= lvl_zmax)) continue;

            if (step > 1) {
               select = (select+1) % step;
               if (select!==0) continue;
            }

            index[icnt++] = i; // remember point index for tooltip

            let x = fp.grx(graph.fX[i]),
                y = fp.gry(graph.fY[i]),
                z = fp.grz(graph.fZ[i]);

            if (pnts) pnts.addPoint(x,y,z);

            if (err) {
               err[ierr]   = fp.grx(graph.fX[i] - (asymm ? graph.fEXlow[i] : graph.fEX[i]));
               err[ierr+1] = y;
               err[ierr+2] = z;
               err[ierr+3] = fp.grx(graph.fX[i] + (asymm ? graph.fEXhigh[i] : graph.fEX[i]));
               err[ierr+4] = y;
               err[ierr+5] = z;
               ierr+=6;
               err[ierr]   = x;
               err[ierr+1] = fp.gry(graph.fY[i] - (asymm ? graph.fEYlow[i] : graph.fEY[i]));
               err[ierr+2] = z;
               err[ierr+3] = x;
               err[ierr+4] = fp.gry(graph.fY[i] + (asymm ? graph.fEYhigh[i] : graph.fEY[i]));
               err[ierr+5] = z;
               ierr+=6;
               err[ierr]   = x;
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

         if (line && (iline > 3) && (line.length == iline)) {
            let lcolor = this.getColor(graph.fLineColor),
                material = new LineBasicMaterial({ color: new Color(lcolor), linewidth: graph.fLineWidth }),
                linemesh = createLineSegments(line, material);
            fp.toplevel.add(linemesh);

            linemesh.graph = graph;
            linemesh.index = index;
            linemesh.painter = fp;
            linemesh.scale0 = 0.7*scale;
            linemesh.tip_name = this.getObjectHint();
            linemesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
            linemesh.nvertex = 2;
            linemesh.check_next = true;

            linemesh.tooltip = this.graph2DTooltip;
         }

         if (err) {
            let lcolor = this.getColor(graph.fLineColor),
                material = new LineBasicMaterial({ color: new Color(lcolor), linewidth: graph.fLineWidth }),
                errmesh = createLineSegments(err, material);
            fp.toplevel.add(errmesh);

            errmesh.graph = graph;
            errmesh.index = index;
            errmesh.painter = fp;
            errmesh.scale0 = 0.7*scale;
            errmesh.tip_name = this.getObjectHint();
            errmesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
            errmesh.nvertex = 6;

            errmesh.tooltip = this.graph2DTooltip;
         }

         if (pnts) {
            let fcolor = 'blue';

            if (!this.options.Circles)
               fcolor = palette ? palette.calcColor(lvl, levels.length)
                                : this.getColor(graph.fMarkerColor);

            let pr = pnts.createPoints({ color: fcolor, style: this.options.Circles ? 4 : graph.fMarkerStyle }).then(mesh => {
               mesh.graph = graph;
               mesh.painter = fp;
               mesh.tip_color = (graph.fMarkerColor === 3) ? 0xFF0000 : 0x00FF00;
               mesh.scale0 = 0.3*scale;
               mesh.index = index;

               mesh.tip_name = this.getObjectHint();
               mesh.tooltip = this.graph2DTooltip;
               fp.toplevel.add(mesh);
            });

            promises.push(pr);
         }
      }

      return Promise.all(promises).then(() => {
         fp.render3D(100);
         return this;
      });
   }

   /** @summary draw TGraph2D object */
   static draw(dom, gr, opt) {
      let painter = new TGraph2DPainter(dom, gr);
      painter.decodeOptions(opt, gr);

      let promise = Promise.resolve(true);

      if (!painter.getMainPainter()) {
         if (!gr.fHistogram)
            gr.fHistogram = painter.createHistogram();
         promise = TH2Painter.draw(dom, gr.fHistogram, "lego;axis");
         painter.ownhisto = true;
      }

      return promise.then(() => {
         painter.addToPadPrimitives();
         return painter.redraw();
      });
   }

} // class TGraph2DPainter

export { TGraph2DPainter };
