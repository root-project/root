import { gStyle, create } from '../core.mjs';

import { DrawOptions, floatToString, buildSvgPath } from '../base/BasePainter.mjs';

import { ObjectPainter } from '../base/ObjectPainter.mjs';

import { TH1Painter } from '../hist/TH1Painter.mjs';

/**
 * @summary Painter for TSpline objects.
 *
 * @private
 */

class TSplinePainter extends ObjectPainter {

   /** @summary Update TSpline object
     * @private */
   updateObject(obj, opt) {
      let spline = this.getObject();

      if (spline._typename != obj._typename) return false;

      if (spline !== obj) Object.assign(spline, obj);

      if (opt !== undefined) this.decodeOptions(opt);

      return true;
   }

   /** @summary Evaluate spline at given position
     * @private */
   eval(knot, x) {
      let dx = x - knot.fX;

      if (knot._typename == "TSplinePoly3")
         return knot.fY + dx*(knot.fB + dx*(knot.fC + dx*knot.fD));

      if (knot._typename == "TSplinePoly5")
         return knot.fY + dx*(knot.fB + dx*(knot.fC + dx*(knot.fD + dx*(knot.fE + dx*knot.fF))));

      return knot.fY + dx;
   }

   /** @summary Find idex for x value
     * @private */
   findX(x) {
      let spline = this.getObject(),
          klow = 0, khig = spline.fNp - 1;

      if (x <= spline.fXmin) return 0;
      if (x >= spline.fXmax) return khig;

      if(spline.fKstep) {
         // Equidistant knots, use histogramming
         klow = Math.round((x - spline.fXmin)/spline.fDelta);
         // Correction for rounding errors
         if (x < spline.fPoly[klow].fX) {
            klow = Math.max(klow-1,0);
         } else if (klow < khig) {
            if (x > spline.fPoly[klow+1].fX) ++klow;
         }
      } else {
         // Non equidistant knots, binary search
         while(khig-klow>1) {
            let khalf = Math.round((klow+khig)/2);
            if(x > spline.fPoly[khalf].fX) klow = khalf;
                                      else khig = khalf;
         }
      }
      return klow;
   }

   /** @summary Create histogram for axes drawing
     * @private */
   createDummyHisto() {

      let xmin = 0, xmax = 1, ymin = 0, ymax = 1,
          spline = this.getObject();

      if (spline && spline.fPoly) {

         xmin = xmax = spline.fPoly[0].fX;
         ymin = ymax = spline.fPoly[0].fY;

         spline.fPoly.forEach(knot => {
            xmin = Math.min(knot.fX, xmin);
            xmax = Math.max(knot.fX, xmax);
            ymin = Math.min(knot.fY, ymin);
            ymax = Math.max(knot.fY, ymax);
         });

         if (ymax > 0.0) ymax *= 1.05;
         if (ymin < 0.0) ymin *= 1.05;
      }

      let histo = create("TH1I");

      histo.fName = spline.fName + "_hist";
      histo.fTitle = spline.fTitle;

      histo.fXaxis.fXmin = xmin;
      histo.fXaxis.fXmax = xmax;
      histo.fYaxis.fXmin = ymin;
      histo.fYaxis.fXmax = ymax;

      return histo;
   }

   /** @summary Process tooltip event
     * @private */
   processTooltipEvent(pnt) {

      let cleanup = false,
          spline = this.getObject(),
          main = this.getFramePainter(),
          funcs = main ? main.getGrFuncs(this.options.second_x, this.options.second_y) : null,
          xx, yy, knot = null, indx = 0;

      if ((pnt === null) || !spline || !funcs) {
         cleanup = true;
      } else {
         xx = funcs.revertAxis("x", pnt.x);
         indx = this.findX(xx);
         knot = spline.fPoly[indx];
         yy = this.eval(knot, xx);

         if ((indx < spline.fN-1) && (Math.abs(spline.fPoly[indx+1].fX-xx) < Math.abs(xx-knot.fX))) knot = spline.fPoly[++indx];

         if (Math.abs(funcs.grx(knot.fX) - pnt.x) < 0.5*this.knot_size) {
            xx = knot.fX; yy = knot.fY;
         } else {
            knot = null;
            if ((xx < spline.fXmin) || (xx > spline.fXmax)) cleanup = true;
         }
      }

      if (cleanup) {
         if (this.draw_g)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      let gbin = this.draw_g.select(".tooltip_bin"),
          radius = this.lineatt.width + 3;

      if (gbin.empty())
         gbin = this.draw_g.append("svg:circle")
                           .attr("class", "tooltip_bin")
                           .style("pointer-events","none")
                           .attr("r", radius)
                           .style("fill", "none")
                           .call(this.lineatt.func);

      let res = { name: this.getObject().fName,
                  title: this.getObject().fTitle,
                  x: funcs.grx(xx),
                  y: funcs.gry(yy),
                  color1: this.lineatt.color,
                  lines: [],
                  exact: (knot !== null) || (Math.abs(funcs.gry(yy) - pnt.y) < radius) };

      res.changed = gbin.property("current_xx") !== xx;
      res.menu = res.exact;
      res.menu_dist = Math.sqrt((res.x-pnt.x)*(res.x-pnt.x) + (res.y-pnt.y)*(res.y-pnt.y));

      if (res.changed)
         gbin.attr("cx", Math.round(res.x))
             .attr("cy", Math.round(res.y))
             .property("current_xx", xx);

      let name = this.getObjectHint();
      if (name.length > 0) res.lines.push(name);
      res.lines.push("x = " + funcs.axisAsText("x", xx));
      res.lines.push("y = " + funcs.axisAsText("y", yy));
      if (knot !== null) {
         res.lines.push("knot = " + indx);
         res.lines.push("B = " + floatToString(knot.fB, gStyle.fStatFormat));
         res.lines.push("C = " + floatToString(knot.fC, gStyle.fStatFormat));
         res.lines.push("D = " + floatToString(knot.fD, gStyle.fStatFormat));
         if ((knot.fE!==undefined) && (knot.fF!==undefined)) {
            res.lines.push("E = " + floatToString(knot.fE, gStyle.fStatFormat));
            res.lines.push("F = " + floatToString(knot.fF, gStyle.fStatFormat));
         }
      }

      return res;
   }

   /** @summary Redraw object
     * @private */
   redraw() {

      let spline = this.getObject(),
          pmain = this.getFramePainter(),
          funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null,
          w = pmain.getFrameWidth(),
          h = pmain.getFrameHeight();

      this.createG(true);

      this.knot_size = 5; // used in tooltip handling

      this.createAttLine({ attr: spline });

      if (this.options.Line || this.options.Curve) {

         let npx = Math.max(10, spline.fNpx),
             xmin = Math.max(pmain.scale_xmin, spline.fXmin),
             xmax = Math.min(pmain.scale_xmax, spline.fXmax),
             indx = this.findX(xmin),
             bins = []; // index of current knot

         if (pmain.logx) {
            xmin = Math.log(xmin);
            xmax = Math.log(xmax);
         }

         for (let n = 0; n < npx; ++n) {
            let xx = xmin + (xmax-xmin)/npx*(n-1);
            if (pmain.logx) xx = Math.exp(xx);

            while ((indx < spline.fNp-1) && (xx > spline.fPoly[indx+1].fX)) ++indx;

            let yy = this.eval(spline.fPoly[indx], xx);

            bins.push({ x: xx, y: yy, grx: funcs.grx(xx), gry: funcs.gry(yy) });
         }

         let h0 = h;  // use maximal frame height for filling
         if ((pmain.hmin!==undefined) && (pmain.hmin >= 0)) {
            h0 = Math.round(funcs.gry(0));
            if ((h0 > h) || (h0 < 0)) h0 = h;
         }

         let path = buildSvgPath("bezier", bins, h0, 2);

         this.draw_g.append("svg:path")
             .attr("class", "line")
             .attr("d", path.path)
             .style("fill", "none")
             .call(this.lineatt.func);
      }

      if (this.options.Mark) {

         // for tooltips use markers only if nodes where not created
         let path = "";

         this.createAttMarker({ attr: spline });

         this.markeratt.resetPos();

         this.knot_size = this.markeratt.getFullSize();

         for (let n=0; n<spline.fPoly.length; n++) {
            let knot = spline.fPoly[n],
                grx = funcs.grx(knot.fX);
            if ((grx > -this.knot_size) && (grx < w + this.knot_size)) {
               let gry = funcs.gry(knot.fY);
               if ((gry > -this.knot_size) && (gry < h + this.knot_size)) {
                  path += this.markeratt.create(grx, gry);
               }
            }
         }

         if (path)
            this.draw_g.append("svg:path")
                       .attr("d", path)
                       .call(this.markeratt.func);
      }
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis/*,min,max*/) {
      if (axis!=="x") return false;

      let spline = this.getObject();
      if (!spline) return false;

      // if function calculated, one always could zoom inside
      return true;
   }

   /** @summary Decode options for TSpline drawing */
   decodeOptions(opt) {
      let d = new DrawOptions(opt);

      if (!this.options) this.options = {};

      let has_main = !!this.getMainPainter();

      Object.assign(this.options, {
         Same: d.check('SAME'),
         Line: d.check('L'),
         Curve: d.check('C'),
         Mark: d.check('P'),
         Hopt: "AXIS",
         second_x: false,
         second_y: false
      });

      if (!this.options.Line && !this.options.Curve && !this.options.Mark)
         this.options.Curve = true;

      if (d.check("X+")) { this.options.Hopt += "X+"; this.options.second_x = has_main; }
      if (d.check("Y+")) { this.options.Hopt += "Y+"; this.options.second_y = has_main; }

      this.storeDrawOpt(opt);
   }

   /** @summary Draw TSpline */
   static draw(dom, spline, opt) {
      let painter = new TSplinePainter(dom, spline);
      painter.decodeOptions(opt);

      let promise = Promise.resolve(), no_main = !painter.getMainPainter();
      if (no_main || painter.options.second_x || painter.options.second_y) {
         if (painter.options.Same && no_main) {
            console.warn('TSpline painter requires histogram to be drawn');
            return null;
         }
         let histo = painter.createDummyHisto();
         promise = TH1Painter.draw(dom, histo, painter.options.Hopt);
      }

      return promise.then(() => {
         painter.addToPadPrimitives();
         painter.redraw();
         return painter;
      });
   }

} // class TSplinePainter

export { TSplinePainter }
