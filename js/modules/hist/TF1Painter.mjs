import { create, gStyle } from '../core.mjs';
import { DrawOptions, buildSvgPath } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import * as jsroot_math from '../base/math.mjs';


function proivdeEvalPar(obj) {

   obj._math = jsroot_math;

   let _func = obj.fTitle, isformula = false, pprefix = "[";
   if (_func === "gaus") _func = "gaus(0)";
   if (obj.fFormula && typeof obj.fFormula.fFormula == "string") {
     if (obj.fFormula.fFormula.indexOf("[](double*x,double*p)")==0) {
        isformula = true; pprefix = "p[";
        _func = obj.fFormula.fFormula.slice(21);
     } else {
        _func = obj.fFormula.fFormula;
        pprefix = "[p";
     }
     if (obj.fFormula.fClingParameters && obj.fFormula.fParams)
        obj.fFormula.fParams.forEach(pair => {
           let regex = new RegExp(`(\\[${pair.first}\\])`, 'g'),
               parvalue = obj.fFormula.fClingParameters[pair.second];
           _func = _func.replace(regex, (parvalue < 0) ? `(${parvalue})` : parvalue);
        });

  }

  if ('formulas' in obj)
     obj.formulas.forEach(entry => {
       _func = _func.replaceAll(entry.fName, entry.fTitle);
     });

  _func = _func.replace(/\b(abs)\b/g, 'TMath::Abs')
               .replace(/\b(TMath::Exp)/g, 'Math.exp')
               .replace(/\b(TMath::Abs)/g, 'Math.abs');

  _func = _func.replace(/xygaus\(/g, 'this._math.gausxy(this, x, y, ')
               .replace(/gaus\(/g, 'this._math.gaus(this, x, ')
               .replace(/gausn\(/g, 'this._math.gausn(this, x, ')
               .replace(/expo\(/g, 'this._math.expo(this, x, ')
               .replace(/landau\(/g, 'this._math.landau(this, x, ')
               .replace(/landaun\(/g, 'this._math.landaun(this, x, ')
               .replace(/TMath::/g, 'this._math.')
               .replace(/ROOT::Math::/g, 'this._math.');

  for (let i = 0; i < obj.fNpar; ++i)
    _func = _func.replaceAll(pprefix + i + "]", `(${obj.GetParValue(i)})`);

  _func = _func.replace(/\b(sin)\b/gi, 'Math.sin')
               .replace(/\b(cos)\b/gi, 'Math.cos')
               .replace(/\b(tan)\b/gi, 'Math.tan')
               .replace(/\b(exp)\b/gi, 'Math.exp')
               .replace(/\b(pow)\b/gi, 'Math.pow')
               .replace(/pi/g, 'Math.PI');
  for (let n = 2; n < 10; ++n)
     _func = _func.replaceAll(`x^${n}`, `Math.pow(x,${n})`);

  if (isformula) {
     _func = _func.replace(/x\[0\]/g,"x");
     if (obj._typename === "TF2") {
        _func = _func.replace(/x\[1\]/g,"y");
        obj.evalPar = new Function("x", "y", _func).bind(obj);
     } else {
        obj.evalPar = new Function("x", _func).bind(obj);
     }
  } else if (obj._typename === "TF2")
     obj.evalPar = new Function("x", "y", "return " + _func).bind(obj);
  else
     obj.evalPar = new Function("x", "return " + _func).bind(obj);
}

/**
  * @summary Painter for TF1 object
  *
  * @private
  */

class TF1Painter extends ObjectPainter {

   /** @summary Create bins for TF1 drawing */
   createBins(ignore_zoom) {
      let tf1 = this.getObject(),
          main = this.getFramePainter(),
          gxmin = 0, gxmax = 0;

      if (main && !ignore_zoom)  {
         let gr = main.getGrFuncs(this.second_x, this.second_y);
         gxmin = gr.scale_xmin;
         gxmax = gr.scale_xmax;
      }

      let xmin = tf1.fXmin, xmax = tf1.fXmax, logx = false;

      if (gxmin !== gxmax) {
         if (gxmin > xmin) xmin = gxmin;
         if (gxmax < xmax) xmax = gxmax;
      }

      if (main && main.logx && (xmin > 0) && (xmax > 0)) {
         logx = true;
         xmin = Math.log(xmin);
         xmax = Math.log(xmax);
      }

      let np = Math.max(tf1.fNpx, 101),
          dx = (xmax - xmin) / (np - 1),
          res = [], iserror = false,
          force_use_save = (tf1.fSave.length > 3) && ignore_zoom;

      if (!force_use_save)
         for (let n = 0; n < np; n++) {
            let xx = xmin + n*dx, yy = 0;
            if (logx) xx = Math.exp(xx);
            try {
               yy = tf1.evalPar(xx);
            } catch(err) {
               iserror = true;
            }

            if (iserror) break;

            if (Number.isFinite(yy))
               res.push({ x: xx, y: yy });
         }

      // in the case there were points have saved and we cannot calculate function
      // if we don't have the user's function
      if ((iserror || ignore_zoom || !res.length) && (tf1.fSave.length > 3)) {

         np = tf1.fSave.length - 2;
         xmin = tf1.fSave[np];
         xmax = tf1.fSave[np+1];
         res = [];
         dx = 0;
         let use_histo = tf1.$histo && (xmin === xmax), bin = 0;

         if (use_histo) {
            xmin = tf1.fSave[--np];
            bin = tf1.$histo.fXaxis.FindBin(xmin, 0);
         } else {
            dx = (xmax - xmin) / (np-1);
         }

         for (let n = 0; n < np; ++n) {
            let xx = use_histo ? tf1.$histo.fXaxis.GetBinCenter(bin+n+1) : xmin + dx*n;
            // check if points need to be displayed at all, keep at least 4-5 points for Bezier curves
            if ((gxmin !== gxmax) && ((xx + 2*dx < gxmin) || (xx - 2*dx > gxmax))) continue;
            let yy = tf1.fSave[n];

            if (Number.isFinite(yy)) res.push({ x : xx, y : yy });
         }
      }

      return res;
   }

   /** @summary Create histogram for axes drawing */
   createDummyHisto() {

      let xmin = 0, xmax = 1, ymin = 0, ymax = 1,
          bins = this.createBins(true);

      if (bins && (bins.length > 0)) {

         xmin = xmax = bins[0].x;
         ymin = ymax = bins[0].y;

         bins.forEach(bin => {
            xmin = Math.min(bin.x, xmin);
            xmax = Math.max(bin.x, xmax);
            ymin = Math.min(bin.y, ymin);
            ymax = Math.max(bin.y, ymax);
         });

         if (ymax > 0.0) ymax *= (1 + gStyle.fHistTopMargin);
         if (ymin < 0.0) ymin *= (1 + gStyle.fHistTopMargin);
      }

      let histo = create("TH1I"),
          tf1 = this.getObject();

      histo.fName = tf1.fName + "_hist";
      histo.fTitle = tf1.fTitle;

      histo.fXaxis.fXmin = xmin;
      histo.fXaxis.fXmax = xmax;
      histo.fYaxis.fXmin = ymin;
      histo.fYaxis.fXmax = ymax;

      histo.fMinimum = tf1.fMinimum;
      histo.fMaximum = tf1.fMaximum;

      return histo;
   }

   updateObject(obj /*, opt */) {
      if (!this.matchObjectType(obj)) return false;
      Object.assign(this.getObject(), obj);
      proivdeEvalPar(this.getObject());
      return true;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      let cleanup = false;

      if (!pnt || !this.bins || pnt.disabled) {
         cleanup = true;
      } else if (!this.bins.length || (pnt.x < this.bins[0].grx) || (pnt.x > this.bins[this.bins.length-1].grx)) {
         cleanup = true;
      }

      if (cleanup) {
         if (this.draw_g)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      let min = 100000, best = -1, bin;

      for(let n = 0; n < this.bins.length; ++n) {
         bin = this.bins[n];
         let dist = Math.abs(bin.grx - pnt.x);
         if (dist < min) { min = dist; best = n; }
      }

      bin = this.bins[best];

      let gbin = this.draw_g.select(".tooltip_bin"),
          radius = this.lineatt.width + 3;

      if (gbin.empty())
         gbin = this.draw_g.append("svg:circle")
                           .attr("class","tooltip_bin")
                           .style("pointer-events","none")
                           .attr("r", radius)
                           .call(this.lineatt.func)
                           .call(this.fillatt.func);

      let res = { name: this.getObject().fName,
                  title: this.getObject().fTitle,
                  x: bin.grx,
                  y: bin.gry,
                  color1: this.lineatt.color,
                  color2: this.fillatt.getFillColor(),
                  lines: [],
                  exact: (Math.abs(bin.grx - pnt.x) < radius) && (Math.abs(bin.gry - pnt.y) < radius) };

      res.changed = gbin.property("current_bin") !== best;
      res.menu = res.exact;
      res.menu_dist = Math.sqrt((bin.grx-pnt.x)**2 + (bin.gry-pnt.y)**2);

      if (res.changed)
         gbin.attr("cx", bin.grx)
             .attr("cy", bin.gry)
             .property("current_bin", best);

      let name = this.getObjectHint();
      if (name.length > 0) res.lines.push(name);

      let pmain = this.getFramePainter(),
          funcs = pmain?.getGrFuncs(this.second_x, this.second_y);
      if (funcs)
         res.lines.push("x = " + funcs.axisAsText("x",bin.x) + " y = " + funcs.axisAsText("y",bin.y));

      return res;
   }

   /** @summary Redraw function */
   redraw() {

      let tf1 = this.getObject(),
          fp = this.getFramePainter(),
          h = fp.getFrameHeight(),
          pmain = this.getMainPainter();

      this.createG(true);

      // recalculate drawing bins when necessary
      this.bins = this.createBins(false);

      this.createAttLine({ attr: tf1 });
      this.lineatt.used = false;

      this.createAttFill({ attr: tf1, kind: 1 });
      this.fillatt.used = false;

      let funcs = fp.getGrFuncs(this.second_x, this.second_y);

      // first calculate graphical coordinates
      for(let n = 0; n < this.bins.length; ++n) {
         let bin = this.bins[n];
         bin.grx = funcs.grx(bin.x);
         bin.gry = funcs.gry(bin.y);
      }

      if (this.bins.length > 2) {

         let h0 = h;  // use maximal frame height for filling
         if (pmain.hmin && (pmain.hmin >= 0)) {
            h0 = Math.round(funcs.gry(0));
            if ((h0 > h) || (h0 < 0)) h0 = h;
         }

         let path = buildSvgPath("bezier", this.bins, h0, 2);

         if (!this.lineatt.empty())
            this.draw_g.append("svg:path")
               .attr("class", "line")
               .attr("d", path.path)
               .style("fill", "none")
               .call(this.lineatt.func);

         if (!this.fillatt.empty())
            this.draw_g.append("svg:path")
               .attr("class", "area")
               .attr("d", path.path + path.close)
               .call(this.fillatt.func);
      }
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      if (axis!=="x") return false;

      let tf1 = this.getObject();

      if (tf1.fSave.length > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         let nb_points = tf1.fNpx,
             xmin = tf1.fSave[nb_points + 1],
             xmax = tf1.fSave[nb_points + 2];

         return Math.abs(xmin - xmax) / nb_points < Math.abs(min - max);
      }

      // if function calculated, one always could zoom inside
      return true;
   }

   /** @summary draw TF1 object */
   static draw(dom, tf1, opt) {
      let painter = new TF1Painter(dom, tf1, opt),
          d = new DrawOptions(opt),
          has_main = !!painter.getMainPainter(),
          aopt = "AXIS";
      d.check('SAME'); // just ignore same
      if (d.check('X+')) { aopt += "X+"; painter.second_x = has_main; }
      if (d.check('Y+')) { aopt += "Y+"; painter.second_y = has_main; }
      if (d.check('RX')) aopt += "RX";
      if (d.check('RY')) aopt += "RY";

      proivdeEvalPar(tf1);

      let pr = Promise.resolve(true);

      if (!has_main || painter.second_x || painter.second_y)
         pr = TH1Painter.draw(dom, painter.createDummyHisto(), aopt);

      return pr.then(() => {
         painter.addToPadPrimitives();
         painter.redraw();
         return painter;
      });
   }

} // class TF1Painter

export { TF1Painter, proivdeEvalPar };
