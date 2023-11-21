import { settings, gStyle, isStr, isFunc, clTH1D, createHistogram, setHistogramTitle, clTF1, clTF2, clTF3, kNoStats } from '../core.mjs';
import { floatToString } from '../base/BasePainter.mjs';
import { getElementMainPainter, ObjectPainter } from '../base/ObjectPainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import * as jsroot_math from '../base/math.mjs';


/** @summary Assign `evalPar` function for TF1 object
  * @private */

function proivdeEvalPar(obj, check_save) {
   obj.$math = jsroot_math;

   let _func = obj.fTitle, isformula = false, pprefix = '[';
   if (_func === 'gaus') _func = 'gaus(0)';
   if (isStr(obj.fFormula?.fFormula)) {
     if (obj.fFormula.fFormula.indexOf('[](double*x,double*p)') === 0) {
        isformula = true; pprefix = 'p[';
        _func = obj.fFormula.fFormula.slice(21);
     } else {
        _func = obj.fFormula.fFormula;
        pprefix = '[p';
     }

     if (obj.fFormula.fClingParameters && obj.fFormula.fParams) {
        obj.fFormula.fParams.forEach(pair => {
           const regex = new RegExp(`(\\[${pair.first}\\])`, 'g'),
               parvalue = obj.fFormula.fClingParameters[pair.second];
           _func = _func.replace(regex, (parvalue < 0) ? `(${parvalue})` : parvalue);
        });
      }
   }

   if (!_func)
      return !check_save || (obj.fSave?.length > 2);

   obj.formulas?.forEach(entry => {
      _func = _func.replaceAll(entry.fName, entry.fTitle);
   });

   _func = _func.replace(/\b(TMath::SinH)\b/g, 'Math.sinh')
                .replace(/\b(TMath::CosH)\b/g, 'Math.cosh')
                .replace(/\b(TMath::TanH)\b/g, 'Math.tanh')
                .replace(/\b(TMath::ASinH)\b/g, 'Math.asinh')
                .replace(/\b(TMath::ACosH)\b/g, 'Math.acosh')
                .replace(/\b(TMath::ATanH)\b/g, 'Math.atanh')
                .replace(/\b(TMath::ASin)\b/g, 'Math.asin')
                .replace(/\b(TMath::ACos)\b/g, 'Math.acos')
                .replace(/\b(TMath::Atan)\b/g, 'Math.atan')
                .replace(/\b(TMath::ATan2)\b/g, 'Math.atan2')
                .replace(/\b(sin|SIN|TMath::Sin)\b/g, 'Math.sin')
                .replace(/\b(cos|COS|TMath::Cos)\b/g, 'Math.cos')
                .replace(/\b(tan|TAN|TMath::Tan)\b/g, 'Math.tan')
                .replace(/\b(exp|EXP|TMath::Exp)\b/g, 'Math.exp')
                .replace(/\b(log|LOG|TMath::Log)\b/g, 'Math.log')
                .replace(/\b(log10|LOG10|TMath::Log10)\b/g, 'Math.log10')
                .replace(/\b(pow|POW|TMath::Power)\b/g, 'Math.pow')
                .replace(/\b(pi|PI)\b/g, 'Math.PI')
                .replace(/\b(abs|ABS|TMath::Abs)\b/g, 'Math.abs')
                .replace(/\bxygaus\(/g, 'this.$math.gausxy(this, x, y, ')
                .replace(/\bgaus\(/g, 'this.$math.gaus(this, x, ')
                .replace(/\bgausn\(/g, 'this.$math.gausn(this, x, ')
                .replace(/\bexpo\(/g, 'this.$math.expo(this, x, ')
                .replace(/\blandau\(/g, 'this.$math.landau(this, x, ')
                .replace(/\blandaun\(/g, 'this.$math.landaun(this, x, ')
                .replace(/\b(TMath::|ROOT::Math::)/g, 'this.$math.');

   if (_func.match(/^pol[0-9]$/) && (parseInt(_func[3]) === obj.fNpar - 1)) {
      _func = '[0]';
      for (let k = 1; k < obj.fNpar; ++k)
         _func += ` + [${k}] * `+ ((k === 1) ? 'x' : `Math.pow(x,${k})`);
   }

   if (_func.match(/^chebyshev[0-9]$/) && (parseInt(_func[9]) === obj.fNpar - 1)) {
      _func = `this.$math.ChebyshevN(${obj.fNpar-1}, x, `;
      for (let k = 0; k < obj.fNpar; ++k)
         _func += (k === 0 ? '[' : ', ') + `[${k}]`;
      _func += '])';
   }

   for (let i = 0; i < obj.fNpar; ++i)
      _func = _func.replaceAll(pprefix + i + ']', `(${obj.GetParValue(i)})`);

   for (let n = 2; n < 10; ++n)
      _func = _func.replaceAll(`x^${n}`, `Math.pow(x,${n})`);

   if (isformula) {
      _func = _func.replace(/x\[0\]/g, 'x');
      if (obj._typename === clTF3) {
         _func = _func.replace(/x\[1\]/g, 'y');
         _func = _func.replace(/x\[2\]/g, 'z');
         obj.evalPar = new Function('x', 'y', 'z', _func).bind(obj);
      } else if (obj._typename === clTF2) {
         _func = _func.replace(/x\[1\]/g, 'y');
         obj.evalPar = new Function('x', 'y', _func).bind(obj);
      } else
         obj.evalPar = new Function('x', _func).bind(obj);
   } else if (obj._typename === clTF3)
      obj.evalPar = new Function('x', 'y', 'z', 'return ' + _func).bind(obj);
   else if (obj._typename === clTF2)
      obj.evalPar = new Function('x', 'y', 'return ' + _func).bind(obj);
   else
      obj.evalPar = new Function('x', 'return ' + _func).bind(obj);

   return true;
}


/** @summary Get interpolation in saved buffer
  * @desc Several checks must be done before function can be used
  * @private */
function _getTF1Save(func, x) {
   const np = func.fSave.length - 3,
         xmin = func.fSave[np + 1],
        xmax = func.fSave[np + 2],
        dx = (xmax - xmin) / np;
    if (x < xmin)
       return func.fSave[0];
    if (x > xmax)
       return func.fSave[np];

    const bin = Math.min(np - 1, Math.floor((x - xmin) / dx));
    let xlow = xmin + bin * dx,
        xup = xlow + dx,
        ylow = func.fSave[bin],
        yup = func.fSave[bin + 1];

    if (!Number.isFinite(ylow) && (bin < np - 1)) {
       xlow += dx; xup += dx;
       ylow = yup; yup = func.fSave[bin + 2];
    } else if (!Number.isFinite(yup) && (bin > 0)) {
       xup -= dx; xlow -= dx;
       yup = ylow; ylow = func.fSave[bin - 1];
    }

    return ((xup * ylow - xlow * yup) + x * (yup - ylow)) / dx;
}

/** @summary Provide TF1 value
  * @desc First try evaluate, if not possible - check saved buffer
  * @private */
function getTF1Value(func, x, skip_eval = undefined) {
   let y = 0;
   if (!func)
      return 0;

   if (!skip_eval && !func.evalPar)
      proivdeEvalPar(func);

   if (func.evalPar) {
      try {
         y = func.evalPar(x);
         return y;
      } catch {
         y = 0;
      }
   }

   const np = func.fSave.length - 3;
   if ((np < 2) || (func.fSave[np + 1] === func.fSave[np + 2])) return 0;
   return _getTF1Save(func, x);
}

/** @summary Create log scale for axis bins
  * @private */
function produceTAxisLogScale(axis, num, min, max) {
   let lmin, lmax;

   if (max > 0) {
      lmax = Math.log(max);
      lmin = min > 0 ? Math.log(min) : lmax - 5;
   } else {
      lmax = -10;
      lmin = -15;
   }

   axis.fNbins = num;
   axis.fXbins = new Array(num + 1);
   for (let i = 0; i <= num; ++i)
      axis.fXbins[i] = Math.exp(lmin + i / num * (lmax - lmin));
   axis.fXmin = Math.exp(lmin);
   axis.fXmax = Math.exp(lmax);
}

/**
  * @summary Painter for TF1 object
  *
  * @private
  */

class TF1Painter extends TH1Painter {

   /** @summary Returns drawn object name */
   getObjectName() { return this.$func?.fName ?? 'func'; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.$func?._typename ?? clTF1; }

   /** @summary Returns true while function is drawn */
   isTF1() { return true; }

   /** @summary Returns primary function which was then drawn as histogram */
   getPrimaryObject() { return this.$func; }

   /** @summary Update function */
   updateObject(obj /*, opt */) {
      if (!obj || (this.getClassName() !== obj._typename)) return false;
      delete obj.evalPar;
      const histo = this.getHisto();

      if (this.webcanv_hist) {
         const h0 = this.getPadPainter()?.findInPrimitives('Func', clTH1D);
         if (h0) this.updateAxes(histo, h0, this.getFramePainter());
      }

      this.$func = obj;
      this.createTF1Histogram(obj, histo);
      this.scanContent();
      return true;
   }

   /** @summary Redraw TF1
     * @private */
   redraw(reason) {
      if (!this._use_saved_points && (reason === 'logx' || reason === 'zoom')) {
         this.createTF1Histogram(this.$func, this.getHisto());
         this.scanContent();
      }

      return super.redraw(reason);
   }

   /** @summary Create histogram for TF1 drawing
     * @private */
   createTF1Histogram(tf1, hist) {
      const fp = this.getFramePainter(),
            pad = this.getPadPainter()?.getRootPad(true),
            logx = pad?.fLogx,
            gr = fp?.getGrFuncs(this.second_x, this.second_y);
      let xmin = tf1.fXmin, xmax = tf1.fXmax;

      if (gr?.zoom_xmin !== gr?.zoom_xmax) {
         xmin = Math.min(xmin, gr.zoom_xmin);
         xmax = Math.max(xmax, gr.zoom_xmax);
      }

      this._use_saved_points = (tf1.fSave.length > 3) && (settings.PreferSavedPoints || this.force_saved);

      const ensureBins = num => {
         if (hist.fNcells !== num + 2) {
            hist.fNcells = num + 2;
            hist.fArray = new Float32Array(hist.fNcells);
         }
         hist.fArray.fill(0);
         hist.fXaxis.fNbins = num;
         hist.fXaxis.fXbins = [];
      };

      delete this._fail_eval;

      // this._use_saved_points = true;

      if (!this._use_saved_points) {
         const np = Math.max(tf1.fNpx, 100);
         let iserror = false;

         if (!tf1.evalPar && !proivdeEvalPar(tf1))
            iserror = true;

         ensureBins(np);

         if (logx)
            produceTAxisLogScale(hist.fXaxis, np, xmin, xmax);
          else {
            hist.fXaxis.fXmin = xmin;
            hist.fXaxis.fXmax = xmax;
         }

         for (let n = 0; (n < np) && !iserror; n++) {
            const x = hist.fXaxis.GetBinCenter(n + 1);
            let y = 0;
            try {
               y = tf1.evalPar(x);
            } catch (err) {
               iserror = true;
            }

            if (!iserror)
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
         }

         if (iserror)
            this._fail_eval = true;

         if (iserror && (tf1.fSave.length > 3))
            this._use_saved_points = true;
      }

      // in the case there were points have saved and we cannot calculate function
      // if we don't have the user's function
      if (this._use_saved_points) {
         const np = tf1.fSave.length - 3;
         let custom_xaxis = null;
         xmin = tf1.fSave[np + 1];
         xmax = tf1.fSave[np + 2];

         if (xmin === xmax) {
            // xmin = tf1.fSave[np];
            const mp = this.getMainPainter();
            if (isFunc(mp?.getHisto))
               custom_xaxis = mp?.getHisto()?.fXaxis;
         }

         if (custom_xaxis) {
            ensureBins(hist.fXaxis.fNbins);
            Object.assign(hist.fXaxis, custom_xaxis);
            // TODO: find first bin

            for (let n = 0; n < np; ++n) {
               const y = tf1.fSave[n];
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
            }
         } else {
            ensureBins(tf1.fNpx);
            hist.fXaxis.fXmin = tf1.fXmin;
            hist.fXaxis.fXmax = tf1.fXmax;

            for (let n = 0; n < tf1.fNpx; ++n) {
               const y = _getTF1Save(tf1, hist.fXaxis.GetBinCenter(n + 1));
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
            }
         }
      }

      hist.fName = 'Func';
      setHistogramTitle(hist, tf1.fTitle);
      hist.fMinimum = tf1.fMinimum;
      hist.fMaximum = tf1.fMaximum;
      hist.fLineColor = tf1.fLineColor;
      hist.fLineStyle = tf1.fLineStyle;
      hist.fLineWidth = tf1.fLineWidth;
      hist.fFillColor = tf1.fFillColor;
      hist.fFillStyle = tf1.fFillStyle;
      hist.fMarkerColor = tf1.fMarkerColor;
      hist.fMarkerStyle = tf1.fMarkerStyle;
      hist.fMarkerSize = tf1.fMarkerSize;
      hist.fBits |= kNoStats;
   }

   /** @summary Extract function ranges */
   extractAxesProperties(ndim) {
      super.extractAxesProperties(ndim);

      const func = this.$func, nsave = func?.fSave.length ?? 0;

      if (nsave > 3 && this._use_saved_points) {
         this.xmin = Math.min(this.xmin, func.fSave[nsave - 2]);
         this.xmax = Math.max(this.xmax, func.fSave[nsave - 1]);
      }
      if (func) {
         this.xmin = Math.min(this.xmin, func.fXmin);
         this.xmax = Math.max(this.xmax, func.fXmax);
      }
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      if ((this.$func?.fSave.length > 0) && this._use_saved_points && (axis === 'x')) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         const nb_points = this.$func.fNpx,
             xmin = this.$func.fSave[nb_points + 1],
             xmax = this.$func.fSave[nb_points + 2];

         return Math.abs(xmax - xmin) / nb_points < Math.abs(max - min);
      }

      // if function calculated, one always could zoom inside
      return (axis === 'x') || (axis === 'y');
   }

      /** @summary retrurn tooltips for TF2 */
   getTF1Tooltips(pnt) {
      delete this.$tmp_tooltip;
      const lines = [this.getObjectHint()],
            funcs = this.getFramePainter()?.getGrFuncs(this.options.second_x, this.options.second_y);

      if (!funcs || !isFunc(this.$func?.evalPar)) {
         lines.push('grx = ' + pnt.x, 'gry = ' + pnt.y);
         return lines;
      }

      const x = funcs.revertAxis('x', pnt.x);
      let y = 0, gry = 0, iserror = false;

       try {
          y = this.$func.evalPar(x);
          gry = Math.round(funcs.gry(y));
       } catch {
          iserror = true;
       }

      lines.push('x = ' + funcs.axisAsText('x', x),
                 'value = ' + (iserror ? '<fail>' : floatToString(y, gStyle.fStatFormat)));

      if (!iserror)
         this.$tmp_tooltip = { y, gry };
      return lines;
   }

   /** @summary process tooltip event for TF1 object */
   processTooltipEvent(pnt) {
      if (this._use_saved_points)
         return super.processTooltipEvent(pnt);

      let ttrect = this.draw_g?.selectChild('.tooltip_bin');

      if (!this.draw_g || !pnt) {
         ttrect?.remove();
         return null;
      }

      const res = { name: this.$func?.fName, title: this.$func?.fTitle,
                    x: pnt.x, y: pnt.y,
                    color1: this.lineatt?.color ?? 'green',
                    color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                    lines: this.getTF1Tooltips(pnt), exact: true, menu: true };

      if (pnt.disabled)
         ttrect.remove();
      else {
         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:circle')
                             .attr('class', 'tooltip_bin')
                             .style('pointer-events', 'none')
                             .style('fill', 'none')
                             .attr('r', (this.lineatt?.width ?? 1) + 4);
         }

         ttrect.attr('cx', pnt.x)
               .attr('cy', this.$tmp_tooltip.gry ?? pnt.y)
               .call(this.lineatt?.func);
      }

      return res;
   }

   /** @summary fill information for TWebCanvas
     * @private */
   fillWebObjectOptions(opt) {
      // mark that saved points are used or evaluation failed
      opt.fcust = this._fail_eval ? 'func_fail' : '';
   }

   /** @summary draw TF1 object */
   static async draw(dom, tf1, opt) {
     if (!isStr(opt)) opt = '';
      let p = opt.indexOf(';webcanv_hist'), webcanv_hist = false, force_saved = false;
      if (p >= 0) {
         webcanv_hist = true;
         opt = opt.slice(0, p);
      }
      p = opt.indexOf(';force_saved');
      if (p >= 0) {
         force_saved = true;
         opt = opt.slice(0, p);
      }

      let hist;

      if (webcanv_hist) {
         const dummy = new ObjectPainter(dom);
         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH1D);
      }

      if (!hist) {
         hist = createHistogram(clTH1D, 100);
         hist.fBits |= kNoStats;
      }

      if (!opt && getElementMainPainter(dom))
         opt = 'same';

      const painter = new TF1Painter(dom, hist);

      painter.$func = tf1;
      painter.webcanv_hist = webcanv_hist;
      painter.force_saved = force_saved;

      painter.createTF1Histogram(tf1, hist);

      return THistPainter._drawHist(painter, opt);
   }

} // class TF1Painter

export { TF1Painter, proivdeEvalPar, produceTAxisLogScale, getTF1Value };
