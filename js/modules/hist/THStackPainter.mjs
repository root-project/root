import { clone, create, createHistogram, gStyle } from '../core.mjs';

import { DrawOptions } from '../base/BasePainter.mjs';

import { ObjectPainter } from '../base/ObjectPainter.mjs';

import { TH1Painter } from './TH1Painter.mjs';

import { TH2Painter } from './TH2Painter.mjs';

import { EAxisBits } from '../gpad/TAxisPainter.mjs';

import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

/**
 * @summary Painter class for THStack
 *
 * @private
 */

class THStackPainter extends ObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} stack - THStack object
     * @param {string} [opt] - draw options */
   constructor(dom, stack, opt) {
      super(dom, stack, opt);
      this.firstpainter = null;
      this.painters = []; // keep painters to be able update objects
   }

   /** @summary Cleanup THStack painter */
   cleanup() {
      let pp = this.getPadPainter();
      if (pp) pp.cleanPrimitives(objp => { return (objp === this.firstpainter) || (this.painters.indexOf(objp) >= 0); });
      delete this.firstpainter;
      delete this.painters;
      super.cleanup();
   }

   /** @summary Build sum of all histograms
     * @desc Build a separate list fStack containing the running sum of all histograms */
   buildStack(stack) {
      if (!stack.fHists) return false;
      let nhists = stack.fHists.arr.length;
      if (nhists <= 0) return false;
      let lst = create("TList");
      lst.Add(clone(stack.fHists.arr[0]), stack.fHists.opt[0]);
      for (let i = 1; i < nhists; ++i) {
         let hnext = clone(stack.fHists.arr[i]),
             hnextopt = stack.fHists.opt[i],
             hprev = lst.arr[i-1];

         if ((hnext.fNbins != hprev.fNbins) ||
             (hnext.fXaxis.fXmin != hprev.fXaxis.fXmin) ||
             (hnext.fXaxis.fXmax != hprev.fXaxis.fXmax)) {
            console.warn(`When drawing THStack, cannot sum-up histograms ${hnext.fName} and ${hprev.fName}`);
            lst.Clear();
            return false;
         }

         // trivial sum of histograms
         for (let n = 0; n < hnext.fArray.length; ++n)
            hnext.fArray[n] += hprev.fArray[n];

         lst.Add(hnext, hnextopt);
      }
      stack.fStack = lst;
      return true;
   }

   /** @summary Returns stack min/max values */
   getMinMax(iserr) {
      let res = { min: 0, max: 0 },
          stack = this.getObject(),
          pad = this.getPadPainter().getRootPad(true);

      const getHistMinMax = (hist, witherr) => {
         let res = { min: 0, max: 0 },
             domin = true, domax = true;
         if (hist.fMinimum !== -1111) {
            res.min = hist.fMinimum;
            domin = false;
         }
         if (hist.fMaximum !== -1111) {
            res.max = hist.fMaximum;
            domax = false;
         }

         if (!domin && !domax) return res;

         let i1 = 1, i2 = hist.fXaxis.fNbins, j1 = 1, j2 = 1, first = true;

         if (hist.fXaxis.TestBit(EAxisBits.kAxisRange)) {
            i1 = hist.fXaxis.fFirst;
            i2 = hist.fXaxis.fLast;
         }

         if (hist._typename.indexOf("TH2")===0) {
            j2 = hist.fYaxis.fNbins;
            if (hist.fYaxis.TestBit(EAxisBits.kAxisRange)) {
               j1 = hist.fYaxis.fFirst;
               j2 = hist.fYaxis.fLast;
            }
         }
         for (let j = j1; j <= j2; ++j)
            for (let i = i1; i <= i2; ++i) {
               let val = hist.getBinContent(i, j),
                   err = witherr ? hist.getBinError(hist.getBin(i,j)) : 0;
               if (domin && (first || (val-err < res.min))) res.min = val-err;
               if (domax && (first || (val+err > res.max))) res.max = val+err;
               first = false;
           }

         return res;
      };

      if (this.options.nostack) {
         for (let i = 0; i < stack.fHists.arr.length; ++i) {
            let resh = getHistMinMax(stack.fHists.arr[i], iserr);
            if (i == 0) {
               res = resh;
             } else {
               res.min = Math.min(res.min, resh.min);
               res.max = Math.max(res.max, resh.max);
            }
         }
      } else {
         res.min = getHistMinMax(stack.fStack.arr[0], iserr).min;
         res.max = getHistMinMax(stack.fStack.arr[stack.fStack.arr.length-1], iserr).max;
      }

      if (stack.fMaximum != -1111) res.max = stack.fMaximum;
      res.max *= (1 + gStyle.fHistTopMargin);
      if (stack.fMinimum != -1111) res.min = stack.fMinimum;

      if (pad && (this.options.ndim == 1 ? pad.fLogy : pad.fLogz)) {
         if (res.max<=0) res.max = 1;
         if (res.min<=0) res.min = 1e-4*res.max;
         let kmin = 1/(1 + 0.5*Math.log10(res.max / res.min)),
             kmax = 1 + 0.2*Math.log10(res.max / res.min);
         res.min *= kmin;
         res.max *= kmax;
      } else {
         if ((res.min>0) && (res.min < 0.05*res.max)) res.min = 0;
      }

      return res;
   }

   /** @summary Draw next stack histogram */
   drawNextHisto(indx, pad_painter) {

      let stack = this.getObject(),
          hlst = this.options.nostack ? stack.fHists : stack.fStack,
          nhists = (hlst && hlst.arr) ? hlst.arr.length : 0;

      if (indx >= nhists)
         return Promise.resolve(this);

      let rindx = this.options.horder ? indx : nhists-indx-1,
          hist = hlst.arr[rindx],
          hopt = hlst.opt[rindx] || hist.fOption || this.options.hopt;

      if (hopt.toUpperCase().indexOf(this.options.hopt) < 0)
         hopt += ' ' + this.options.hopt;
      if (this.options.draw_errors && !hopt)
         hopt = "E";

      if (this.options._pfc || this.options._plc || this.options._pmc) {
         let mp = this.getMainPainter();
         if (mp && mp.createAutoColor) {
            let icolor = mp.createAutoColor(nhists);
            if (this.options._pfc) hist.fFillColor = icolor;
            if (this.options._plc) hist.fLineColor = icolor;
            if (this.options._pmc) hist.fMarkerColor = icolor;
         }
      }

      // handling of "pads" draw option
      if (pad_painter) {
         let subpad_painter = pad_painter.getSubPadPainter(indx+1);
         if (!subpad_painter)
            return Promise.resolve(this);

         let prev_name = subpad_painter.selectCurrentPad(subpad_painter.this_pad_name);

         return this.hdraw_func(subpad_painter.getDom(), hist, hopt).then(subp => {
            this.painters.push(subp);
            subpad_painter.selectCurrentPad(prev_name);
            return this.drawNextHisto(indx+1, pad_painter);
         });
      }

      // special handling of stacked histograms - set $baseh object for correct drawing
      // also used to provide tooltips
      if ((rindx > 0) && !this.options.nostack)
         hist.$baseh = hlst.arr[rindx - 1];

      return this.hdraw_func(this.getDom(), hist, hopt + " same nostat").then(subp => {
          this.painters.push(subp);
          return this.drawNextHisto(indx+1, pad_painter);
      });
   }

   /** @summary Decode draw options of THStack painter */
   decodeOptions(opt) {
      if (!this.options) this.options = {};
      Object.assign(this.options, { ndim: 1, nostack: false, same: false, horder: true, has_errors: false, draw_errors: false, hopt: "" });

      let stack = this.getObject(),
          hist = stack.fHistogram || (stack.fHists ? stack.fHists.arr[0] : null) || (stack.fStack ? stack.fStack.arr[0] : null);

      const hasErrors = hist => {
         if (hist.fSumw2 && (hist.fSumw2.length > 0))
            for (let n = 0;n < hist.fSumw2.length; ++n)
               if (hist.fSumw2[n] > 0) return true;
         return false;
      };

      if (hist && (hist._typename.indexOf("TH2")==0)) this.options.ndim = 2;

      if ((this.options.ndim == 2) && !opt) opt = "lego1";

      if (stack.fHists && !this.options.nostack)
         for (let k = 0; k < stack.fHists.arr.length; ++k)
            this.options.has_errors = this.options.has_errors || hasErrors(stack.fHists.arr[k]);

      this.options.nhist = stack.fHists ? stack.fHists.arr.length : 1;

      let d = new DrawOptions(opt);

      this.options.nostack = d.check("NOSTACK");
      if (d.check("STACK")) this.options.nostack = false;
      this.options.same = d.check("SAME");

      d.check("NOCLEAR"); // ignore noclear option

      this.options._pfc = d.check("PFC");
      this.options._plc = d.check("PLC");
      this.options._pmc = d.check("PMC");

      this.options.pads = d.check("PADS");
      if (this.options.pads) this.options.nostack = true;

      this.options.hopt = d.remain(); // use remaining draw options for histogram draw

      let dolego = d.check("LEGO");

      this.options.errors = d.check("E");

      // if any histogram appears with pre-calculated errors, use E for all histograms
      if (!this.options.nostack && this.options.has_errors && !dolego && !d.check("HIST") && (this.options.hopt.indexOf("E")<0)) this.options.draw_errors = true;

      this.options.horder = this.options.nostack || dolego;
   }

   /** @summary Create main histogram for THStack axis drawing */
   createHistogram(stack) {
      let histos = stack.fHists,
          numhistos = histos ? histos.arr.length : 0;

      if (!numhistos) {
         let histo = createHistogram("TH1I", 100);
         histo.fTitle = stack.fTitle;
         return histo;
      }

      let h0 = histos.arr[0],
          histo = createHistogram((this.options.ndim==1) ? "TH1I" : "TH2I", h0.fXaxis.fNbins, h0.fYaxis.fNbins);
      histo.fName = "axis_hist";
      Object.assign(histo.fXaxis, h0.fXaxis);
      if (this.options.ndim==2)
         Object.assign(histo.fYaxis, h0.fYaxis);

      // this code is not exists in ROOT painter, can be skipped?
      for (let n=1;n<numhistos;++n) {
         let h = histos.arr[n];

         if (!histo.fXaxis.fLabels) {
            histo.fXaxis.fXmin = Math.min(histo.fXaxis.fXmin, h.fXaxis.fXmin);
            histo.fXaxis.fXmax = Math.max(histo.fXaxis.fXmax, h.fXaxis.fXmax);
         }

         if ((this.options.ndim==2) && !histo.fYaxis.fLabels) {
            histo.fYaxis.fXmin = Math.min(histo.fYaxis.fXmin, h.fYaxis.fXmin);
            histo.fYaxis.fXmax = Math.max(histo.fYaxis.fXmax, h.fYaxis.fXmax);
         }
      }

      histo.fTitle = stack.fTitle;

      return histo;
   }

   /** @summary Update thstack object */
   updateObject(obj) {
      if (!this.matchObjectType(obj)) return false;

      let stack = this.getObject();

      stack.fHists = obj.fHists;
      stack.fStack = obj.fStack;
      stack.fTitle = obj.fTitle;

      if (!this.options.nostack)
         this.options.nostack = !this.buildStack(stack);

      if (this.firstpainter) {
         let src = obj.fHistogram;
         if (!src)
            src = stack.fHistogram = this.createHistogram(stack);

         this.firstpainter.updateObject(src);

         let mm = this.getMinMax(this.options.errors || this.options.draw_errors);

         this.firstpainter.options.minimum = mm.min;
         this.firstpainter.options.maximum = mm.max;

         if (this.options.ndim == 1) {
            this.firstpainter.ymin = mm.min;
            this.firstpainter.ymax = mm.max;
         }
      }

      // and now update histograms
      let hlst = this.options.nostack ? stack.fHists : stack.fStack,
          nhists = (hlst && hlst.arr) ? hlst.arr.length : 0;

      if (nhists !== this.painters.length) {
         let pp = this.getPadPainter();
         if (pp) pp.cleanPrimitives(objp => { return this.painters.indexOf(objp) >= 0; });
         this.painters = [];
         this.did_update = true;
      } else {
         for (let indx = 0; indx < nhists; ++indx) {
            let rindx = this.options.horder ? indx : nhists-indx-1;
            let hist = hlst.arr[rindx];
            this.painters[indx].updateObject(hist);
         }
      }

      return true;
   }

   /** @summary Redraw THStack
     * @desc Do something if previous update had changed number of histograms */
   redraw() {
      if (this.did_update) {
         delete this.did_update;
         return this.drawNextHisto(0, this.options.pads ? this.getPadPainter() : null);
      }
   }

   /** @summary draw THStack object */
   static draw(dom, stack, opt) {
      if (!stack.fHists || !stack.fHists.arr)
         return null; // drawing not needed

      let painter = new THStackPainter(dom, stack, opt),
          pad_painter = null,
          skip_drawing = false;

      return ensureTCanvas(painter, false).then(() => {

         painter.decodeOptions(opt);

         painter.hdraw_func = (painter.options.ndim == 1) ? TH1Painter.draw : TH2Painter.draw;

         if (painter.options.pads) {
            pad_painter = painter.getPadPainter();
            if (pad_painter.doingDraw() && pad_painter.pad && pad_painter.pad.fPrimitives &&
                pad_painter.pad.fPrimitives.arr.length > 1 && (pad_painter.pad.fPrimitives.arr.indexOf(stack)==0)) {
               skip_drawing = true;
               console.log('special case with THStack with is already rendered - do nothing');
               return;
            }

            pad_painter.cleanPrimitives(p => p !== painter);
            return pad_painter.divide(painter.options.nhist);
         }

         if (!painter.options.nostack)
             painter.options.nostack = !painter.buildStack(stack);

         if (painter.options.same) return;

         if (!stack.fHistogram)
             stack.fHistogram = painter.createHistogram(stack);

         let mm = painter.getMinMax(painter.options.errors || painter.options.draw_errors),
             hopt = painter.options.hopt + " axis";

         if (mm) hopt += ";minimum:" + mm.min + ";maximum:" + mm.max;

         return painter.hdraw_func(dom, stack.fHistogram, hopt).then(subp => {
            painter.firstpainter = subp;
         });
      }).then(() => skip_drawing ? painter : painter.drawNextHisto(0, pad_painter));
   }

} // class THStackPainter

export { THStackPainter }
