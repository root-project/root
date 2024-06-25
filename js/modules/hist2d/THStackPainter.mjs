import { clone, create, createHistogram, setHistogramTitle, BIT,
         gStyle, clTList, clTH1I, clTH2, clTH2I, kNoZoom, kNoStats } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter, EAxisBits } from '../base/ObjectPainter.mjs';
import { TH1Painter } from './TH1Painter.mjs';
import { TH2Painter } from './TH2Painter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


const kIsZoomed = BIT(16); // bit set when zooming on Y axis

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
      this.getPadPainter()?.cleanPrimitives(objp => { return (objp === this.firstpainter) || (this.painters.indexOf(objp) >= 0); });
      delete this.firstpainter;
      delete this.painters;
      super.cleanup();
   }

   /** @summary Build sum of all histograms
     * @desc Build a separate list fStack containing the running sum of all histograms */
   buildStack(stack) {
      if (!stack.fHists) return false;
      const nhists = stack.fHists.arr.length;
      if (nhists <= 0) return false;
      const lst = create(clTList);
      lst.Add(clone(stack.fHists.arr[0]), stack.fHists.opt[0]);
      for (let i = 1; i < nhists; ++i) {
         const hnext = clone(stack.fHists.arr[i]),
               hnextopt = stack.fHists.opt[i],
               hprev = lst.arr[i-1],
               xnext = hnext.fXaxis, xprev = hprev.fXaxis;

         let match = (xnext.fNbins === xprev.fNbins) &&
                     (xnext.fXmin === xprev.fXmin) &&
                     (xnext.fXmax === xprev.fXmax);

         if (!match && (xnext.fNbins > 0) && (xnext.fNbins < xprev.fNbins) && (xnext.fXmin === xprev.fXmin) &&
             (Math.abs((xnext.fXmax - xnext.fXmin)/xnext.fNbins - (xprev.fXmax - xprev.fXmin)/xprev.fNbins) < 0.0001)) {
            // simple extension of histogram to make sum
            const arr = new Array(hprev.fNcells).fill(0);
            for (let n = 1; n <= xnext.fNbins; ++n)
               arr[n] = hnext.fArray[n];
            hnext.fNcells = hprev.fNcells;
            Object.assign(xnext, xprev);
            hnext.fArray = arr;
            match = true;
         }
         if (!match) {
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
      const stack = this.getObject(),
            pad = this.getPadPainter()?.getRootPad(true),
            logscale = pad?.fLogv ?? (this.options.ndim === 1 ? pad?.fLogy : pad?.fLogz);
      let themin = 0, themax = 0;

      const getHistMinMax = (hist, witherr) => {
         const res = { min: 0, max: 0 };
         let domin = true, domax = true;
         if (hist.fMinimum !== kNoZoom) {
            res.min = hist.fMinimum;
            domin = false;
         }
         if (hist.fMaximum !== kNoZoom) {
            res.max = hist.fMaximum;
            domax = false;
         }

         if (!domin && !domax) return res;

         let i1 = 1, i2 = hist.fXaxis.fNbins, j1 = 1, j2 = 1, first = true;

         if (hist.fXaxis.TestBit(EAxisBits.kAxisRange)) {
            i1 = hist.fXaxis.fFirst;
            i2 = hist.fXaxis.fLast;
         }

         if (hist._typename.indexOf(clTH2) === 0) {
            j2 = hist.fYaxis.fNbins;
            if (hist.fYaxis.TestBit(EAxisBits.kAxisRange)) {
               j1 = hist.fYaxis.fFirst;
               j2 = hist.fYaxis.fLast;
            }
         }
         for (let j = j1; j <= j2; ++j) {
            for (let i = i1; i <= i2; ++i) {
               const val = hist.getBinContent(i, j),
                     err = witherr ? hist.getBinError(hist.getBin(i, j)) : 0;
               if (logscale && (val - err <= 0))
                  continue;
               if (domin && (first || (val - err < res.min)))
                  res.min = val - err;
               if (domax && (first || (val + err > res.max)))
                  res.max = val + err;
               first = false;
           }
         }

         return res;
      };

      if (this.options.nostack) {
         for (let i = 0; i < stack.fHists.arr.length; ++i) {
            const resh = getHistMinMax(stack.fHists.arr[i], iserr);
            if (i === 0) {
               themin = resh.min;
               themax = resh.max;
             } else {
               themin = Math.min(themin, resh.min);
               themax = Math.max(themax, resh.max);
            }
         }
      } else {
         // when stacked histogram drawn error is not used
         themin = getHistMinMax(stack.fStack.arr[0]).min;
         themax = getHistMinMax(stack.fStack.arr[stack.fStack.arr.length-1]).max;
      }

      if (logscale)
         themin = (themin > 0) ? themin*0.9 : themax*1e-3;
      else if (themin > 0)
         themin = 0;

      if (stack.fMaximum !== kNoZoom)
         themax = stack.fMaximum;

      if (stack.fMinimum !== kNoZoom)
         themin = stack.fMinimum;

      // redo code from THStack::BuildAndPaint

      if (!this.options.nostack || (stack.fMaximum === kNoZoom)) {
         if (logscale) {
            if (themin > 0)
               themax *= (1+0.2*Math.log10(themax/themin));
         } else if (stack.fMaximum === kNoZoom)
            themax *= (1 + gStyle.fHistTopMargin);
      }
      if (!this.options.nostack || (stack.fMinimum === kNoZoom)) {
         if (logscale)
            themin = (themin > 0) ? themin/(1+0.5*Math.log10(themax/themin)) : 1e-3*themax;
      }

      const res = { min: themin, max: themax, hopt: `hmin:${themin};hmax:${themax}` };
      if (stack.fHistogram?.TestBit(kIsZoomed))
         res.hopt += ';zoom_min_max';

      return res;
   }

   /** @summary Provide draw options for the histogram */
   getHistDrawOption(hist, opt) {
      let hopt = opt || hist.fOption || this.options.hopt;
      if (hopt.toUpperCase().indexOf(this.options.hopt) < 0)
         hopt += ' ' + this.options.hopt;
      if (this.options.draw_errors && !hopt)
         hopt = 'E';
      if (!this.options.pads)
         hopt += ' same nostat' + this.options.auto;
      return hopt;
   }

   /** @summary Draw next stack histogram */
   async drawNextHisto(indx, pad_painter) {
      const stack = this.getObject(),
            hlst = this.options.nostack ? stack.fHists : stack.fStack,
            nhists = hlst?.arr?.length || 0;

      if (indx >= nhists)
         return this;

      const rindx = this.options.horder ? indx : nhists-indx-1,
            subid = this.options.nostack ? `hists_${rindx}` : `stack_${rindx}`,
            hist = hlst.arr[rindx],
            hopt = this.getHistDrawOption(hist, hlst.opt[rindx]);

      // handling of 'pads' draw option
      if (pad_painter) {
         const subpad_painter = pad_painter.getSubPadPainter(indx+1);
         if (!subpad_painter)
            return this;

         subpad_painter.cleanPrimitives(true);

         return this.drawHist(subpad_painter, hist, hopt).then(subp => {
            if (subp) {
               subp.setSecondaryId(this, subid);
               this.painters.push(subp);
            }
            return this.drawNextHisto(indx+1, pad_painter);
         });
      }

      // special handling of stacked histograms - set $baseh object for correct drawing
      // also used to provide tooltips
      if ((rindx > 0) && !this.options.nostack)
         hist.$baseh = hlst.arr[rindx - 1];
      // this number used for auto colors creation
      if (this.options.auto)
         hist.$num_histos = nhists;

      return this.drawHist(this.getPadPainter(), hist, hopt).then(subp => {
          subp.setSecondaryId(this, subid);
          this.painters.push(subp);
          return this.drawNextHisto(indx+1, pad_painter);
      });
   }

   /** @summary Decode draw options of THStack painter */
   decodeOptions(opt) {
      if (!this.options) this.options = {};
      Object.assign(this.options, { ndim: 1, nostack: false, same: false, horder: true, has_errors: false, draw_errors: false, hopt: '', auto: '' });

      const stack = this.getObject(),
            hist = stack.fHistogram || (stack.fHists ? stack.fHists.arr[0] : null) || (stack.fStack ? stack.fStack.arr[0] : null),

       hasErrors = hist => {
         if (hist.fSumw2 && (hist.fSumw2.length > 0)) {
            for (let n = 0; n < hist.fSumw2.length; ++n)
               if (hist.fSumw2[n] > 0) return true;
         }
         return false;
      };

      if (hist && (hist._typename.indexOf(clTH2) === 0))
         this.options.ndim = 2;

      if ((this.options.ndim === 2) && !opt)
         opt = 'lego1';

      if (stack.fHists && !this.options.nostack) {
         for (let k = 0; k < stack.fHists.arr.length; ++k)
            this.options.has_errors = this.options.has_errors || hasErrors(stack.fHists.arr[k]);
      }

      this.options.nhist = stack.fHists?.arr?.length ?? 1;

      const d = new DrawOptions(opt);

      this.options.nostack = d.check('NOSTACK');
      if (d.check('STACK')) this.options.nostack = false;
      this.options.same = d.check('SAME');

      d.check('NOCLEAR'); // ignore noclear option

      ['PFC', 'PLC', 'PMC'].forEach(f => { if (d.check(f)) this.options.auto += ' ' + f; });

      this.options.pads = d.check('PADS');
      if (this.options.pads) this.options.nostack = true;

      this.options.hopt = d.remain(); // use remaining draw options for histogram draw

      const dolego = d.check('LEGO');

      this.options.errors = d.check('E');

      // if any histogram appears with pre-calculated errors, use E for all histograms
      if (!this.options.nostack && this.options.has_errors && !dolego && !d.check('HIST') && (this.options.hopt.indexOf('E') < 0))
         this.options.draw_errors = true;

      this.options.horder = this.options.nostack || dolego;
   }

   /** @summary Create main histogram for THStack axis drawing */
   createHistogram(stack) {
      const histos = stack.fHists,
            numhistos = histos ? histos.arr.length : 0;

      if (!numhistos) {
         const histo = createHistogram(clTH1I, 100);
         setHistogramTitle(histo, stack.fTitle);
         histo.fBits |= kNoStats;
         return histo;
      }

      const h0 = histos.arr[0],
            histo = createHistogram((this.options.ndim === 1) ? clTH1I : clTH2I, h0.fXaxis.fNbins, h0.fYaxis.fNbins);
      histo.fName = 'axis_hist';
      histo.fBits |= kNoStats;
      Object.assign(histo.fXaxis, h0.fXaxis);
      if (this.options.ndim === 2)
         Object.assign(histo.fYaxis, h0.fYaxis);

      // this code is not exists in ROOT painter, can be skipped?
      for (let n = 1; n < numhistos; ++n) {
         const h = histos.arr[n];

         if (!histo.fXaxis.fLabels) {
            histo.fXaxis.fXmin = Math.min(histo.fXaxis.fXmin, h.fXaxis.fXmin);
            histo.fXaxis.fXmax = Math.max(histo.fXaxis.fXmax, h.fXaxis.fXmax);
         }

         if ((this.options.ndim === 2) && !histo.fYaxis.fLabels) {
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

      const stack = this.getObject();

      stack.fHists = obj.fHists;
      stack.fStack = obj.fStack;
      stack.fTitle = obj.fTitle;
      stack.fMinimum = obj.fMinimum;
      stack.fMaximum = obj.fMaximum;

      if (!this.options.nostack)
         this.options.nostack = !this.buildStack(stack);

      if (this.firstpainter) {
         let src = obj.fHistogram;
         if (!src)
            src = stack.fHistogram = this.createHistogram(stack);

         const mm = this.getMinMax(this.options.errors || this.options.draw_errors);
         this.firstpainter.options.hmin = mm.min;
         this.firstpainter.options.hmax = mm.max;

         this.firstpainter._checked_zooming = false; // force to check 3d zooming

         if (this.options.ndim === 1) {
            this.firstpainter.ymin = mm.min;
            this.firstpainter.ymax = mm.max;
         } else {
            this.firstpainter.zmin = mm.min;
            this.firstpainter.zmax = mm.max;
         }

         this.firstpainter.updateObject(src);

         this.firstpainter.options.zoom_min_max = src.TestBit(kIsZoomed);
      }

      // and now update histograms
      const hlst = this.options.nostack ? stack.fHists : stack.fStack,
            nhists = hlst?.arr?.length ?? 0;

      if (nhists !== this.painters.length) {
         this.did_update = 1;
         this.getPadPainter()?.cleanPrimitives(objp => this.painters.indexOf(objp) >= 0);
         this.painters = [];
      } else {
         this.did_update = 2;
         for (let indx = 0; indx < nhists; ++indx) {
            const rindx = this.options.horder ? indx : nhists - indx - 1,
                  hist = hlst.arr[rindx];
            this.painters[indx].updateObject(hist, this.getHistDrawOption(hist, hlst.opt[rindx]));
         }
      }

      return true;
   }

   /** @summary Redraw THStack
     * @desc Do something if previous update had changed number of histograms */
   redraw(reason) {
      if (!this.did_update)
         return;

      const full_redraw = this.did_update === 1;
      delete this.did_update;

      const pr = this.firstpainter ? this.firstpainter.redraw(reason) : Promise.resolve(this);

      return pr.then(() => {
         if (full_redraw)
            return this.drawNextHisto(0, this.options.pads ? this.getPadPainter() : null);

         const redrawSub = indx => {
            if (indx >= this.painters.length)
               return this;
            return this.painters[indx].redraw(reason).then(() => redrawSub(indx+1));
         };
         return redrawSub(0);
      });
   }

   /** @summary Fill hstack context menu */
   fillContextMenuItems(menu) {
      menu.addRedrawMenu(this);
      if (!this.options.pads) {
         menu.addchk(this.options.draw_errors, 'Draw errors', flag => {
            this.options.draw_errors = flag;
            const stack = this.getObject(),
                  hlst = this.options.nostack ? stack.fHists : stack.fStack,
                  nhists = hlst?.arr?.length ?? 0;
            for (let indx = 0; indx < nhists; ++indx) {
               const rindx = this.options.horder ? indx : nhists - indx - 1,
                     hist = hlst.arr[rindx];
               this.painters[indx].decodeOptions(this.getHistDrawOption(hist, hlst.opt[rindx]));
            }
            this.redrawPad();
         }, 'Change draw erros in the stack');
      }
   }

   /** @summary Invoke histogram drawing */
   drawHist(dom, hist, hopt) {
      const func = (this.options.ndim === 1) ? TH1Painter.draw : TH2Painter.draw;
      return func(dom, hist, hopt);
   }

   /** @summary Full stack redraw with specified draw option */
   async redrawWith(opt, skip_cleanup) {
      const pp = this.getPadPainter();
      if (!skip_cleanup && pp) {
         this.firstpainter = null;
         this.painters = [];
         if (this.options.pads)
            pp.divide(0, 0);
         pp.removePrimitive(this, true);
      }

      this.decodeOptions(opt);

      const stack = this.getObject();

      let pr = Promise.resolve(this), pad_painter = null;

      if (this.options.pads) {
         pr = ensureTCanvas(this, false).then(() => {
            pad_painter = this.getPadPainter();
            return pad_painter.divide(this.options.nhist, 0, true);
         });
      } else {
         if (!this.options.nostack)
             this.options.nostack = !this.buildStack(stack);

         if (!this.options.same && stack.fHists?.arr.length) {
            if (!stack.fHistogram)
               stack.fHistogram = this.createHistogram(stack);

            const mm = this.getMinMax(this.options.errors || this.options.draw_errors),
                  hopt = this.options.hopt + ';' + mm.hopt;

            pr = this.drawHist(this.getDrawDom(), stack.fHistogram, hopt).then(subp => {
               this.firstpainter = subp;
               subp.setSecondaryId(this, 'hist'); // mark hist painter as created by hstack
            });
         }
      }

      return pr.then(() => this.drawNextHisto(0, pad_painter)).then(() => {
         if (!this.options.pads)
            this.addToPadPrimitives();
         return this;
      });
   }

   /** @summary draw THStack object in 2D */
   static async draw(dom, stack, opt) {
      if (!stack.fHists || !stack.fHists.arr)
         return null; // drawing not needed

      const painter = new THStackPainter(dom, stack, opt);

      return painter.redrawWith(opt, true);
   }

} // class THStackPainter

export { THStackPainter };
