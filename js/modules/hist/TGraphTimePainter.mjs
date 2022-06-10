import { internals } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import { draw } from '../draw.mjs';


/**
 * @summary Painter for TGraphTime object
 *
 * @private
 */

class TGraphTimePainter extends ObjectPainter {

   /** @summary Redraw object */
   redraw() {
      if (this.step === undefined) this.startDrawing();
   }

   /** @summary Decode drawing options */
   decodeOptions(opt) {

      let d = new DrawOptions(opt || "REPEAT");

      if (!this.options) this.options = {};

      Object.assign(this.options, {
          once: d.check("ONCE"),
          repeat: d.check("REPEAT"),
          first: d.check("FIRST")
      });

      this.storeDrawOpt(opt);
   }

   /** @summary Draw primitives */
   drawPrimitives(indx) {

      if (!indx) {
         indx = 0;
         this._doing_primitives = true;
      }

      let lst = this.getObject().fSteps.arr[this.step];

      if (!lst || (indx >= lst.arr.length)) {
         delete this._doing_primitives;
         return Promise.resolve();
      }

      return draw(this.getDom(), lst.arr[indx], lst.opt[indx]).then(ppainter => {

         if (ppainter) ppainter.$grtimeid = this.selfid; // indicator that painter created by ourself
         return this.drawPrimitives(indx+1);

      });
   }

   /** @summary Continue drawing */
   continueDrawing() {
      if (!this.options) return;

      let gr = this.getObject();

      if (this.options.first) {
         // draw only single frame, cancel all others
         delete this.step;
         return;
      }

      if (this.wait_animation_frame) {
         delete this.wait_animation_frame;

         // clear pad
         let pp = this.getPadPainter();
         if (!pp) {
            // most probably, pad is cleared
            delete this.step;
            return;
         }

         // clear primitives produced by the TGraphTime
         pp.cleanPrimitives(p => (p.$grtimeid === this.selfid));

         // draw ptrimitives again
         this.drawPrimitives().then(() => this.continueDrawing());
      } else if (this.running_timeout) {
         clearTimeout(this.running_timeout);
         delete this.running_timeout;

         this.wait_animation_frame = true;
         // use animation frame to disable update in inactive form
         requestAnimationFrame(() => this.continueDrawing());
      } else {

         let sleeptime = gr.fSleepTime;
         if (!sleeptime || (sleeptime<100)) sleeptime = 10;

         if (++this.step > gr.fSteps.arr.length) {
            if (this.options.repeat) {
               this.step = 0; // start again
               sleeptime = Math.max(5000, 5*sleeptime); // increase sleep time
            } else {
               delete this.step;    // clear indicator that animation running
               return;
            }
         }

         this.running_timeout = setTimeout(() => this.continueDrawing(), sleeptime);
      }
   }

   /** @ummary Start drawing of graph time */
   startDrawing() {
      this.step = 0;

      return this.drawPrimitives().then(() => {
         this.continueDrawing();
         return this;
      });
   }

   /** @summary Draw TGraphTime object */
   static draw(dom, gr, opt) {
      if (!gr.fFrame) {
        console.error('Frame histogram not exists');
        return null;
      }

      let painter = new TGraphTimePainter(dom, gr);

      if (painter.getMainPainter()) {
         console.error('Cannot draw graph time on top of other histograms');
         return null;
      }

      painter.decodeOptions(opt);

      if (!gr.fFrame.fTitle && gr.fTitle)
         gr.fFrame.fTitle = gr.fTitle;

      painter.selfid = "grtime" + internals.id_counter++; // use to identify primitives which should be clean

      return TH1Painter.draw(dom, gr.fFrame, "AXIS").then(() => {
         painter.addToPadPrimitives();
         return painter.startDrawing();
      });
   }

} // class TGraphTimePainter

export { TGraphTimePainter };
