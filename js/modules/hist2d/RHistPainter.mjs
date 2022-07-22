import { gStyle, settings } from '../core.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';


/** @summary assign methods for the RAxis objects
  * @private */
function assignRAxisMethods(axis) {
   if ((axis._typename == "ROOT::Experimental::RAxisEquidistant") || (axis._typename == "ROOT::Experimental::RAxisLabels")) {
      if (axis.fInvBinWidth === 0) {
         axis.$dummy = true;
         axis.fInvBinWidth = 1;
         axis.fNBinsNoOver = 0;
         axis.fLow = 0;
      }

      axis.min = axis.fLow;
      axis.max = axis.fLow + axis.fNBinsNoOver/axis.fInvBinWidth;
      axis.GetNumBins = function() { return this.fNBinsNoOver; }
      axis.GetBinCoord = function(bin) { return this.fLow + bin/this.fInvBinWidth; }
      axis.FindBin = function(x,add) { return Math.floor((x - this.fLow)*this.fInvBinWidth + add); }
   } else if (axis._typename == "ROOT::Experimental::RAxisIrregular") {
      axis.min = axis.fBinBorders[0];
      axis.max = axis.fBinBorders[axis.fBinBorders.length - 1];
      axis.GetNumBins = function() { return this.fBinBorders.length; }
      axis.GetBinCoord = function(bin) {
         let indx = Math.round(bin);
         if (indx <= 0) return this.fBinBorders[0];
         if (indx >= this.fBinBorders.length) return this.fBinBorders[this.fBinBorders.length - 1];
         if (indx==bin) return this.fBinBorders[indx];
         let indx2 = (bin < indx) ? indx - 1 : indx + 1;
         return this.fBinBorders[indx] * Math.abs(bin-indx2) + this.fBinBorders[indx2] * Math.abs(bin-indx);
      }
      axis.FindBin = function(x,add) {
         for (let k = 1; k < this.fBinBorders.length; ++k)
            if (x < this.fBinBorders[k]) return Math.floor(k-1+add);
         return this.fBinBorders.length - 1;
      }
   }

   // to support some code from ROOT6 drawing

   axis.GetBinCenter = function(bin) { return this.GetBinCoord(bin-0.5); }
   axis.GetBinLowEdge = function(bin) { return this.GetBinCoord(bin-1); }
}

/** @summary Returns real histogram impl
  * @private */
function getHImpl(obj) {
   return (obj && obj.fHistImpl) ? obj.fHistImpl.fIO : null;
}


/** @summary Base painter class for RHist objects
 *
 * @private
 */

class RHistPainter extends RObjectPainter {

   /** @summary Constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} histo - RHist object */
   constructor(dom, histo) {
      super(dom, histo);
      this.csstype = "hist";
      this.draw_content = true;
      this.nbinsx = 0;
      this.nbinsy = 0;
      this.accept_drops = true; // indicate that one can drop other objects like doing Draw("same")
      this.mode3d = false;

      // initialize histogram methods
      this.getHisto(true);
   }

   /** @summary Returns true if RHistDisplayItem is used */
   isDisplayItem() {
      let obj = this.getObject();
      return obj && obj.fAxes ? true : false;
   }

   /** @summary get histogram */
   getHisto(force) {
      let obj = this.getObject(), histo = getHImpl(obj);

      if (histo && (!histo.getBinContent || force)) {
         if (histo.fAxes._2) {
            assignRAxisMethods(histo.fAxes._0);
            assignRAxisMethods(histo.fAxes._1);
            assignRAxisMethods(histo.fAxes._2);
            histo.getBin = function(x, y, z) { return (x-1) + this.fAxes._0.GetNumBins()*(y-1) + this.fAxes._0.GetNumBins()*this.fAxes._1.GetNumBins()*(z-1); }
            // FIXME: all normal ROOT methods uses indx+1 logic, but RHist has no underflow/overflow bins now
            histo.getBinContent = function(x, y, z) { return this.fStatistics.fBinContent[this.getBin(x, y, z)]; }
            histo.getBinError = function(x, y, z) {
               let bin = this.getBin(x, y, z);
               if (this.fStatistics.fSumWeightsSquared)
                  return Math.sqrt(this.fStatistics.fSumWeightsSquared[bin]);
               return Math.sqrt(Math.abs(this.fStatistics.fBinContent[bin]));
            }
         } else if (histo.fAxes._1) {
            assignRAxisMethods(histo.fAxes._0);
            assignRAxisMethods(histo.fAxes._1);
            histo.getBin = function(x, y) { return (x-1) + this.fAxes._0.GetNumBins()*(y-1); }
            // FIXME: all normal ROOT methods uses indx+1 logic, but RHist has no underflow/overflow bins now
            histo.getBinContent = function(x, y) { return this.fStatistics.fBinContent[this.getBin(x, y)]; }
            histo.getBinError = function(x, y) {
               let bin = this.getBin(x, y);
               if (this.fStatistics.fSumWeightsSquared)
                  return Math.sqrt(this.fStatistics.fSumWeightsSquared[bin]);
               return Math.sqrt(Math.abs(this.fStatistics.fBinContent[bin]));
            }
         } else {
            assignRAxisMethods(histo.fAxes._0);
            histo.getBin = function(x) { return x-1; }
            // FIXME: all normal ROOT methods uses indx+1 logic, but RHist has no underflow/overflow bins now
            histo.getBinContent = function(x) { return this.fStatistics.fBinContent[x-1]; }
            histo.getBinError = function(x) {
               if (this.fStatistics.fSumWeightsSquared)
                  return Math.sqrt(this.fStatistics.fSumWeightsSquared[x-1]);
               return Math.sqrt(Math.abs(this.fStatistics.fBinContent[x-1]));
            }
         }
      } else if (!histo && obj && obj.fAxes) {
         // case of RHistDisplayItem

         histo = obj;

         if (!histo.getBinContent || force) {
            if (histo.fAxes.length == 3) {
               assignRAxisMethods(histo.fAxes[0]);
               assignRAxisMethods(histo.fAxes[1]);
               assignRAxisMethods(histo.fAxes[2]);

               histo.nx = histo.fIndicies[1] - histo.fIndicies[0];
               histo.dx = histo.fIndicies[0] + 1;
               histo.stepx = histo.fIndicies[2];

               histo.ny = histo.fIndicies[4] - histo.fIndicies[3];
               histo.dy = histo.fIndicies[3] + 1;
               histo.stepy = histo.fIndicies[5];

               histo.nz = histo.fIndicies[7] - histo.fIndicies[6];
               histo.dz = histo.fIndicies[6] + 1;
               histo.stepz = histo.fIndicies[8];

               // this is index in original histogram
               histo.getBin = function(x, y, z) { return (x-1) + this.fAxes[0].GetNumBins()*(y-1) + this.fAxes[0].GetNumBins()*this.fAxes[1].GetNumBins()*(z-1); }

               // this is index in current available data
               if ((histo.stepx > 1) || (histo.stepy > 1) || (histo.stepz > 1))
                  histo.getBin0 = function(x, y, z) { return Math.floor((x-this.dx)/this.stepx) + this.nx/this.stepx*Math.floor((y-this.dy)/this.stepy) + this.nx/this.stepx*this.ny/this.stepy*Math.floor((z-this.dz)/this.stepz); }
               else
                  histo.getBin0 = function(x, y, z) { return (x-this.dx) + this.nx*(y-this.dy) + this.nx*this.ny*(z-dz); }

               histo.getBinContent = function(x, y, z) { return this.fBinContent[this.getBin0(x, y, z)]; }
               histo.getBinError = function(x, y, z) { return Math.sqrt(Math.abs(this.getBinContent(x, y, z))); }


            } else if (histo.fAxes.length == 2) {
               assignRAxisMethods(histo.fAxes[0]);
               assignRAxisMethods(histo.fAxes[1]);

               histo.nx = histo.fIndicies[1] - histo.fIndicies[0];
               histo.dx = histo.fIndicies[0] + 1;
               histo.stepx = histo.fIndicies[2];

               histo.ny = histo.fIndicies[4] - histo.fIndicies[3];
               histo.dy = histo.fIndicies[3] + 1;
               histo.stepy = histo.fIndicies[5];

               // this is index in original histogram
               histo.getBin = function(x, y) { return (x-1) + this.fAxes[0].GetNumBins()*(y-1); }

               // this is index in current available data
               if ((histo.stepx > 1) || (histo.stepy > 1))
                  histo.getBin0 = function(x, y) { return Math.floor((x-this.dx)/this.stepx) + this.nx/this.stepx*Math.floor((y-this.dy)/this.stepy); }
               else
                  histo.getBin0 = function(x, y) { return (x-this.dx) + this.nx*(y-this.dy); }

               histo.getBinContent = function(x, y) { return this.fBinContent[this.getBin0(x, y)]; }
               histo.getBinError = function(x, y) { return Math.sqrt(Math.abs(this.getBinContent(x, y))); }
            } else {
               assignRAxisMethods(histo.fAxes[0]);
               histo.nx = histo.fIndicies[1] - histo.fIndicies[0];
               histo.dx = histo.fIndicies[0] + 1;
               histo.stepx = histo.fIndicies[2];

               histo.getBin = function(x) { return x-1; }
               if (histo.stepx > 1)
                  histo.getBin0 = function(x) { return Math.floor((x-this.dx)/this.stepx); }
               else
                  histo.getBin0 = function(x) { return x-this.dx; }
               histo.getBinContent = function(x) { return this.fBinContent[this.getBin0(x)]; }
               histo.getBinError = function(x) { return Math.sqrt(Math.abs(this.getBinContent(x))); }
            }
         }
      }
      return histo;
   }

   /** @summary Decode options */
   decodeOptions(/*opt*/) {
      if (!this.options) this.options = { Hist : 1 };
   }

   /** @summary Copy draw options from other painter */
   copyOptionsFrom(src) {
      if (src === this) return;
      let o = this.options, o0 = src.options;
      o.Mode3D = o0.Mode3D;
   }

   /** @summary copy draw options to all other histograms in the pad*/
   copyOptionsToOthers() {
      this.forEachPainter(painter => {
         if ((painter !== this) && (typeof painter.copyOptionsFrom == 'function'))
            painter.copyOptionsFrom(this);
      }, "objects");
   }

   /** @summary Clear 3d drawings - if any */
   clear3DScene() {
      let fp = this.getFramePainter();
      if (fp && typeof fp.create3DScene === 'function')
         fp.create3DScene(-1);
      this.mode3d = false;
   }

   /** @summary Cleanup hist painter */
   cleanup() {
      this.clear3DScene();

      delete this.options;

      super.cleanup();
   }

   /** @summary Returns histogram dimension */
   getDimension() { return 1; }

   /** @summary Scan histogram content
     * @abstract */
   scanContent(/*when_axis_changed*/) {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed
   }

   /** @summary Draw axes */
   drawFrameAxes() {
      // return true when axes was drawn
      let main = this.getFramePainter();
      if (!main)
         return Promise.resolve(false);

      if (!this.draw_content)
         return Promise.resolve(true);

      if (!this.isMainPainter()) {
         if (!this.options.second_x && !this.options.second_y)
            return Promise.resolve(true);

         main.setAxes2Ranges(this.options.second_x, this.getAxis("x"), this.xmin, this.xmax, this.options.second_y, this.getAxis("y"), this.ymin, this.ymax);
         return main.drawAxes2(this.options.second_x, this.options.second_y);
      }

      main.cleanupAxes();
      main.xmin = main.xmax = 0;
      main.ymin = main.ymax = 0;
      main.zmin = main.zmax = 0;
      main.setAxesRanges(this.getAxis("x"), this.xmin, this.xmax, this.getAxis("y"), this.ymin, this.ymax, this.getAxis("z"), this.zmin, this.zmax);
      return main.drawAxes();
   }

   /** @summary create attributes */
   createHistDrawAttributes() {
      this.createv7AttFill();
      this.createv7AttLine();
   }

   /** @summary update display item */
   updateDisplayItem(obj, src) {
      if (!obj || !src) return false;

      obj.fAxes = src.fAxes;
      obj.fIndicies = src.fIndicies;
      obj.fBinContent = src.fBinContent;
      obj.fContMin = src.fContMin;
      obj.fContMinPos = src.fContMinPos;
      obj.fContMax = src.fContMax;

      // update histogram attributes
      this.getHisto(true);

      return true;
   }

   /** @summary update histogram object */
   updateObject(obj /*, opt*/) {

      let origin = this.getObject();

      if (obj !== origin) {

         if (!this.matchObjectType(obj)) return false;

         if (this.isDisplayItem()) {

            this.updateDisplayItem(origin, obj);

         } else {

            let horigin = getHImpl(origin),
                hobj = getHImpl(obj);

            if (!horigin || !hobj) return false;

            // make it easy - copy statistics without axes
            horigin.fStatistics = hobj.fStatistics;

            origin.fTitle = obj.fTitle;
         }
      }

      this.scanContent();

      this.histogram_updated = true; // indicate that object updated

      return true;
   }

   /** @summary Get axis object */
   getAxis(name) {
      let histo = this.getHisto(), obj = this.getObject(), axis = null;

      if (obj && obj.fAxes) {
         switch(name) {
            case "x": axis = obj.fAxes[0]; break;
            case "y": axis = obj.fAxes[1]; break;
            case "z": axis = obj.fAxes[2]; break;
            default: axis = obj.fAxes[0]; break;
         }
      } else if (histo && histo.fAxes) {
         // console.log('histo fAxes', histo.fAxes, histo.fAxes._0)
         switch(name) {
            case "x": axis = histo.fAxes._0; break;
            case "y": axis = histo.fAxes._1; break;
            case "z": axis = histo.fAxes._2; break;
            default: axis = histo.fAxes._0; break;
         }
      }

      if (axis && !axis.GetBinCoord)
         assignRAxisMethods(axis);

      return axis;
   }

   /** @summary Get tip text for axis bin */
   getAxisBinTip(name, bin, step) {
      let pmain = this.getFramePainter(),
          handle = pmain[name+"_handle"],
          axis = this.getAxis(name),
          x1 = axis.GetBinCoord(bin);

      if (handle.kind === 'labels')
         return pmain.axisAsText(name, x1);

      let x2 = axis.GetBinCoord(bin+(step || 1));

      if (handle.kind === 'time')
         return pmain.axisAsText(name, (x1+x2)/2);

      return "[" + pmain.axisAsText(name, x1) + ", " + pmain.axisAsText(name, x2) + ")";
   }

   /** @summary Extract axes ranges and bins numbers
     * @desc Also here ensured that all axes objects got their necessary methods */
   extractAxesProperties(ndim) {

      let histo = this.getHisto();
      if (!histo) return;

      this.nbinsx = this.nbinsy = this.nbinsz = 0;

      let axis = this.getAxis("x");
      this.nbinsx = axis.GetNumBins();
      this.xmin = axis.min;
      this.xmax = axis.max;

      if (ndim < 2) return;
      axis = this.getAxis("y");
      this.nbinsy = axis.GetNumBins();
      this.ymin = axis.min;
      this.ymax = axis.max;

      if (ndim < 3) return;
      axis = this.getAxis("z");
      this.nbinsz = axis.GetNumBins();
      this.zmin = axis.min;
      this.zmax = axis.max;
   }

   /** @summary Add interactive features, only main painter does it */
   addInteractivity() {
      // only first painter in list allowed to add interactive functionality to the frame

      let ismain =  this.isMainPainter(),
          second_axis = this.options.second_x || this.options.second_y,
          fp = ismain || second_axis ? this.getFramePainter() : null;
      return fp ? fp.addInteractivity(!ismain && second_axis) : true;
   }

   /** @summary Process item reply */
   processItemReply(reply, req) {
      if (!this.isDisplayItem())
         return console.error('Get item when display normal histogram');

      if (req.reqid === this.current_item_reqid) {

         if (reply !== null) {
            this.updateDisplayItem(this.getObject(), reply.item);
         }

         req.resolveFunc(true);
      }
   }

   /** @summary Special method to request bins from server if existing data insufficient
     * @returns {Promise} when ready */
   drawingBins(reason) {

      let is_axes_zoomed = false;
      if (reason && (typeof reason == "string") && (reason.indexOf("zoom") == 0)) {
         if (reason.indexOf("0") > 0) is_axes_zoomed = true;
         if ((this.getDimension() > 1) && (reason.indexOf("1") > 0)) is_axes_zoomed = true;
         if ((this.getDimension() > 2) && (reason.indexOf("2") > 0)) is_axes_zoomed = true;
      }

      if (this.isDisplayItem() && is_axes_zoomed && this.v7NormalMode()) {

         let handle = this.prepareDraw({ only_indexes: true });

         // submit request if histogram data not enough for display
         if (handle.incomplete)
            return new Promise(resolveFunc => {
               // use empty kind to always submit request
               let req = this.v7SubmitRequest("", { _typename: "ROOT::Experimental::RHistDrawableBase::RRequest" },
                                                  this.processItemReply.bind(this));
               if (req) {
                  this.current_item_reqid = req.reqid; // ignore all previous requests, only this one will be processed
                  req.resolveFunc = resolveFunc;
                  setTimeout(this.processItemReply.bind(this, null, req), 1000); // after 1 s draw something that we can
               } else {
                  resolveFunc(true);
               }
            });
      }

      return Promise.resolve(true);
   }

   /** @summary Toggle stat box drawing
     * @desc Not yet implemented */
   toggleStat(/*arg*/) {
   }

   /** @summary get selected index for axis */
   getSelectIndex(axis, size, add) {
      // be aware - here indexes starts from 0
      let indx = 0,
          taxis = this.getAxis(axis),
          nbins = this['nbins'+axis] || 0;

      if (this.options.second_x && axis == "x") axis = "x2";
      if (this.options.second_y && axis == "y") axis = "y2";

      let main = this.getFramePainter(),
          min = main ? main['zoom_' + axis + 'min'] : 0,
          max = main ? main['zoom_' + axis + 'max'] : 0;

      if ((min !== max) && taxis) {
         if (size == "left")
            indx = taxis.FindBin(min, add || 0);
         else
            indx = taxis.FindBin(max, (add || 0) + 0.5);
         if (indx<0) indx = 0; else if (indx>nbins) indx = nbins;
      } else {
         indx = (size == "left") ? 0 : nbins;
      }

      return indx;
   }

   /** @summary Auto zoom into histogram non-empty range
     * @abstract */
   autoZoom() {}

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      // TODO: move to frame painter
      switch(funcname) {
         case "ToggleZoom":
            if ((this.zoom_xmin !== this.zoom_xmax) || (this.zoom_ymin !== this.zoom_ymax) || (this.zoom_zmin !== this.zoom_zmax)) {
               this.unzoom();
               this.getFramePainter().zoomChangedInteractive('reset');
               return true;
            }
            if (this.draw_content) {
               this.autoZoom();
               return true;
            }
            break;
         case "ToggleLogX": this.getFramePainter().toggleAxisLog("x"); break;
         case "ToggleLogY": this.getFramePainter().toggleAxisLog("y"); break;
         case "ToggleLogZ": this.getFramePainter().toggleAxisLog("z"); break;
         case "ToggleStatBox": this.toggleStat(); return true;
      }
      return false;
   }

   /** @summary Fill pad toolbar with hist-related functions */
   fillToolbar(not_shown) {
      let pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton("auto_zoom", 'Toggle between unzoom and autozoom-in', 'ToggleZoom', "Ctrl *");
      pp.addPadButton("arrow_right", "Toggle log x", "ToggleLogX", "PageDown");
      pp.addPadButton("arrow_up", "Toggle log y", "ToggleLogY", "PageUp");
      if (this.getDimension() > 1)
         pp.addPadButton("arrow_diag", "Toggle log z", "ToggleLogZ");
      if (this.draw_content)
         pp.addPadButton("statbox", 'Toggle stat box', "ToggleStatBox");
      if (!not_shown) pp.showPadButtons();
   }

   /** @summary get tool tips used in 3d mode */
   get3DToolTip(indx) {
      let histo = this.getHisto(),
          tip = { bin: indx, name: histo.fName || "histo", title: histo.fTitle };
      switch (this.getDimension()) {
         case 1:
            tip.ix = indx + 1; tip.iy = 1;
            tip.value = histo.getBinContent(tip.ix);
            tip.error = histo.getBinError(tip.ix);
            tip.lines = this.getBinTooltips(indx-1);
            break;
         case 2:
            tip.ix = (indx % this.nbinsx) + 1;
            tip.iy = (indx - (tip.ix - 1)) / this.nbinsx + 1;
            tip.value = histo.getBinContent(tip.ix, tip.iy);
            tip.error = histo.getBinError(tip.ix, tip.iy);
            tip.lines = this.getBinTooltips(tip.ix-1, tip.iy-1);
            break;
         case 3:
            tip.ix = indx % this.nbinsx + 1;
            tip.iy = ((indx - (tip.ix - 1)) / this.nbinsx) % this.nbinsy + 1;
            tip.iz = (indx - (tip.ix - 1) - (tip.iy - 1) * this.nbinsx) / this.nbinsx / this.nbinsy + 1;
            tip.value = histo.getBinContent(tip.ix, tip.iy, tip.iz);
            tip.error = histo.getBinError(tip.ix, tip.iy, tip.iz);
            tip.lines = this.getBinTooltips(tip.ix-1, tip.iy-1, tip.iz-1);
            break;
      }

      return tip;
   }

   /** @summary Create contour levels for currently selected Z range */
   createContour(main, palette, args) {
      if (!main || !palette) return;

      if (!args) args = {};

      let nlevels = gStyle.fNumberContours,
          zmin = this.minbin, zmax = this.maxbin, zminpos = this.minposbin;

      if (args.scatter_plot) {
         if (nlevels > 50) nlevels = 50;
         zmin = this.minposbin;
      }

      if (zmin === zmax) { zmin = this.gminbin; zmax = this.gmaxbin; zminpos = this.gminposbin; }

      if (this.getDimension() < 3) {
         if (main.zoom_zmin !== main.zoom_zmax) {
            zmin = main.zoom_zmin;
            zmax = main.zoom_zmax;
         } else if (args.full_z_range) {
            zmin = main.zmin;
            zmax = main.zmax;
         }
      }

      palette.setFullRange(main.zmin, main.zmax);
      palette.createContour(main.logz, nlevels, zmin, zmax, zminpos);

      if (this.getDimension() < 3) {
         main.scale_zmin = palette.colzmin;
         main.scale_zmax = palette.colzmax;
      }
   }

   /** @summary Start dialog to modify range of axis where histogram values are displayed */
   changeValuesRange(menu, arg) {
      let pmain = this.getFramePainter();
      if (!pmain) return;
      let prefix = pmain.isAxisZoomed(arg) ? "zoom_" + arg : arg;
      let curr = "[" + pmain[prefix+'min'] + "," + pmain[prefix+'max'] + "]";
      menu.input("Enter values range for axis " + arg + " like [0,100] or empty string to unzoom", curr).then(res => {
         res = res ? JSON.parse(res) : [];
         if (!res || (typeof res != "object") || (res.length!=2) || !Number.isFinite(res[0]) || !Number.isFinite(res[1]))
            pmain.unzoom(arg);
         else
            pmain.zoom(arg, res[0], res[1]);
      });
   }

   /** @summary Fill histogram context menu */
   fillContextMenu(menu) {

      menu.add("header:v7histo::anyname");

      if (this.draw_content) {
         menu.addchk(this.toggleStat('only-check'), "Show statbox", () => this.toggleStat());

         if (this.getDimension() == 2)
             menu.add("Values range", () => this.changeValuesRange(menu, "z"));

         if (typeof this.fillHistContextMenu == 'function')
            this.fillHistContextMenu(menu);
      }

      let fp = this.getFramePainter();

      if (this.options.Mode3D) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add("separator");

         let main = this.getMainPainter() || this;

         menu.addchk(main.isTooltipAllowed(), 'Show tooltips', () => main.setTooltipAllowed("toggle"));

         menu.addchk(fp.enable_highlight, 'Highlight bins', () => {
            fp.enable_highlight = !fp.enable_highlight;
            if (!fp.enable_highlight && main.highlightBin3D && main.mode3d) main.highlightBin3D(null);
         });

         if (fp && fp.render3D) {
            menu.addchk(main.options.FrontBox, 'Front box', () => {
               main.options.FrontBox = !main.options.FrontBox;
               fp.render3D();
            });
            menu.addchk(main.options.BackBox, 'Back box', () => {
               main.options.BackBox = !main.options.BackBox;
               fp.render3D();
            });
         }

         if (this.draw_content) {
            menu.addchk(!this.options.Zero, 'Suppress zeros', () => {
               this.options.Zero = !this.options.Zero;
               this.redrawPad();
            });

            if ((this.options.Lego==12) || (this.options.Lego==14)) {
               if (this.fillPaletteMenu) this.fillPaletteMenu(menu);
            }
         }

         if (main.control && typeof main.control.reset === 'function')
            menu.add('Reset camera', () => main.control.reset());
      }

      menu.addAttributesMenu(this);

      if (this.histogram_updated && fp.zoomChangedInteractive())
         menu.add('Let update zoom', () => fp.zoomChangedInteractive('reset'));

      return true;
   }

   /** @summary Update palette drawing */
   updatePaletteDraw() {
      if (this.isMainPainter()) {
         let pp = this.getPadPainter().findPainterFor(undefined, undefined, "ROOT::Experimental::RPaletteDrawable");
         if (pp) pp.drawPalette();
      }
   }

   /** @summary Fill menu entries for palette */
   fillPaletteMenu(menu) {
      menu.addPaletteMenu(this.options.Palette || settings.Palette, arg => {
         // TODO: rewrite for RPalette functionality
         this.options.Palette = parseInt(arg);
         this.redraw(); // redraw histogram
      });
   }

   /** @summary Toggle 3D drawing mode */
   toggleMode3D() {
      this.options.Mode3D = !this.options.Mode3D;

      if (this.options.Mode3D) {
         if (!this.options.Surf && !this.options.Lego && !this.options.Error) {
            if ((this.nbinsx>=50) || (this.nbinsy>=50))
               this.options.Lego = this.options.Color ? 14 : 13;
            else
               this.options.Lego = this.options.Color ? 12 : 1;

            this.options.Zero = false; // do not show zeros by default
         }
      }

      this.copyOptionsToOthers();
      this.interactiveRedraw("pad", "drawopt");
   }

   /** @summary Calculate histogram inidicies and axes values for each visible bin */
   prepareDraw(args) {

      if (!args) args = { rounding: true, extra: 0, middle: 0 };

      if (args.extra === undefined) args.extra = 0;
      if (args.right_extra === undefined) args.right_extra = args.extra;
      if (args.middle === undefined) args.middle = 0;

      let histo = this.getHisto(), xaxis = this.getAxis("x"), yaxis = this.getAxis("y"),
          pmain = this.getFramePainter(),
          hdim = this.getDimension(),
          i, j, x, y, binz, binarea,
          res = {
             i1: this.getSelectIndex("x", "left", 0 - args.extra),
             i2: this.getSelectIndex("x", "right", 1 + args.right_extra),
             j1: (hdim < 2) ? 0 : this.getSelectIndex("y", "left", 0 - args.extra),
             j2: (hdim < 2) ? 1 : this.getSelectIndex("y", "right", 1 + args.right_extra),
             k1: (hdim < 3) ? 0 : this.getSelectIndex("z", "left", 0 - args.extra),
             k2: (hdim < 3) ? 1 : this.getSelectIndex("z", "right", 1 + args.right_extra),
             stepi: 1, stepj: 1, stepk: 1,
             min: 0, max: 0, sumz: 0, xbar1: 0, xbar2: 1, ybar1: 0, ybar2: 1
          };

      if (this.isDisplayItem() && histo.fIndicies) {
         if (res.i1 < histo.fIndicies[0]) { res.i1 = histo.fIndicies[0]; res.incomplete = true; }
         if (res.i2 > histo.fIndicies[1]) { res.i2 = histo.fIndicies[1]; res.incomplete = true; }
         res.stepi = histo.fIndicies[2];
         if (res.stepi > 1) res.incomplete = true;
         if ((hdim > 1) && (histo.fIndicies.length > 5)) {
            if (res.j1 < histo.fIndicies[3]) { res.j1 = histo.fIndicies[3]; res.incomplete = true; }
            if (res.j2 > histo.fIndicies[4]) { res.j2 = histo.fIndicies[4]; res.incomplete = true; }
            res.stepj = histo.fIndicies[5];
            if (res.stepj > 1) res.incomplete = true;
         }
         if ((hdim > 2) && (histo.fIndicies.length > 8)) {
            if (res.k1 < histo.fIndicies[6]) { res.k1 = histo.fIndicies[6]; res.incomplete = true; }
            if (res.k2 > histo.fIndicies[7]) { res.k2 = histo.fIndicies[7]; res.incomplete = true; }
            res.stepk = histo.fIndicies[8];
            if (res.stepk > 1) res.incomplete = true;
         }
      }

      if (args.only_indexes) return res;

      // no need for Float32Array, plain Array is 10% faster
      // reserve more places to avoid complex boundary checks

      res.grx = new Array(res.i2+res.stepi+1);
      res.gry = new Array(res.j2+res.stepj+1);

      if (args.original) {
         res.original = true;
         res.origx = new Array(res.i2+1);
         res.origy = new Array(res.j2+1);
      }

      if (args.pixel_density) args.rounding = true;

      let funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);

       // calculate graphical coordinates in advance
      for (i = res.i1; i <= res.i2; ++i) {
         x = xaxis.GetBinCoord(i + args.middle);
         if (funcs.logx && (x <= 0)) { res.i1 = i+1; continue; }
         if (res.origx) res.origx[i] = x;
         res.grx[i] = funcs.grx(x);
         if (args.rounding) res.grx[i] = Math.round(res.grx[i]);

         if (args.use3d) {
            if (res.grx[i] < -pmain.size_x3d) { res.i1 = i; res.grx[i] = -pmain.size_x3d; }
            if (res.grx[i] > pmain.size_x3d) { res.i2 = i; res.grx[i] = pmain.size_x3d; }
         }
      }

      if (args.use3d) {
         if ((res.i1 < res.i2-2) && (res.grx[res.i1] == res.grx[res.i1+1])) res.i1++;
         if ((res.i1 < res.i2-2) && (res.grx[res.i2-1] == res.grx[res.i2])) res.i2--;
      }

      // copy last valid value to higher indicies
      while (i < res.i2 + res.stepi + 1)
         res.grx[i++] = res.grx[res.i2];

      if (hdim === 1) {
         res.gry[0] = funcs.gry(0);
         res.gry[1] = funcs.gry(1);
      } else
      for (j = res.j1; j <= res.j2; ++j) {
         y = yaxis.GetBinCoord(j + args.middle);
         if (funcs.logy && (y <= 0)) { res.j1 = j+1; continue; }
         if (res.origy) res.origy[j] = y;
         res.gry[j] = funcs.gry(y);
         if (args.rounding) res.gry[j] = Math.round(res.gry[j]);

         if (args.use3d) {
            if (res.gry[j] < -pmain.size_y3d) { res.j1 = j; res.gry[j] = -pmain.size_y3d; }
            if (res.gry[j] > pmain.size_y3d) { res.j2 = j; res.gry[j] = pmain.size_y3d; }
         }
      }

      if (args.use3d && (hdim > 1)) {
         if ((res.j1 < res.j2-2) && (res.gry[res.j1] == res.gry[res.j1+1])) res.j1++;
         if ((res.j1 < res.j2-2) && (res.gry[res.j2-1] == res.gry[res.j2])) res.j2--;
      }

      // copy last valid value to higher indicies
      while ((hdim > 1) && (j < res.j2 + res.stepj + 1))
         res.gry[j++] = res.gry[res.j2];

      //  find min/max values in selected range
      this.maxbin = this.minbin = this.minposbin = null;

      for (i = res.i1; i < res.i2; i += res.stepi) {
         for (j = res.j1; j < res.j2; j += res.stepj) {
            binz = histo.getBinContent(i + 1, j + 1);
            if (!Number.isFinite(binz)) continue;
            res.sumz += binz;
            if (args.pixel_density) {
               binarea = (res.grx[i+res.stepi]-res.grx[i])*(res.gry[j]-res.gry[j+res.stepj]);
               if (binarea <= 0) continue;
               res.max = Math.max(res.max, binz);
               if ((binz>0) && ((binz<res.min) || (res.min===0))) res.min = binz;
               binz = binz/binarea;
            }
            if (this.maxbin===null) {
               this.maxbin = this.minbin = binz;
            } else {
               this.maxbin = Math.max(this.maxbin, binz);
               this.minbin = Math.min(this.minbin, binz);
            }
            if (binz > 0)
               if ((this.minposbin===null) || (binz<this.minposbin)) this.minposbin = binz;
         }
      }

      res.palette = pmain.getHistPalette();

      if (res.palette)
         this.createContour(pmain, res.palette, args);

      return res;
   }

} // class RHistPainter


export { RHistPainter }
