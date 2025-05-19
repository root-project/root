sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/core/Fragment',
   'sap/ui/core/HTML'
], function (Controller, JSONModel, Fragment, HTML) {
   "use strict";

   return Controller.extend('rootui5.canv.controller.Ged', {

      currentPainter: null,

      gedFragments : [],

      onInit() {
         let model = new JSONModel({ SelectedClass: 'none' });
         this.getView().setModel(model);

         let data = this.getView().getViewData();
         this.jsroot = data?.jsroot;
      },

      onExit() {
         this.cleanupGed();
      },

      /** @summary Produce exec string for WebCanas to set color value
        * @desc Color can be id or string, but should belong to list of known colors
        * For higher color numbers TColor::GetColor(r,g,b) will be invoked to ensure color will be created on server side
        * @private */
      getColorExec(painter, col, method) {
         return this.jsroot?.getColorExec(col, method) || `exec:${method}(1)`;
      },

      /** @summary Returns EAxisBits member
        * @private */
      getAxisBit(name) {
         return this.jsroot?.EAxisBits ? this.jsroot?.EAxisBits[name] : 1;
      },

      cleanupGed() {
         // empty fragments
         this.getView().byId('ged_page').removeAllContent();

         // set dummy model
         this.getView().setModel(new JSONModel({ SelectedClass: 'none', SelectedPlace: '' }));

         // remove references
         this.currentPainter = null;
         this.currentPadPainter = null;
         this.currentPlace = undefined;
         this.currentHistPainter = null;
         this.currentHistAxis = '';


         // TODO: deregsiter for all events
      },

      addFragment(page, kind, model) {
         let fragm = this.gedFragments[kind];

         if (!fragm)
            return Fragment.load({
               name: `rootui5.canv.view.${kind}`,
               type: 'XML',
               controller: this
            }).then(function(_page, _kind, _model, oFragm) {
               this.gedFragments[_kind] = oFragm;
               return this.addFragment(_page, _kind, _model);
            }.bind(this, page, kind, model));

         fragm.ged_fragment = true; // mark as ged fragment

         let html = new HTML();
         html.setContent('<hr>');
         html.setTooltip(kind);
         page.addContent(html);

         fragm.setModel(model);
         fragm.setTooltip(kind);
         page.addContent(fragm);

         return Promise.resolve(false);

      },

      /// function called when user changes model property
      /// data object includes _kind, _painter and _handle (optionally)
      modelPropertyChange(evnt, data) {
         let pars = evnt.getParameters();

         if (data._handle) {
            if (typeof data._handle.verifyDirectChange === 'function')
                data._handle.verifyDirectChange(data._painter);
            data._handle.changed = true;
         }

         let exec = '', item = pars.path.substr(1),
             obj = data._painter?.getSnapId() ? data._painter.getObject() : null;

         if (obj) {
            if ((data._kind === 'TAttLine') && (obj.fLineColor !== undefined) && (obj.fLineStyle !== undefined) && (obj.fLineWidth !== undefined)) {
               if (item == 'attline/width')
                  exec = `exec:SetLineWidth(${pars.value})`;
               else if (item == 'attline/style')
                  exec = `exec:SetLineStyle(${pars.value})`;
               else if (item == 'attline/color')
                  exec = this.getColorExec(data._painter, pars.value, 'SetLineColor');
            } else if ((data._kind === 'TAttFill') && (obj.fFillColor !== undefined) && (obj.fFillStyle !== undefined)) {
               if (item == 'attfill/pattern')
                  exec = `exec:SetFillStyle(${pars.value})`;
               else if (item == 'attfill/color')
                  exec = this.getColorExec(data._painter, pars.value, 'SetFillColor');
            } else if ((data._kind === 'TAttMarker') && (obj.fMarkerColor !== undefined) && (obj.fMarkerStyle !== undefined) && (obj.fMarkerSize !== undefined)) {
               if (item == 'attmark/size')
                  exec = `exec:SetMarkerSize(${pars.value})`;
               else if (item == 'attmark/style')
                  exec = `exec:SetMarkerStyle(${pars.value})`;
               else if (item == 'attmark/color')
                  exec = this.getColorExec(data._painter, pars.value, 'SetMarkerColor');
            } else if ((data._kind === 'TAttText') && (obj.fTextColor !== undefined) && (obj.fTextFont !== undefined) && (obj.fTextSize !== undefined)) {
               if (item == 'atttext/color')
                  exec = this.getColorExec(data._painter, pars.value, 'SetTextColor');
               else if (item == 'atttext/size')
                  exec = `exec:SetTextSize(${pars.value})`;
               else if ((item == 'atttext/font_index') && data._painter?.textatt)
                  exec = `exec:SetTextFont(${data._painter.textatt.setGedFont(pars.value)})`;
               else if (item == 'atttext/align')
                  exec = `exec:SetTextAlign(${pars.value})`;
               else if (item == 'atttext/angle')
                  exec = `exec:SetTextAngle(${pars.value})`;
            }
         }

         if (data._painter)
            data._painter.interactiveRedraw('pad', exec); // TODO: some objects can readraw directly, no need to redraw pad
         else if (this.currentPadPainter)
            this.currentPadPainter.redraw();
      },

      getAxisHandle() {
         const fp = this.currentPainter?.getFramePainter();
         if (!fp || !this.currentPlace)
            return this.currentPainter;

         let found = null;
         ['x_handle','y_handle','z_handle','x2_handle', 'y2_handle', 'z2_handle'].forEach(name => {
            const handle = fp[name];
            if ((handle?.hist_painter === this.currentPainter) && (handle?.hist_axis === this.currentPlace))
               found = handle;
         });

         return found;
      },

      setAxisModel(model) {
         let painter = this.getAxisHandle(),
             obj = painter?.getObject(),
             axis_chopt = '', axis_ticksize = 0, color_title = '';
         if (painter?.is_gaxis) {
            axis_chopt = obj.fChopt;
            axis_ticksize = obj.fTickSize;
            color_title = this.currentPadPainter.getColor(obj.fTextColor);
         } else {
            axis_ticksize = obj.fTickLength;
            color_title = this.currentPadPainter.getColor(obj.fTitleColor);
         }

         let data = {
             is_gaxis: painter?.is_gaxis,
             mode2d: !painter.hist_painter?.options.Mode3D,
             specialRefresh: 'setAxisModel',
             axis: obj,
             axis_chopt,
             axis_ticksize,
             axiscolor: painter.lineatt.color,
             color_label: this.currentPadPainter.getColor(obj.fLabelColor),
             center_label: obj.TestBit(this.getAxisBit('kCenterLabels')),
             vert_label: obj.TestBit(this.getAxisBit('kLabelsVert')),
             font_label: painter.labelsFont?.index || 0,
             color_title,
             center_title: obj.TestBit(this.getAxisBit('kCenterTitle')),
             rotate_title: obj.TestBit(this.getAxisBit('kRotateTitle')),
             font_title: painter.titleFont?.index || 0,
         };

         model.setData(data);
      },

      processAxisModelChange(evnt /*, data */) {
         let pars = evnt.getParameters(),
             item = pars.path.substr(1),
             handle = this.getAxisHandle(),
             axis = handle?.getObject(),
             is_gaxis = handle?.is_gaxis,
             kind = '',
             exec = '',
             col, fontid;

         // while axis painter is temporary object, we should not try change it attributes
         if (!this.currentPadPainter || !axis)
            return;

         if (!is_gaxis && handle?.hist_painter && handle?.hist_axis)
            kind = handle.hist_axis;

         switch(item) {
            case 'axis/fTitle':
               exec = `exec:SetTitle("${pars.value}")`;
               break;
            case 'axiscolor':
               col = this.currentPadPainter.addColor(pars.value);
               if (is_gaxis)
                  axis.fLineColor = col;
               else
                  axis.fAxisColor = col;
               exec = this.getColorExec(this.currentPadPainter, pars.value, is_gaxis ? 'SetLineColor' : 'SetAxisColor');
               break;
            case 'color_label':
               axis.fLabelColor = this.currentPadPainter.addColor(pars.value);
               exec = this.getColorExec(this.currentPadPainter, pars.value, 'SetLabelColor');
               break;
            case 'center_label':
               axis.InvertBit(this.getAxisBit('kCenterLabels'));
               exec = `exec:CenterLabels(${pars.value ? true : false})`;
               break;
            case 'vert_label':
               axis.InvertBit(this.getAxisBit('kLabelsVert'));
               exec = `exec:SetBit(TAxis::kLabelsVert, ${pars.value ? true : false})`;
               break;
            case 'font_label':
               fontid = parseInt(pars.value)*10 + 2;
               axis.fLabelFont = fontid;
               exec = `exec:SetLabelFont(${fontid})`;
               break;
            case 'axis/fLabelOffset':
               exec = `exec:SetLabelOffset(${pars.value})`;
               break;
            case 'axis/fLabelSize':
               exec = `exec:SetLabelSize(${pars.value})`;
               break;
            case 'color_title':
               col = this.currentPadPainter.addColor(pars.value);
               if (is_gaxis)
                  axis.fTextColor = col;
               else
                  axis.fTitleColor = col;
               exec = this.getColorExec(this.currentPadPainter, pars.value, 'SetTitleColor');
               break;
            case 'center_title':
               axis.InvertBit(this.getAxisBit('kCenterTitle'));
               exec = `exec:CenterTitle(${pars.value ? true : false})`;
               break;
            case 'rotate_title':
               axis.InvertBit(this.getAxisBit('kRotateTitle'));
               exec = is_gaxis ? `exec:SetBit(TAxis::kRotateTitle, ${pars.value ? true : false})` : `exec:RotateTitle(${pars.value ? true : false})`;
               break;
            case 'font_title':
               fontid = parseInt(pars.value)*10 + 2;
               if (is_gaxis)
                  axis.fTextFont = fontid;
               else
                  axis.fTitleFont = fontid;
               exec = `exec:SetTitleFont(${fontid})`;
               break;
            case 'axis_ticksize':
               if (is_gaxis)
                  axis.fTickSize = pars.value;
               else
                  axis.fTickLength = pars.value;
               exec = `exec:SetTickLength(${pars.value})`;
               break;
            case 'axis/fTitleOffset':
               exec = `exec:SetTitleOffset(${pars.value})`;
               break;
            case 'axis/fTitleSize':
               exec = `exec:SetTitleSize(${pars.value})`;
               break;
            case 'axis_chopt':
               axis.fChopt = pars.value;
               exec = `exec:SetOption("${pars.value}")`;
               break;
         }

         // TAxis belongs to main painter like TH1, therefore submit commands there
         const main = is_gaxis ? handle : handle.hist_painter;

         if (main?.getSnapId())
            main.interactiveRedraw('pad', exec, kind);
         else
            this.currentPadPainter.redraw();
      },

      setRAxisModel(model) {
         let handle = this.getAxisHandle();
         if (!handle) return;

         let data = {
            logbase: handle.logbase || 0,
            handle,
            mode2d: true,
            ticks_size: handle.ticksSize/handle.scaling_size,
            labels_offset: handle.labelsOffset/handle.scaling_size,
            labels_rotate: handle.labelsFont.angle != 0,
            title_offset: handle.titleOffset/handle.scaling_size,
            title_rotate: handle.isTitleRotated(),
            specialRefresh: 'setRAxisModel'
         };

         if (Math.abs(data.logbase - 2.7) < 0.1) data.logbase = 3; // for ln

         model.setData(data);
      },

      processRAxisModelChange(evnt, data) {
         let pars = evnt.getParameters(),
             item = pars.path.substr(1);

         let handle = this.getAxisHandle();
         if (!handle) return;

         switch(item) {
            case 'logbase': handle.changeAxisLog((pars.value == 3) ? Math.exp(1) : pars.value); break;
            case 'handle/ticksColor': handle.changeAxisAttr(1, 'ticks_color_name', pars.value); break;
            case 'ticks_size': handle.changeAxisAttr(1, 'ticks_size', pars.value); break;
            case 'handle/ticksSide': handle.changeAxisAttr(1, 'ticks_side', pars.value); break;
            case 'labels_offset': handle.changeAxisAttr(1, 'labels_offset', pars.value); break;
            case 'labels_rotate': handle.changeAxisAttr(1, 'labels_angle', pars.value ? 180 : 0); break;
            case 'handle/fTitle': handle.changeAxisAttr(1, 'title', pars.value); break;
            case 'title_offset': handle.changeAxisAttr(1, 'title_offset', pars.value); break;
            case 'handle/titlePos': handle.changeAxisAttr(1, 'title_position', pars.value); break;
            case 'title_rotate': handle.changeAxisAttr(1, 'title_angle', pars.value ? 180 : 0); break;
         }
      },

      processHistModelChange(evnt, data) {
         // let pars = evnt.getParameters();
         let opts = data.options;

         opts.Mode3D = opts.Mode3Dindx > 0;

         opts.Lego = parseInt(opts.Lego);
         opts.Surf = parseInt(opts.Surf);

         let cl = this.getView().getModel().getProperty('/SelectedClass');

         if ((typeof cl == 'string') && opts.Mode3D && (cl.indexOf('ROOT::Experimental::RHist') == 0))
            opts.Lego = 12;

         opts.Contour = parseInt(opts.Contour);
         opts.ErrorKind = parseInt(opts.ErrorKind);
         opts.BoxStyle = parseInt(opts.BoxStyle);
         opts.GLBox = parseInt(opts.GLBox);

         this.currentPainter?.interactiveRedraw('pad', 'drawopt');
      },

      async onObjectSelect(padpainter, painter) {
         let place = '';
         if (painter.hist_painter && painter.hist_axis) {
            // always keep reference on hist painter for x/y/z axis
            // while axis painter is temporary and can change at any time
            place = painter.hist_axis;
            painter = painter.hist_painter;
         }

         if ((this.currentPainter === painter) && (place === this.currentPlace)) return;

         this.currentPadPainter = padpainter;
         this.currentPainter = painter;
         this.currentPlace = place;

         if (place)
            painter = this.getAxisHandle();

         let obj = painter.getObject(),
             selectedClass = obj?._typename ?? painter.getObjectHint();


         this.getView().getModel().setProperty('/SelectedClass', selectedClass);
         this.getView().getModel().setProperty('/SelectedPlace', place);

         let oPage = this.getView().byId('ged_page');
         oPage.removeAllContent();

         if (selectedClass == 'RAttrAxis') {
            let model = new JSONModel({});
            this.setRAxisModel(model);
            model.attachPropertyChange({ _kind: 'RAttrAxis' }, this.processRAxisModelChange, this);
            await this.addFragment(oPage, 'RAxis', model);
            return;
         }

         if (painter.lineatt?.used && (selectedClass !== 'TAxis') && (selectedClass !== 'TGaxis')) {
            console.log('Assign line attributes', painter.getObject()._typename, 'not_standard', painter.lineatt.not_standard);
            let model = new JSONModel( { attline: painter.lineatt } );
            model.attachPropertyChange({ _kind: 'TAttLine', _painter: painter, _handle: painter.lineatt }, this.modelPropertyChange, this);
            await this.addFragment(oPage, 'TAttLine', model);
         }

         if (painter.fillatt?.used) {
            let model = new JSONModel( { attfill: painter.fillatt } );
            model.attachPropertyChange({ _kind: 'TAttFill', _painter: painter, _handle: painter.fillatt }, this.modelPropertyChange, this);
            await this.addFragment(oPage, 'TAttFill', model);
         }

         if (painter.markeratt?.used) {
            let model = new JSONModel( { attmark: painter.markeratt } );
            model.attachPropertyChange({ _kind: 'TAttMarker', _painter: painter, _handle: painter.markeratt }, this.modelPropertyChange, this);
            await this.addFragment(oPage, 'TAttMarker', model);
         }

         if (painter.textatt) {
            painter.textatt.font_index = Math.floor(painter.textatt.font/10);
            painter.textatt.size_visible = painter.textatt.size > 0;

            let model = new JSONModel( { atttext: painter.textatt } );
            model.attachPropertyChange({ _kind: 'TAttText', _painter: painter, _handle: painter.textatt }, this.modelPropertyChange, this);
            await this.addFragment(oPage, 'TAttText', model);
         }

         if (typeof painter.processTitleChange == 'function') {
            let tobj = painter.processTitleChange('check');
            if (tobj) {
               let model = new JSONModel({ tnamed: tobj });
               model.attachPropertyChange( {}, painter.processTitleChange, painter );
               await this.addFragment(oPage, 'TNamed', model);
            }
         }

         if (selectedClass == 'TAxis') {
            let model = new JSONModel({ is_gaxis: false });
            this.setAxisModel(model);
            model.attachPropertyChange({ _kind: 'TAxis' }, this.processAxisModelChange, this);
            await this.addFragment(oPage, 'Axis', model);
         } else if (selectedClass == 'TGaxis') {
            let model = new JSONModel({ is_gaxis: true });
            this.setAxisModel(model);
            model.attachPropertyChange({ _kind: 'TGaxis' }, this.processAxisModelChange, this);
            await this.addFragment(oPage, 'Axis', model);
         } else if (typeof painter.getHisto == 'function') {
            painter.options.Mode3Dindx = painter.options.Mode3D ? 1 : 0;
            painter.options.Error = !!painter.options.Error;
            painter.options.Palette = !!painter.options.Palette;
            painter.options.Zero = !!painter.options.Zero;
            let model = new JSONModel({ opts: painter.options });
            model.attachPropertyChange({ options: painter.options }, this.processHistModelChange, this);
            await this.addFragment(oPage, 'Hist', model);
         }
      },

      onObjectRedraw(padpainter, painter) {
         if ((this.currentPadPainter !== padpainter) || (this.currentPainter !== painter)) return;

         let page = this.getView().byId('ged_page'),
             cont = page.getContent();

         for (let n = 0; n < cont.length; ++n)
            if (cont[n] && cont[n].ged_fragment) {
               let model = cont[n].getModel();

               let func = model.getProperty('/specialRefresh');
               if (func && (typeof func == 'function'))
                  this[func](model);
               else
                  model.refresh();
            }
      },

      onPadRedraw(padpainter) {
         if (this.currentPadPainter === padpainter)
            this.onObjectRedraw(this.currentPadPainter, this.currentPainter);
      },

      padEventsReceiver(evnt) {
         if (!evnt) return;

         if (evnt.what == 'select')
            this.onObjectSelect(evnt.padpainter, evnt.painter);
         else if (evnt.what == 'redraw')
            this.onObjectRedraw(evnt.padpainter, evnt.painter);
         else if (evnt.what == 'padredraw')
            this.onPadRedraw(evnt.padpainter);
      }

   });

});
