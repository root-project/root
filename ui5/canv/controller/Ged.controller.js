sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/core/Fragment',
   'sap/ui/core/HTML'
], function (Controller, JSONModel, Fragment, HTML) {
   "use strict";

   return Controller.extend("rootui5.canv.controller.Ged", {

      currentPainter: null,

      gedFragments : [],

      onInit : function() {
         console.log('init GED editor');
         var model = new JSONModel({ SelectedClass: "none" });
         this.getView().setModel(model);
      },

      onExit : function() {
         console.log('exit GED editor');
         this.cleanupGed();

      },

      cleanupGed : function() {
         console.log('Clenup GED editor');

         // empty fragments
         this.getView().byId("ged_page").removeAllContent();

         // set dummy model
         this.getView().setModel(new JSONModel({ SelectedClass: "none" }));

         // remove references
         this.currentPainter = null;
         this.currentPadPainter = null;
         this.currentPlace = undefined;

         // TODO: deregsiter for all events

      },

      addFragment : function(page, kind, model) {
         let fragm = this.gedFragments[kind];

         if (!fragm)
            return Fragment.load({
               name: "rootui5.canv.view." + kind,
               type: "XML",
               controller: this
            }).then(function(_page, _kind, _model, oFragm) {
               this.gedFragments[_kind] = oFragm;
               this.addFragment(_page, _kind, _model);
            }.bind(this, page, kind, model));

         fragm.ged_fragment = true; // mark as ged fragment

         let html = new HTML();
         html.setContent("<hr>");
         html.setTooltip(kind);
         page.addContent(html);

         fragm.setModel(model);
         page.addContent(fragm);
      },

      /// function called when user changes model property
      /// data object includes _kind, _painter and _handle (optionally)
      modelPropertyChange : function(evnt, data) {
         let pars = evnt.getParameters();
         console.log('Model property changes', pars.path, pars.value, data._kind);

         if (data._handle) {
            //var subname = pars.path.substr(1);
            //if (subname in data._handle) data._handle[subname] = pars.value;

            if (typeof data._handle.verifyDirectChange === 'function')
                data._handle.verifyDirectChange(data._painter);
            data._handle.changed = true;
         }

         let exec = "", item = pars.path.substr(1), obj;

         if (data._painter) {
            obj = data._painter.snapid ? data._painter.GetObject() : null;

            if (typeof data._painter.AttributeChange === 'function')
               data._painter.AttributeChange(data._kind, item, pars.value);
         }

         if (obj) {
            if ((data._kind === "TAttLine") && (obj.fLineColor!==undefined) && (obj.fLineStyle!==undefined) && (obj.fLineWidth!==undefined)) {
               if (item == "attline/width")
                  exec = "exec:SetLineWidth(" + pars.value + ")";
               else if (item == "attline/style")
                  exec = "exec:SetLineStyle(" + pars.value + ")";
               else if (item == "attline/color")
                  exec = data._painter.GetColorExec(pars.value, "SetLineColor");
            } else if ((data._kind === "TAttFill") && (obj.fFillColor!==undefined) && (obj.fFillStyle!==undefined))  {
               if (item == "attfill/pattern")
                  exec = "exec:SetFillStyle(" + pars.value + ")";
               else if (item == "attfill/color")
                  exec = data._painter.GetColorExec(pars.value, "SetFillColor");
            }
         }

         if (data._painter)
            data._painter.InteractiveRedraw("pad", exec); // TODO: some objects can readraw directly, no need to redraw pad
         else if (this.currentPadPainter)
            this.currentPadPainter.Redraw();
      },

      processAxisModelChange: function(evnt, data) {
         let pars = evnt.getParameters(),
             item = pars.path.substr(1),
             exec = "", 
             painter = this.currentPainter,
             kind = this.currentPlace,
             axis = painter.GetObject(kind);

         // while axis painter is temporary object, we should not try change it attributes

         if (!this.currentPadPainter || !axis) return;

         if ((typeof kind == 'string') && (kind.indexOf("axis")==1)) kind = kind.substr(0,1);

         console.log(`Change axis ${kind} item ${item} value ${pars.value}`);

         switch(item) {
            case "axis/fTitle":
               exec = `exec:SetTitle("${pars.value}")`;
               break;
            case "axiscolor":
               axis.fAxisColor = this.currentPadPainter.add_color(pars.value);
               exec = painter.GetColorExec(pars.value, "SetAxisColor");
               break;
            case "color_label":
               axis.fLabelColor = this.currentPadPainter.add_color(pars.value);
               exec = painter.GetColorExec(pars.value, "SetLabelColor");
               break;
            case "center_label":
               axis.InvertBit(JSROOT.EAxisBits.kCenterLabels);
               exec = `exec:CenterLabels(${pars.value ? true : false})`;
               break;
            case "vert_label":
               axis.InvertBit(JSROOT.EAxisBits.kLabelsVert);
               exec = `exec:SetBit(TAxis::kLabelsVert,${pars.value ? true : false})`;
               break;
            case "axis/fLabelOffset":
               exec = `exec:SetLabelOffset(${pars.value})`;
               break;
            case "axis/fLabelSize":
               exec = `exec:SetLabelOffset(${pars.value})`;
               break;
            case "color_title":
               axis.fLabelColor = this.currentPadPainter.add_color(pars.value);
               exec = painter.GetColorExec(pars.value, "SetTitleColor");
               break;
            case "center_title":
               axis.InvertBit(JSROOT.EAxisBits.kCenterTitle);
               exec = `exec:CenterTitle(${pars.value ? true : false})`;
               break;
            case "rotate_title":
               axis.InvertBit(JSROOT.EAxisBits.kRotateTitle);
               exec = `exec:RotateTitle(${pars.value ? true : false})`;
               break;
            case "axis/fTickLength":
               exec = `exec:SetTickLength(${pars.value})`;
               break;
            case "axis/fTitleOffset":
               exec = `exec:SetTitleOffset(${pars.value})`;
               break;
            case "axis/fTitleSize":
               exec = `exec:SetTitleSize(${pars.value})`;
               break;
         }

         // TAxis belongs to main painter like TH1, therefore submit commands there
         let main = this.currentPainter.main_painter(true);

         if (main && main.snapid) {
            console.log('Invoke interactive redraw ', main.snapid, kind)
            main.InteractiveRedraw("pad", exec, kind);
         } else {
            console.log('pad', this.currentPadPainter.this_pad_name, this.currentPadPainter.pad_name, 'iscan', this.currentPadPainter.iscan)
            console.log('Do have main = ', main ? main.smapid : "---")
            this.currentPadPainter.Redraw();
         }
      },

      processHistModelChange : function(evnt, data) {
         let pars = evnt.getParameters(), opts = data.options;

         opts.Mode3D = opts.Mode3Dindx > 0;
         opts.Lego = parseInt(opts.Lego);
         opts.Contor = parseInt(opts.Contor);
         opts.ErrorKind = parseInt(opts.ErrorKind);

         if (this.currentPadPainter)
            this.currentPadPainter.InteractiveRedraw("pad","drawopt");
      },
      
      setAxisModel : function(model) {
         let obj =  this.currentPainter.GetObject(this.currentPlace),
             painter = this.currentPlace == "xaxis" ? this.currentPainter.x_handle : this.currentPainter.y_handle;  
        
         let data = {
             specialRefresh: 'setAxisModel', 
             axis: obj,
             axiscolor: painter.lineatt.color, 
             color_label: this.currentPadPainter.get_color(obj.fLabelColor), 
             center_label: obj.TestBit(JSROOT.EAxisBits.kCenterLabels),
             vert_label: obj.TestBit(JSROOT.EAxisBits.kLabelsVert),
             color_title: this.currentPadPainter.get_color(obj.fTitleColor),
             center_title: obj.TestBit(JSROOT.EAxisBits.kCenterTitle),
             rotate_title: obj.TestBit(JSROOT.EAxisBits.kRotateTitle),
         };
        
         model.setData(data);
      },

      onObjectSelect : function(padpainter, painter, place) {

         if (this.currentPainter === painter) return;

         this.currentPadPainter = padpainter;
         this.currentPainter = painter;
         this.currentPlace = place;
         
         let obj = painter.GetObject(place);
         if (place == "xaxis" && painter.x_handle) painter = painter.x_handle; else
         if (place == "yaxis" && painter.y_handle) painter = painter.y_handle;

         let selectedClass = obj ? obj._typename : painter.GetTipName();

         this.getView().getModel().setProperty("/SelectedClass", selectedClass);

         let oPage = this.getView().byId("ged_page");
         oPage.removeAllContent();

         if (painter.lineatt && painter.lineatt.used && !painter.lineatt.not_standard) {
            let model = new JSONModel( { attline: painter.lineatt } );
            model.attachPropertyChange({ _kind: "TAttLine", _painter: painter, _handle: painter.lineatt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttLine", model);
         }

         if (painter.fillatt && painter.fillatt.used) {
            let model = new JSONModel( { attfill: painter.fillatt } );
            model.attachPropertyChange({ _kind: "TAttFill", _painter: painter, _handle: painter.fillatt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttFill", model);
         }

         if (painter.markeratt && painter.markeratt.used) {
            let model = new JSONModel( { attmark: painter.markeratt } );
            model.attachPropertyChange({ _kind: "TAttMarker", _painter: painter, _handle: painter.markeratt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttMarker", model);
         }

         if (typeof painter.processTitleChange == 'function') {
            let tobj = painter.processTitleChange("check");
            if (tobj) {
               let model = new JSONModel({ tnamed: tobj });
               model.attachPropertyChange( {}, painter.processTitleChange, painter );
               this.addFragment(oPage, "TNamed", model);
            }
         }

         if (selectedClass == "TAxis") {
            console.log('place', place)
            
            let model = new JSONModel({}); 
            this.setAxisModel(model); 
            this.addFragment(oPage, "Axis", model);
            model.attachPropertyChange({ _kind: "TAxis" }, this.processAxisModelChange, this);
         }

         if (typeof painter.GetHisto == 'function') {

            painter.options.Mode3Dindx = painter.options.Mode3D ? 1 : 0;

            let model = new JSONModel({ opts : painter.options });

            // model.attachPropertyChange({}, painter.processTitleChange, painter);
            this.addFragment(oPage, "Hist", model);

            model.attachPropertyChange({ options: painter.options }, this.processHistModelChange, this);
         }
      },

      onObjectRedraw : function(padpainter, painter) {
         if ((this.currentPadPainter !== padpainter) || (this.currentPainter !== painter)) return;

         // console.log('GED sees selected object redraw');

         let page = this.getView().byId("ged_page");
         let cont = page.getContent();

         for (let n = 0; n < cont.length; ++n)
            if (cont[n] && cont[n].ged_fragment) {
               let model = cont[n].getModel();
               
               let func = model.getProperty("/specialRefresh");
               if (func)
                  this[func](model);
               else
                  model.refresh();
            }
      },
      
      onPadRedraw : function(padpainter) {
         if (this.currentPadPainter === padpainter)
            this.onObjectRedraw(this.currentPadPainter, this.currentPainter);
      },

      padEventsReceiver : function(evnt) {
         if (!evnt) return;

         if (evnt.what == "select")
            this.onObjectSelect(evnt.padpainter, evnt.painter, evnt.place);
         else if (evnt.what == "redraw")
            this.onObjectRedraw(evnt.padpainter, evnt.painter);
         else if (evnt.what == "padredraw") 
            this.onPadRedraw(evnt.padpainter);
      }

   });

});
