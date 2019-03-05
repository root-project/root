sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/core/Fragment',
   'sap/m/Dialog',
   'sap/m/Button',
   'sap/ui/core/HTML'
], function (Controller, JSONModel, Fragment, Dialog, Button, HTML) {
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

         // TODO: deregsiter for all events

      },

      addFragment : function(page, kind, model) {
         var fragm = this.gedFragments[kind];

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

         var html = new HTML();
         html.setContent("<hr>");
         html.setTooltip(kind);
         page.addContent(html);

         fragm.setModel(model);
         page.addContent(fragm);
      },

      /// function called when user changes model property
      /// data object includes _kind, _painter and _handle (optionally)
      modelPropertyChange : function(evnt, data) {
         var pars = evnt.getParameters();
         console.log('Model property changes', pars.path, pars.value, data._kind);

         if (data._handle) {
            //var subname = pars.path.substr(1);
            //if (subname in data._handle) data._handle[subname] = pars.value;

            if (typeof data._handle.verifyDirectChange === 'function')
                data._handle.verifyDirectChange(data._painter);
            data._handle.changed = true;
         }

         if (data._painter && (typeof data._painter.AttributeChange === 'function'))
            data._painter.AttributeChange(data._kind, pars.path.substr(1), pars.value);

         if (this.currentPadPainter)
            this.currentPadPainter.Redraw();
      },

      processHistModelChange : function(evnt, data) {
         var pars = evnt.getParameters(), opts = data.options;
         console.log('Hist model changes', pars.path, pars.value);

         opts.Mode3D = opts.Mode3Dindx > 0;
         opts.Lego = parseInt(opts.Lego);
         opts.Contor = parseInt(opts.Contor);
         opts.ErrorKind = parseInt(opts.ErrorKind);

         if (this.currentPadPainter)
            this.currentPadPainter.InteractiveRedraw("object","drawopt");

      },

      onObjectSelect : function(padpainter, painter, place) {

         if (this.currentPainter === painter) return;

         this.currentPadPainter = padpainter;
         this.currentPainter = painter;

         var obj = painter.GetObject();

         this.getView().getModel().setProperty("/SelectedClass", obj ? obj._typename : painter.GetTipName());

         var oPage = this.getView().byId("ged_page");
         oPage.removeAllContent();

         if (painter.lineatt && painter.lineatt.used) {
            var model = new JSONModel( { attline: painter.lineatt } );
            model.attachPropertyChange({ _kind: "TAttLine", _painter: painter, _handle: painter.lineatt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttLine", model);
         }

         if (painter.fillatt && painter.fillatt.used) {
            var model = new JSONModel( { attfill: painter.fillatt } );
            model.attachPropertyChange({ _kind: "TAttFill", _painter: painter, _handle: painter.fillatt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttFill", model);
         }

         if (painter.markeratt && painter.markeratt.used) {
            var model = new JSONModel( { attmark: painter.markeratt } );
            model.attachPropertyChange({ _kind: "TAttMarker", _painter: painter, _handle: painter.markeratt }, this.modelPropertyChange, this);

            this.addFragment(oPage, "TAttMarker", model);
         }

         if (typeof painter.processTitleChange == 'function') {
            var obj = painter.processTitleChange("check");
            if (obj) {
               var model = new JSONModel( { tnamed: obj } );
               model.attachPropertyChange( {}, painter.processTitleChange, painter );
               this.addFragment(oPage, "TNamed", model);
            }

            if (typeof painter.GetHisto == 'function') {

               // this object used to copy ged setting from options and back,
               // do not (yet) allow to change options directly

/*               if (painter.ged === undefined)
                  painter.ged = { ndim: painter.Dimension(), Errors: 0, Style: 0, Contor: 1, Lego: 2 };

               painter.ged.mode3d = painter.mode3d ? 1 : 0;
               painter.ged.markers = painter.options.Mark;
               painter.ged.bar = painter.options.Bar;
               painter.ged.Lego = painter.options.Lego;
*/

               painter.options.Mode3Dindx = painter.options.Mode3D ? 1 : 0;

               var model = new JSONModel( { opts : painter.options } );

               // model.attachPropertyChange({}, painter.processTitleChange, painter);
               this.addFragment(oPage, "Hist", model);

               model.attachPropertyChange({ options: painter.options }, this.processHistModelChange, this);
            }
         }
      },

      onObjectRedraw : function(padpainter, painter) {
         if ((this.currentPadPainter !== padpainter) || (this.currentPainter !== painter)) return;

         console.log('GED sees selected object redraw');

         var page = this.getView().byId("ged_page");
         var cont = page.getContent();

         for (var n=0;n<cont.length;++n)
            if (cont[n] && cont[n].ged_fragment) {
               var model = cont[n].getModel();

               model.refresh();
            }
      },

      padEventsReceiver : function(evnt) {
         if (!evnt) return;

         if (evnt.what == "select")
            this.onObjectSelect(evnt.padpainter, evnt.painter);
         else if (evnt.what == "redraw")
            this.onObjectRedraw(evnt.padpainter, evnt.painter);
      }

   });

});
