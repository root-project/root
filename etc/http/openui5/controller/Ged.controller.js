sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/m/Dialog',
   'sap/m/Button',
   'sap/ui/commons/ColorPicker'
], function (Controller, JSONModel, Dialog, Button, ColorPicker) {
   "use strict";

   return Controller.extend("sap.ui.jsroot.controller.Ged", {

      currentPainter: null,

      gedFragments : [],

      onInit : function() {
         var model = new JSONModel({ SelectedClass: "none" });
         this.getView().setModel(model);
      },

      onExit : function() {
         console.log('exit GED editor');
         this.currentPainter = null; // remove cross ref
      },

      getFragment : function(kind, force) {
          var fragm = this.gedFragments[kind];
          if (!fragm && force)
             fragm = this.gedFragments[kind] = sap.ui.xmlfragment(this.getView().getId(), "sap.ui.jsroot.view." + kind, this);
          return fragm;
      },

      /// function called when user changes model property
      /// data object includes _kind, _painter and _handle (optionally)
      modelPropertyChange : function(evnt, data) {
         var pars = evnt.getParameters();
         // console.log('Model property changes', pars.path, pars.value, data._kind);

         if (data._handle && (typeof data._handle.verifyDirectChange === 'function'))
            data._handle.verifyDirectChange(data._painter);

         if (data._painter && data._painter.AttributeChange)
            data._painter.AttributeChange(data._kind, pars.path.substr(1), pars.value);

         if (this.currentPadPainter)
            this.currentPadPainter.Redraw();
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
            var model = new JSONModel( painter.lineatt );
            var fragm = this.getFragment("TAttLine", true);
            model.attachPropertyChange({ _kind: "TAttLine", _painter: painter, _handle: painter.lineatt }, this.modelPropertyChange, this);
            fragm.setModel(model);
            oPage.addContent(fragm);
         }

         if (painter.fillatt && painter.fillatt.used) {
            var model = new JSONModel( painter.fillatt );
            var fragm = this.getFragment("TAttFill", true);
            model.attachPropertyChange({ _kind: "TAttFill", _painter: painter, _handle: painter.fillatt }, this.modelPropertyChange, this);
            fragm.setModel(model);
            oPage.addContent(fragm);
         }

         if (painter.markeratt && painter.markeratt.used) {
            var model = new JSONModel( painter.markeratt );
            var fragm = this.getFragment("TAttMarker", true);
            model.attachPropertyChange({ _kind: "TAttMarker", _painter: painter, _handle: painter.markeratt }, this.modelPropertyChange, this);
            fragm.setModel(model);
            oPage.addContent(fragm);
         }

      },

      // TODO: special controller for each fragment?

      makeColorDialog : function(fragment, property) {

         var that = this, fragm = this.getFragment(fragment);

         if (!fragm) return null;

         if (!this.colorPicker)
            this.colorPicker = new ColorPicker("colorPicker");

         if (!this.colorDialog) {
            this.colorDialog = new Dialog({
               title: 'Select color',
               content: this.colorPicker,
               beginButton: new Button({
                  text: 'Apply',
                  press: function () {
                     if (that.colorPicker) {
                        var fragm = that.getFragment(that.colorFragment);
                        var col = that.colorPicker.getColorString();

                        fragm.getModel().setProperty(that.colorProperty, col);
                        fragm.getModel().firePropertyChange({ path: that.colorProperty, value: col });
                     }
                     that.colorDialog.close();
                  }
               }),
               endButton: new Button({
                  text: 'Cancel',
                  press: function () {
                     that.colorDialog.close();
                  }
               })
            });
         }

         this.colorFragment = fragment;
         this.colorProperty = property;

         var col = fragm.getModel().getProperty(property);

         this.colorPicker.setColorString(col);
         this.colorDialog.open();
      },

      // TODO: make it generic
      processTAttLine_Color : function() {
         this.makeColorDialog('TAttLine', '/color');
      },

      // TODO: make it generic
      processTAttFill_Color : function() {
         this.makeColorDialog('TAttFill', '/color');
      },

      processTAttMarker_Color : function() {
         this.makeColorDialog('TAttMarker', '/color');
      }

   });

});
