sap.ui.define([
   'sap/ui/core/mvc/Controller'
], function (Controller) {
   "use strict";

   return Controller.extend("sap.ui.jsroot.controller.Inspector", {

       renderObj:  null,

       setObject: function(obj) {
          this.renderObj = obj;
       },

       onAfterRendering: function() {
          if (!this.renderObj) return;

          if (this.ipainter) this.ipainter.Cleanup();
          delete this.ipainter;

          var pthis = this;
          JSROOT.draw(this.getView().getDomRef(), this.renderObj, 'inspect', function(ipainter) {
             pthis.ipainter = ipainter;
          });
       },

       onInit : function() {
       },

       onExit : function() {
          if (this.ipainter) this.ipainter.Cleanup();
          delete this.ipainter;
       }
   });

});
