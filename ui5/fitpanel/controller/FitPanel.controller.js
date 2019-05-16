sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/model/Filter',
   'sap/ui/model/FilterOperator',
   "sap/m/ColorPalettePopover"
], function (GuiPanelController, JSONModel, Filter, FilterOperator, ColorPalettePopover) {

   "use strict";

   var colorConf = "rgb(0,0,0)";

   return GuiPanelController.extend("rootui5.fitpanel.controller.FitPanel", {

         //function called from GuiPanelController
      onPanelInit : function() {

         // WORKAROUND, need to be FIXED IN THE FUTURE
         JSROOT.loadScript('rootui5sys/fitpanel/style/style.css');

         // for linev.github.io
         // JSROOT.loadScript('../rootui5/fitpanel/style/style.css');

         this.inputId = "";
         var data = {
               fDataSet:[ { key:"1", value: "----" } ],
               fSelectedData: "1",
               fMinRangeX: -1,
               fShowRangeX: false,
               fMaxRangeX: 1,
               fStepX: 0.1,
               fRangeX: [-1,1],
               fShowRangeY: false,
               fMinRangeY: -1,
               fMaxRangeY: 1,
               fStepY: 0.1,
               fRangeY: [-1,1]

         };
         this.getView().setModel(new JSONModel(data));
      },

      // returns actual model object of class RFitPanelModel
      data: function() {
        return this.getView().getModel().getData();
      },

      // cause refresh of complete fit panel
      refresh: function() {
         this.doing_refresh = true;
         this.getView().getModel().refresh();
         this.doing_refresh = false;
      },

      sendModel: function(prefix) {
         if (!prefix || (typeof prefix!="string")) {
            // this is protection against infinite loop
            // may happen if by refresh of model any callbacks are activated and trying update server side
            // this should be prevented
            if (this.doing_refresh) return;
            prefix = "UPDATE:";
         }

         if (this.websocket)
            this.websocket.Send(prefix + this.getView().getModel().getJSON());
      },

      // Assign the new JSONModel to data
      OnWebsocketMsg: function(handle, msg) {

         if(msg.startsWith("MODEL:")) {
            var data = JSROOT.parse(msg.substr(6));

            if(data) {
               this.getView().setModel(new JSONModel(data));

               this.verifySelectedMethodMin(data);

               this.refresh();
            }
         } else if (msg.startsWith("PARS:")) {

            this.data().fFuncPars = JSROOT.parse(msg.substr(5));

            this.refresh();
         }
      },

      // Update Button
      doUpdate: function() {
         if (this.websocket)
            this.websocket.Send("RELOAD");
      },

      // Fit Button
      doFit: function() {
         this.sendModel("DOFIT:");
      },

      // Draw Button
      doDraw: function() {
         this.sendModel("DODRAW:");
      },

      onPanelExit: function(){
      },

      // when selected data is changing - cause update of complete model
      onSelectedDataChange: function() {
         this.sendModel();
      },

      // when change function many elements may be changed - resync model
      onSelectedFuncChange: function() {
         this.sendModel();
      },

      // approve current fSelectMethodMin value - and change if require
      verifySelectedMethodMin: function(data) {

         this.getView().byId("MethodMin").getBinding("items").filter(new Filter("lib", FilterOperator.EQ, data.fLibrary));

         var first = 0;

         for (var k=0;k<data.fMethodMinAll.length;++k) {
            var item = data.fMethodMinAll[k];
            if (item.lib != data.fLibrary) continue;
            if (!first) first = item.id;
            if (item.id === data.fSelectMethodMin) return;
         }

         data.fSelectMethodMin = first;
      },

      //change the combo box in Minimization Tab --- Method depending on Radio Buttons values
      selectMinimizationLibrary: function() {
         this.verifySelectedMethodMin(this.data());

         // refresh all UI elements
         this.refresh();
      },

      onContourPar1Change: function() {
         var data = this.data();
         if (data.fContourPar1Id == data.fContourPar2Id) {
            var par2 = parseInt(data.fContourPar2Id);
            if (par2 > 0) par2--; else par2 = 1;
            data.fContourPar2Id = par2.toString();
            this.refresh();
         }
      },

      onContourPar2Change: function() {
         var data = this.data();
         if (data.fContourPar1Id == data.fContourPar2Id) {
            var par1 = parseInt(data.fContourPar1Id);
            if (par1 > 0) par1--; else par1 = 1;
            data.fContourPar1Id = par1.toString();
            this.refresh();
         }
      },

      drawContour: function() {

      	var contourPoints = this.byId("contourPoints").getValue();
         var contourPar1 = parseInt(this.byId("ContourPar1").getSelectedKey());
         var contourPar2 = parseInt(this.byId("ContourPar2").getSelectedKey());
         var confLevel = this.byId("ConfLevel").getValue();
         var colorContourNum = (String((this.colorContour.replace( /^\D+/g, '')).replace(/[()]/g, ''))).split(',');

         var data = this.data();
         data.fContourPoints = contourPoints;
      	data.fContourPar1 = contourPar1;
      	data.fContourPar2 = contourPar2;
         data.fColorContour = colorContourNum;

         console.log("COLOR ", colorContourNum, typeof colorContourNum, " origin ", this.colorContour);
       //   var colConfN = colorConf.replace( /^\D+/g, '');
       //   var colorConfNum = colConfN.replace(/[()]/g, '');
      	// data.fConfLevel = colorConfNum;

	  	  this.refresh();
        //Each time we click the button, we keep the current state of the model
        if (this.websocket)
            this.websocket.Send('SETCONTOUR:'+this.getView().getModel().getJSON());
      },

      drawScan: function() {
      	var data = this.data();
      	data.fScanPoints = this.byId("scanPoints").getValue();
      	data.fScanPar = parseInt(this.byId("ScanPar").getSelectedKey());
      	data.fScanMin = this.byId("scanMin").getValue();
      	data.fScanMax = this.byId("scanMax").getValue();

      	this.refresh();
         //Each time we click the button, we keep the current state of the model
         if (this.websocket)
            this.websocket.Send('SETSCAN:'+this.getView().getModel().getJSON());

      },

      pressApplyPars: function() {
         var json = JSROOT.toJSON(this.data().fFuncPars);

         if (this.websocket)
            this.websocket.Send("SETPARS:" + json);
      },

      colorPickerContour: function (oEvent) {
         // official ROOT colors 1 .. 15, more is not supported by ColorPalettePopover
         var colorMap = ['black','red','green','blue','yellow','magenta','cyan', '#59d354','#5954d8',
                         '#fefefe', '#c0b6ac','#4c4c4c','#666666','#7f7f7f', '#999999'];
         // ,, '#b2b2b2','#cccccc','#e5e5e5','#f2f2f2','#ccc6aa','#ccc6aa','#c1bfa8','#bab5a3','#b2a596','#b7a39b','#ad998c','#9b8e82','#876656','#afcec6'];

          var oCPPop = new ColorPalettePopover( {
             defaultColor: "cyan",
             colors: colorMap,
             colorSelect: function(event) {
                console.log('select ', event.getParameters().value);
             }
          });

          oCPPop.openBy(oEvent.getSource());
      },



      handleChangeContour: function (oEvent) {
         var oView = this.getView();
         this.inputId = "";
         var color1 = oEvent.getParameter("colorString");
         var oButtonContour = this.getView().byId("colorContour");
         var oButtonInnerContour = oButtonContour.$().find('.sapMBtnInner');
         oButtonInnerContour.css('background',color1);
         oButtonInnerContour.css('color','#FFFFFF');
         oButtonInnerContour.css('text-shadow','1px 1px 2px #333333');

         this.colorContour = color1;
         return this.colorContour;
	  },

	  colorPickerConf: function (oEvent) {
         this.inputId = oEvent.getSource().getId();
         if (!this.oColorPickerPopoverConf) {
            this.oColorPickerPopoverConf = new sap.ui.unified.ColorPickerPopover({
               colorString: "blue",
               mode: sap.ui.unified.ColorPickerMode.HSL,
               change: this.handleChangeConf.bind(this)
            });
         }
         this.oColorPickerPopoverConf.openBy(oEvent.getSource());
      },

      handleChangeConf: function (oEvent) {
         var oView = this.getView();
         this.inputId = "";
         var color2 = oEvent.getParameter("colorString");
         var oButtonContour = this.getView().byId("colorConf");
         var oButtonInnerContour = oButtonContour.$().find('.sapMBtnInner');
         oButtonInnerContour.css('background',color2);
         oButtonInnerContour.css('color','#FFFFFF');
         oButtonInnerContour.css('text-shadow','1px 1px 2px #333333');

         colorConf = color2;
         return colorConf;
	  }

   });

   return
});
