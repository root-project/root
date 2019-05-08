sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel'
], function (GuiPanelController, JSONModel) {

   "use strict";

   var colorConf = "rgb(0,0,0)";

   return GuiPanelController.extend("rootui5.fitpanel.controller.FitPanel", {

         //function called from GuiPanelController
      onPanelInit : function() {

         // WORKAROUND, need to be FIXED IN THE FUTURE
         JSROOT.loadScript('rootui5sys/fitpanel/style/style.css');

         // for linev.github.io
         // JSROOT.loadScript('../rootui5/fitpanel/style/style.css');

         var id = this.getView().getId();
         this.inputId = "";
         var opText = this.getView().byId("OperationText");
         var data = {
               //fDataSet:[ { fId:"1", fSet: "----" } ],
               fSelectDataId: "2",
               // fMinRange: -4,
               // fMaxRange: 4,
               fStep: 0.01,
               fRange: [-4,4],
               fUpdateRange: [-4,4]
         };
         this.getView().setModel(new JSONModel(data));
      },

      // returns actual model object of class RFitPanel6Model
      data: function() {
        return this.getView().getModel().getData();
      },

      // cause refresh of complete fit panel
      refresh: function() {
         this.getView().getModel().refresh();
      },

      // Assign the new JSONModel to data
      OnWebsocketMsg: function(handle, msg) {

         if(msg.startsWith("MODEL:")){
            var data = JSROOT.parse(msg.substr(6));
            if(data) {
               data.fFuncList = data.fFuncListAll[parseInt(data.fSelectTypeFunc)];
               data.fMethodMin = data.fMethodMinAll[parseInt(data.fLibrary)];

               this.getView().setModel(new JSONModel(data));
            }
         } else if (msg.startsWith("PARS:")) {

            this.data().fFuncPars = JSROOT.parse(msg.substr(5));

            this.refresh();
         }
      },

      //Fitting Button
      doFit: function() {
         //Keep the #times the button is clicked
         //Data is a new model. With getValue() we select the value of the parameter specified from id

         var libMin = this.getView().byId("MethodMin").getValue();
         var errorDefinition = parseFloat(this.getView().byId("errorDef").getValue());
         var maxTolerance = parseFloat(this.getView().byId("maxTolerance").getValue());
         var maxInterations = Number(this.getView().byId("maxInterations").getValue());

         var data = this.data();

         data.fMinLibrary = libMin;
         data.fErrorDef = errorDefinition;
         data.fMaxTol = maxTolerance;
         data.fMaxInter = maxInterations;

         //Refresh the model
         this.refresh();
         //Each time we click the button, we keep the current state of the model

         // TODO: skip "fMethodMin" and "fFuncList" from output object
         // Requires changes in JSROOT.toJSON(), can be done after REVE-selection commit

         if (this.websocket)
            this.websocket.Send('DOFIT:'+this.getView().getModel().getJSON());

      },

      onPanelExit: function(){

      },

      /** Returns selected function name */
      getSelectedFunc: function() {
         return this.data().fSelectedFunc;
      },

      //Change the input text field. When a function is seleced, it appears on the text input field and
      //on the text area.
      onSelectedFuncChange: function(){

         var func = this.getSelectedFunc();

         this.data().fFuncChange = func; // FIXME: seems to be, not required
         this.refresh();

         if (this.websocket && func)
            this.websocket.Send("GETPARS:" + func);

         //updates the text area and text in selected tab, depending on the choice in TypeXY ComboBox
         this.byId("OperationText").setValueLiveUpdate();
         this.byId("OperationText").setValue(func);
         this.byId("selectedOpText").setText(func);
      },

      operationTextChange: function(oEvent) {
         var newValue = oEvent.getParameter("value");
         this.byId("selectedOpText").setText(newValue);
      },


      //change the combo box in Minimization Tab --- Method depending on Radio Buttons values
      selectRB: function(){

         var data = this.data();

         // same code as initialization
         data.fMethodMin = data.fMethodMinAll[parseInt(data.fLibrary)];

         // refresh all UI elements
         this.refresh();
      },

      //Change the combobox in Type Function
      //When the Type (TypeFunc) is changed (Predef etc) then the combobox with the funtions (TypeXY),
      //is also changed
      selectTypeFunc: function(){

         var data = this.data();

         // console.log("Func Type = " + data.fSelectTypeFunc);
         data.fFuncList = data.fFuncListAll[parseInt(data.fSelectTypeFunc)];

         this.refresh();
      },

      //Change the selected checkbox of Draw Options
      //if Do not Store is selected then No Drawing is also selected
      storeChange: function(){
         var data = this.data();
         var fDraw = this.getView().byId("noStore").getSelected();
         console.log("fDraw = ", fDraw);
         data.fNoStore = fDraw;
         this.refresh();
         console.log("fNoDrawing ", data.fNoStore);
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


      updateRange: function() {
         var data = this.data();
         var range = this.getView().byId("Slider").getRange();
         console.log("Slider " + range);

         //We pass the values from range array in JS to C++ fRange array
         data.fUpdateRange[0] = range[0];
         data.fUpdateRange[1] = range[1];
      },

      colorPickerContour: function (oEvent) {
         this.inputId = oEvent.getSource().getId();
         if (!this.oColorPickerPopoverContour) {
            this.oColorPickerPopoverContour = new sap.ui.unified.ColorPickerPopover({
               colorString: "blue",
               mode: sap.ui.unified.ColorPickerMode.HSL,
               change: this.handleChangeContour.bind(this)
            });
         }
         this.oColorPickerPopoverContour.openBy(oEvent.getSource());
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
