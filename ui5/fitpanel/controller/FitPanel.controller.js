sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/unified/ColorPickerPopover',
   'sap/m/Button',
   'sap/m/Table',
   'sap/m/Dialog',
   'sap/m/List',
   'sap/m/InputListItem',
   'sap/m/Input',
   'sap/m/Label'
], function (GuiPanelController, JSONModel, ColorPickerPopover, Button, Table,
             Dialog, List, InputListItem, Input, Label) {

   "use strict";
   var count = 0;
   return GuiPanelController.extend("rootui5.fitpanel.controller.FitPanel", {

         //function called from GuiPanelController
      onPanelInit : function() {

         JSROOT.loadScript('rootui5sys/fitpanel/style/style.css');

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
         this._data = data; 
      },

      // Assign the new JSONModel to data      
      OnWebsocketMsg: function(handle, msg){

         if(msg.startsWith("MODEL:")){
            var json = msg.substr(6);
            var data = JSROOT.parse(json);

            if(data) {
               this.getView().setModel(new JSONModel(data));
               this._data = data;

               this.copyModel = JSROOT.extend({},data);
            }
         } else if (msg.startsWith("PARS:")) {
         	var data = JSROOT.parse(msg.substr(5));
         	this.showParametersDialog(data);
         }

      },

      //Fitting Button
      doFit: function() {
         //Keep the #times the button is clicked
         count++;
         //Data is a new model. With getValue() we select the value of the parameter specified from id
         var data = this.getView().getModel().getData();
         //var func = this.getView().byId("TypeXY").getValue();
         var func = this.getView().byId("selectedOpText").getText();
         //We pass the value from func to C++ fRealFunc
         data.fRealFunc = func;

         //Refresh the model
         this.getView().getModel().refresh();
         //Each time we click the button, we keep the current state of the model
         this.copyModel[count] = JSROOT.extend({},data);

         if (this.websocket)
            this.websocket.Send('DOFIT:'+this.getView().getModel().getJSON());

      },

      onPanelExit: function(){

      },

      resetPanel: function(oEvent){

         if(!this.copyModel) return;

         JSROOT.extend(this._data, this.copyModel);
         this.getView().getModel().updateBindings();
         this.byId("selectedOpText").setText("gaus");
         this.byId("OperationText").setValue("");
         return;
      },

      backPanel: function() {
         //Each time we click the button, we go one step back
         count--;
         if(count < 0) return;
         if(!this.copyModel[count]) return;

         JSROOT.extend(this._data, this.copyModel[count]);
         this.getView().getModel().updateBindings();
         return;
      },

      //Change the input text field. When a function is seleced, it appears on the text input field and
      //on the text area.
      onTypeXYChange: function(){
         var data = this.getView().getModel().getData();
         var linear = this.getView().getModel().getData().fSelectXYId;
         data.fFuncChange = linear;
         this.getView().getModel().refresh();

         //updates the text area and text in selected tab, depending on the choice in TypeXY ComboBox
         var func = this.getView().byId("TypeXY").getValue();
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

         var data = this.getView().getModel().getData();
         var lib = this.getView().getModel().getData().fLibrary;

         // same code as initialization
         data.fMethodMin = data.fMethodMinAll[parseInt(lib)];


         // refresh all UI elements
         this.getView().getModel().refresh();
         console.log("Method = ", data.fMethodMinAll[parseInt(lib)]);

      },
      //Change the combobox in Type Function
      //When the Type (TypeFunc) is changed (Predef etc) then the combobox with the funtions (TypeXY), 
      //is also changed 
      selectTypeFunc: function(){

         var data = this.getView().getModel().getData();

         var typeXY = this.getView().getModel().getData().fSelectTypeId;
         var dataSet = this.getView().getModel().getData().fSelectDataId;
         console.log("typeXY = " + dataSet);

         data.fTypeXY = data.fTypeXYAll[parseInt(typeXY)];

         this.getView().getModel().refresh();
         console.log("Type = ", data.fTypeXYAll[parseInt(typeXY)]);
      },

      //Change the selected checkbox of Draw Options 
      //if Do not Store is selected then No Drawing is also selected
      storeChange: function(){
         var data = this.getView().getModel().getData();
         var fDraw = this.getView().byId("noStore").getSelected();
         console.log("fDraw = ", fDraw);
         data.fNoStore = fDraw;
         this.getView().getModel().refresh();
         console.log("fNoDrawing ", data.fNoStore);
      },

      closeParametersDialog: function(is_ok) {
         if (is_ok && this.parData) {
         	var json = JSROOT.toJSON(this.parData);
         	if (this.websocket)
               this.websocket.Send("SETPARS:" + json);
         }

         this.parsDialog.close();
         this.parsDialog.destroy();
         delete this.parsDialog;

      },

      setParametersDialog: function(){
      	var func = this.getView().byId("TypeXY").getValue();
      	var msg = "GETPARS:" + func;
        if (this.websocket)
            this.websocket.Send(msg);
      },

      showParametersDialog: function(data){

         var aData = { Data: data.pars };

         this.parData = data;

         console.log("GOT PARAMETERS", data.pars.length, data.name, data.min);

         var oModel = new sap.ui.model.json.JSONModel(aData);
         sap.ui.getCore().setModel(oModel, "aDataData");

         var oColName = new sap.m.Column({ header: new sap.m.Label({text: "Name"})});
         var oColFix = new sap.m.Column({ header: new sap.m.Label({text: "Fix"})});
         var oColBound = new sap.m.Column({ header: new sap.m.Label({text: "Bound"})});
         //var oColValue = new sap.m.Column({ header: new sap.m.Label({text: "Value"})});
         var oColMin = new sap.m.Column({ header: new sap.m.Label({text: "Min"})});
         //var oColSetRange = new sap.m.Column({ header: new sap.m.Label({text: "Set Range"})});
         var oColMax = new sap.m.Column({ header: new sap.m.Label({text: "Max"})});
         //var oColStep = new sap.m.Column({ header: new sap.m.Label({text: "Step"})});
         //var oColErrors = new sap.m.Column({ header: new sap.m.Label({text: "Errors"})});

         var oNameTxt = new sap.m.TextArea({value: "{name}", height: "30px"});
         var oFixTxt = new sap.m.CheckBox({selected: "{Fix}"});
         var oBoundTxt = new sap.m.CheckBox({selected: "{Bound}"});
         //var oValueTxt = new sap.m.StepInput({min:"{min}", max:"{max}", value:"{value}", displayValuePrecision: 3, width: "120%" });
         //var oMinTxt = new sap.m.TextArea({ value:"{min}", height: "30px"});
         var oMinTxt = new sap.m.TextArea({value: "{min}", height: "30px"});
         //var oSetRangeTxt = new sap.m.RangeSlider({min:"{Min}", max:"{Max}", value:"{Value}"});
         //var oMaxTxt = new sap.m.TextArea({ value:"{max}", height: "30px"});
         var oMaxTxt = new sap.m.TextArea({value: "{max}", height: "30px"});
         //var oStepTxt = new sap.m.StepInput({min:"{min}", max:"{max}", value:"{Step}", displayValuePrecision: 3 });
         //var oErrorTxt = new sap.m.TextArea({value:"{error}", height: "30px", enabled:false});

         var oTableItems = new sap.m.ColumnListItem({vAlign:"Middle",cells:[oNameTxt,oFixTxt,oBoundTxt,
         										oMinTxt,oMaxTxt]});

         var oTable = new sap.m.Table({
         	id:"PrmsTable",
         	fixedLayout:false,
         	mode: sap.m.ListMode.SingleSelectMaster,
         	includeItemInSelection:true,
         	growing: true
         });

         oTable.addColumn(oColName);
         oTable.addColumn(oColFix);
         oTable.addColumn(oColBound);
         //oTable.addColumn(oColValue);
         oTable.addColumn(oColMin);
         //oTable.addColumn(oColSetRange);
         oTable.addColumn(oColMax);
         // oTable.addColumn(oColStep);
         // oTable.addColumn(oColErrors);

         oTable.bindAggregation("items","/Data",oTableItems,null);
         oTable.setModel(oModel);

         this.parsDialog = new Dialog({
            title: "Set Prarameters",
            beginButton: new Button({
               text: 'Cancel',
               press: this.closeParametersDialog.bind(this)
            }),
            endButton: new Button({
               text: 'Ok',
               press: this.closeParametersDialog.bind(this, true)
            })
         });

         this.parsDialog.addContent(oTable);

         this.parsDialog.addStyleClass("sapUiSizeCompact sapUiResponsiveMargin");

         this.parsDialog.open();
      },

      startExtraParametersDialog: function() {
         // placeholder for triggering new window with parameters editor only
      },

      //Cancel Button on Set Parameters Dialog Box
      onCancel: function(oEvent){
         oEvent.getSource().close();
      },

      colorPicker: function (oEvent) {
         this.inputId = oEvent.getSource().getId();
         if (!this.oColorPickerPopover) {
            this.oColorPickerPopover = new sap.ui.unified.ColorPickerPopover({
               colorString: "blue",
               mode: sap.ui.unified.ColorPickerMode.HSL,
               change: this.handleChange.bind(this)
            });
         }
         this.oColorPickerPopover.openBy(oEvent.getSource());
      },

      handleChange: function (oEvent) {
         var oView = this.getView();
         //oView.byId(this.inputId).setValue(oEvent.getParameter("colorString"));
         this.inputId = "";
         var color = oEvent.getParameter("colorString");
      },

      updateRange: function() {
         var data = this.getView().getModel().getData();
         var range = this.getView().byId("Slider").getRange();
         console.log("Slider " + range);

         //We pass the values from range array in JS to C++ fRange array
         data.fUpdateRange[0] = range[0];
         data.fUpdateRange[1] = range[1];
      },

   });

   return 
});
