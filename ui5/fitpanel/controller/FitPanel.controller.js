sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/m/Button',
   'sap/m/Table',
   'sap/m/Dialog',
   'sap/m/List',
   'sap/m/Input',
   'sap/m/Label',
   'sap/m/CheckBox',
   'sap/m/Column',
   'sap/m/ColumnListItem'
], function (GuiPanelController, JSONModel, mButton, mTable,
             mDialog, mList, mInput, mLabel, mCheckBox, mColumn, mColumnListItem) {

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

         var libMin = this.getView().byId("MethodMin").getValue();
         data.fMinLibrary = libMin;
         console.log("Method Min " + libMin);

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
            // first convert back to float values

            for (var i=0;i<this.parData.pars.length;++i) {
               var par = this.parData.pars[i];
               // convert value into floats back
               this.toFloat(par, "value");
               this.toFloat(par, "error");
               this.toFloat(par, "min");
               this.toFloat(par, "max");
            }

            var json = JSROOT.toJSON(this.parData);
            if (this.websocket)
               this.websocket.Send("SETPARS:" + json);
                 console.log("JSON " + json)
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

      toString: function(par, field, digits) {
         if (par[field] == Math.round(par[field])) digits = 0;
         par[field+"Txt"] = par[field+"Txt0"] = par[field].toFixed(digits);
      },

      toFloat: function(par, field) {
         if (par[field+"Txt"] !== par[field+"Txt0"]) {
            var res = parseFloat(par[field+"Txt"]);
            if (!isNaN(res)) par[field] = res;
         }
         delete par[field+"Txt"];
         delete par[field+"Txt0"];
      },

      showParametersDialog: function(data){

         var aData = { Data: data.pars };

         this.parData = data;

         // prepare text formatting
         for (var i=0;i<data.pars.length;++i) {
            var par = data.pars[i];
            // convert value into strings, requird by sap.m.Input
            this.toString(par, "value", 3);
            this.toString(par, "error", 3);
            this.toString(par, "min", 3);
            this.toString(par, "max", 3);
         }

         var oModel = new sap.ui.model.json.JSONModel(aData);
         // sap.ui.getCore().setModel(oModel, "aDataData");

         var oTableItems = new mColumnListItem({ vAlign:"Middle", cells:[
              new mLabel({ text: "{name}" }),
              new mCheckBox({ selected: "{fixed}" }),
              // new mCheckBox({ selected: "{Bound}" }),
              new mInput({ value: "{valueTxt}", type: "Number", width: "75px" }),
              new mInput({ value: "{minTxt}", type: "Number", width: "75px" }),
              new mInput({ value: "{maxTxt}", type: "Number", width: "75px" }),
              new mInput({ value: "{errorTxt}", type: "Number", width: "75px" })
         ]});

         var oTable = new mTable({
            id: "PrmsTable",
            fixedLayout: false,
            mode: sap.m.ListMode.SingleSelectMaster,
            includeItemInSelection: true,
            growing: true,
            columns: [
                new mColumn({ header: new mLabel({text: "Name"})}),
                new mColumn({ header: new mLabel({text: "Fix"})}),
                // new mColumn({ header: new mLabel({text: "Bound"})}),
                new mColumn({ header: new mLabel({text: "Value"})}),
                new mColumn({ header: new mLabel({text: "Min"})}),
                new mColumn({ header: new mLabel({text: "Max"})}),
                new mColumn({ header: new mLabel({text: "Errors"})})
            ]
         });

         oTable.bindAggregation("items","/Data",oTableItems,null);
         oTable.setModel(oModel);

         this.parsDialog = new mDialog({
            title: "Set Prarameters",
            beginButton: new mButton({
               text: 'Cancel',
               press: this.closeParametersDialog.bind(this)
            }),
            endButton: new mButton({
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

      updateRange: function() {
         var data = this.getView().getModel().getData();
         var range = this.getView().byId("Slider").getRange();
         console.log("Slider " + range);

         //We pass the values from range array in JS to C++ fRange array
         data.fUpdateRange[0] = range[0];
         data.fUpdateRange[1] = range[1];
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
         var oButtonContour = this.getView().byId("colorContour");
         var oButtonInnerContour = oButtonContour.$().find('.sapMBtnInner');
         oButtonInnerContour.css('background',color);
         oButtonInnerContour.css('color','#FFFFFF');
         oButtonInnerContour.css('text-shadow','1px 1px 2px #333333');

         var oButtonConf = this.getView().byId("colorConf");
         var oButtonInnerConf = oButtonConf.$().find('.sapMBtnInner');
         oButtonInnerConf.css('background',color);
         oButtonInnerConf.css('color','#FFFFFF');
         oButtonInnerConf.css('text-shadow','1px 1px 2px #333333');
},

   });

   return
});
