sap.ui.define([
	'sap/m/MessageView',
	'sap/m/Dialog',
	'sap/m/Button',
	'sap/m/Bar',
	'sap/m/Title',
	'sap/m/MessageItem',
	'sap/m/MessageToast',
	'sap/m/Link',
	'sap/ui/core/mvc/Controller',
	'sap/ui/model/json/JSONModel'
], function(MessageView, Dialog, Button, Bar, Title, MessageItem, MessageToast, Link, Controller, JSONModel) {
	"use strict";

	var oMessagePopover;

	return Controller.extend("rootui5.eve7.controller.ClientLog", {
		onInit: function () {
			// create any data and a model and set it to the view
			var oLink = new Link({
				text: "Show more information",
				href: "http://sap.com",
				target: "_blank"
			});

			var oMessageTemplate = new MessageItem({
				type: '{type}',
				title: '{title}',
				activeTitle: "{active}",
				description: '{description}',
				subtitle: '{subtitle}',
				counter: '{counter}',
				link: oLink
			});

			this.oMessageView = new MessageView({
				class:"sapUiSizeCompact",
				showDetailsPageHeader: false,
				itemSelect: function () {
					oBackButton.setVisible(true);
				},
				items: {
					path: '/',
					template: oMessageTemplate
				},
				activeTitlePress: function () {
					MessageToast.show('Active title is pressed');
				}
			});


			var oBackButton = new Button({
				icon: "sap-icon://nav-back",
				visible: false,
				press: function ()
				{
					that.oMessageView.navigateBack();
					this.setVisible(false);
				}
			});

			/*
			this.getView().setModel(JSROOT.EVE.consoleModel);
			this.oMessageView.setModel(JSROOT.EVE.consoleModel);
			this.byId("messagePopoverBtn").addDependent(this.oMessageView); // ? AMT
			this.byId("messagePopoverBtn").setModel(JSROOT.EVE.consoleModel);
*/
			
			this.oDialog = new Dialog({
				resizable: true,
				content: this.oMessageView,
				state: 'Error',
				beginButton: new Button({
					press: function () {
						this.getParent().close();
					},
					text: "Close"
				}),
				customHeader: new Bar({
					contentLeft: [oBackButton],
					contentMiddle: [
						new Title({text: "Client Log"})
					]
				}),
				contentHeight: "50%",
				contentWidth: "50%",
				verticalScrolling: false
			});
			this.oDialog.addStyleClass("sapUiSizeCompact"); 
		},

		getButton: function() {
			return this.byId("messagePopoverBtn");
		},

		// Display the button type according to the message with the highest severity
		// The priority of the message types are as follows: Error > Warning > Success > Info
		buttonTypeFormatter: function () {
			var sHighestSeverityIcon;
			var aMessages = this.getView().getModel().oData;

			aMessages.forEach(function (sMessage) {
				switch (sMessage.type) {
					case "Error":
						sHighestSeverityIcon = "Negative";
						break;
					case "Warning":
						sHighestSeverityIcon = sHighestSeverityIcon !== "Negative" ? "Critical" : sHighestSeverityIcon;
						break;
					case "Success":
						sHighestSeverityIcon = sHighestSeverityIcon !== "Negative" && sHighestSeverityIcon !== "Critical" ?  "Success" : sHighestSeverityIcon;
						break;
					default:
						sHighestSeverityIcon = !sHighestSeverityIcon ? "Neutral" : sHighestSeverityIcon;
						break;
				}
			});

			return sHighestSeverityIcon;
		},

		// Display the number of messages with the highest severity
		highestSeverityMessages: function () {
			var sHighestSeverityIconType = this.buttonTypeFormatter();
			var sHighestSeverityMessageType;

			switch (sHighestSeverityIconType) {
				case "Negative":
					sHighestSeverityMessageType = "Error";
					break;
				case "Critical":
					sHighestSeverityMessageType = "Warning";
					break;
				case "Success":
					sHighestSeverityMessageType = "Success";
					break;
				default:
					sHighestSeverityMessageType = !sHighestSeverityMessageType ? "Information" : sHighestSeverityMessageType;
					break;
			}

			return this.getView().getModel().oData.reduce(function(iNumberOfMessages, oMessageItem) {
				return oMessageItem.type === sHighestSeverityMessageType ? ++iNumberOfMessages : iNumberOfMessages;
			}, 0);
		},

		// Set the button icon according to the message with the highest severity
		buttonIconFormatter: function () {
			var sIcon;
			var aMessages = this.getView().getModel().oData;

			aMessages.forEach(function (sMessage) {
				switch (sMessage.type) {
					case "Error":
						sIcon = "sap-icon://message-error";
						break;
					case "Warning":
						sIcon = sIcon !== "sap-icon://message-error" ? "sap-icon://message-warning" : sIcon;
						break;
					case "Success":
						sIcon = "sap-icon://message-error" && sIcon !== "sap-icon://message-warning" ? "sap-icon://message-success" : sIcon;
						break;
					default:
						sIcon = !sIcon ? "sap-icon://message-information" : sIcon;
						break;
				}
			});

			return sIcon;
		},
		
		handleMessagePopoverPress: function (oEvent) {
			this.oMessageView.navigateBack();
			this.oDialog.open();
			
		}

	});

});
