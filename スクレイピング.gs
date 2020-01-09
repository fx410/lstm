//スクレイピングクラス
var exchange = {
  //シート取得関数
  getSheet: function() {
    if(this.getSheet.sheet) { return this.getSheet.sheet; }

    var SHEET_URL = PropertiesService.getScriptProperties().getProperty('SHEET_URL');
    if (!SHEET_URL) {
      throw 'You should set "SHEET_URL" property from [File] > [Project properties] > [Script properties]';
    }

    var sheets = SpreadsheetApp.openByUrl(SHEET_URL);
    this.getSheet.sheet = sheets;
    return this.getSheet.sheet;
  },
  
   // call exchange API関数
  callExchangeAPI: function() {
    var now      = new Date(),
        url      = "https://www.gaitameonline.com/rateaj/getrate",
        response = UrlFetchApp.fetch(url),
        content  = response.getContentText(),
        fx       = JSON.parse(content);

    fx.date = now;
    return fx;
  },

  //APIから取得したJSONデータをシートに書き込む関数
  // Write exchange data (JSON) to the Google Sheet
  writeSheets: function(ex_json) {
    var sheet = this.getSheet().getActiveSheet();

    // get last row to add exchange data
    var last_row = sheet.getLastRow() + 1;

    var col = 1;
    sheet.getRange(last_row, col++).setValue(ex_json.date);

    for each(var quote in ex_json.quotes) {
      if(quote.currencyPairCode == 'GBPJPY'){
        sheet.getRange(last_row, col++).setValue(quote.currencyPairCode);
        sheet.getRange(last_row, col++).setValue(quote.high);
        sheet.getRange(last_row, col++).setValue(quote.low);
        sheet.getRange(last_row, col++).setValue(quote.ask);
        sheet.getRange(last_row, col++).setValue(quote.bid);
        sheet.getRange(last_row, col++).setValue(quote.open);
      }
    }
  },
  
  //大きいトレンドの判断
   big_trend: function(){
    return 0;
  },
  
  //小さいトレンドの判断(１時間前と比較)
  small_trend: function(sheet){
    var last_row = sheet.getLastRow();
    var past_ask = sheet.getRange(last_row-6,5).getValue();
    var now_ask = sheet.getRange(last_row,5).getValue();
    if(now_ask-past_ask>0){
      return 1;
    }else{
      return 0;
    }
  },
  
  //最高値更新
  over_high: function(sheet){
    var last_row = sheet.getLastRow();
    var past_high = sheet.getRange(last_row-1,3).getValue();
    var now_high = sheet.getRange(last_row,3).getValue();
    return now_high-past_high;
  },
  
  //最安値更新
  over_low: function(sheet){
    var last_row = sheet.getLastRow();
    var past_low = sheet.getRange(last_row-1,4).getValue();
    var now_low = sheet.getRange(last_row,4).getValue();
    return now_low-past_low;
  },
  
  //移動平均線
  ema: function(sheet){
    var space = 20;
    var last_row = sheet.getLastRow();
    var space_row = sheet.getRange(last_row-space, 5, space, 1).getValues();
    var sum = 0;
    for (i=0;1<space;i++){
         sum+=space_row[i][0];
    }
    Logger.log(space_row[1][0]);
    
  },
  
  //トレンド判断の書き込み
  writeTrend: function(){
    var sheet = this.getSheet().getActiveSheet();
    var sheet2 = this.getSheet().getSheetByName('data_1day');
    var last_row = sheet.getLastRow();
    //sheet.getRange(last_row,8).setValue(this.big_trend());
    //sheet.getRange(last_row,9).setValue(this.small_trend(sheet));
    //sheet.getRange(last_row,10).setValue(this.over_high(sheet));
    //sheet.getRange(last_row,11).setValue(this.over_low(sheet));
  },
  
  //古いデータを消去
  delOld: function(){
    var sheet = this.getSheet().getActiveSheet();
    sheet.deleteRow(2);
    sheet.getRange(2,10,1,2).clear();
    sheet.getRange(2,9,6,1).clear();
  }
}

