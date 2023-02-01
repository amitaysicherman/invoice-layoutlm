# invoice-layoutlm 
This project used to predict different fields in English invoices ('date', 'company,' 'address,' 'total') using the layoutLM model finetuning on the SROIE dataset.

# Project structure
1. layoutlm -code for layoutlm model
2. model - pre-trained models
3. fine_tune_layoutlm_on_sroie.ipynb - colab notebook used to finetuning.
4. predict.py - the prediction code.
5. predict_examples.py - show an example of the model's prediction of some examples invoices.
6. examples - invoices images and OCR examples.

# How to use
The function in predict.py get json input (see below), and returns the labels according to the model prediction.
The labels shape is : 
```
{
'date': [..], 
'company': [], 
'address': [], 
'total': []
}
```



# Prediction from input string
The input string is one line json with the following shape:
```
{
   "img_height": The height of the image,
   "img_width": The widtt of the image,
   "tokens": List of tokens from the OCR.
}
```
While each token in the list has the following shape:
```
{
 "upper_left_x": X coordinate of the upper left corner.
 "upper_left_y": Y coordinate of the upper left corner. 
 "lower_right_x" X coordinate of the lower right corner.
 "lower_right_y" Y coordinate of the lower right corner.
 "words": Words separated by a space
}
```
###All coordinates are not normalized after rotation of the image.

# Input String Example

##One line:
```
{"img_height": 1678, "img_width": 884, "tokens": [{"upper_left_x": "46", "upper_left_y": "276", "lower_right_x": "780", "lower_right_y": "317", "words": "HON HWA HARDWARE TRADING"}, {"upper_left_x": "122", "upper_left_y": "325", "lower_right_x": "699", "lower_right_y": "366", "words": "COMPANY REG. NO. : 001055194X"}, {"upper_left_x": "45", "upper_left_y": "368", "lower_right_x": "762", "lower_right_y": "410", "words": "NO 37  JALAN MANIS 7  TAMAN SEGAR "}, {"upper_left_x": "106", "upper_left_y": "412", "lower_right_x": "709", "lower_right_y": "454", "words": "56100 CHERAS  KUALA LUMPUR."}, {"upper_left_x": "268", "upper_left_y": "457", "lower_right_x": "551", "lower_right_y": "491", "words": "+603-9130 2672"}, {"upper_left_x": "208", "upper_left_y": "503", "lower_right_x": "601", "lower_right_y": "540", "words": "GST REG : 001125220352"}, {"upper_left_x": "283", "upper_left_y": "562", "lower_right_x": "545", "lower_right_y": "604", "words": "TAX INVOICE"}, {"upper_left_x": "50", "upper_left_y": "640", "lower_right_x": "249", "lower_right_y": "678", "words": "CB# : 87870"}, {"upper_left_x": "418", "upper_left_y": "646", "lower_right_x": "792", "lower_right_y": "681", "words": "21/09/2017 10:20:37 AM"}, {"upper_left_x": "50", "upper_left_y": "680", "lower_right_x": "250", "lower_right_y": "717", "words": "M# : C2 - 0"}, {"upper_left_x": "51", "upper_left_y": "724", "lower_right_x": "172", "lower_right_y": "760", "words": "CASHIER"}, {"upper_left_x": "203", "upper_left_y": "729", "lower_right_x": "318", "lower_right_y": "762", "words": "CASH1 -"}, {"upper_left_x": "50", "upper_left_y": "785", "lower_right_x": "107", "lower_right_y": "822", "words": "QTY"}, {"upper_left_x": "131", "upper_left_y": "782", "lower_right_x": "300", "lower_right_y": "824", "words": "DESCRIPTION"}, {"upper_left_x": "546", "upper_left_y": "782", "lower_right_x": "628", "lower_right_y": "816", "words": "PRICE"}, {"upper_left_x": "661", "upper_left_y": "783", "lower_right_x": "810", "lower_right_y": "821", "words": "TOTAL(RM)"}, {"upper_left_x": "57", "upper_left_y": "831", "lower_right_x": "462", "lower_right_y": "868", "words": "0.9 3/4\" ALUMINIUM ROD"}, {"upper_left_x": "558", "upper_left_y": "832", "lower_right_x": "631", "lower_right_y": "864", "words": "6.00"}, {"upper_left_x": "680", "upper_left_y": "835", "lower_right_x": "813", "lower_right_y": "867", "words": "5.40 SR"}, {"upper_left_x": "86", "upper_left_y": "880", "lower_right_x": "390", "lower_right_y": "915", "words": "5 PVC WALLPLUG"}, {"upper_left_x": "562", "upper_left_y": "884", "lower_right_x": "627", "lower_right_y": "913", "words": "1.00"}, {"upper_left_x": "684", "upper_left_y": "880", "lower_right_x": "814", "lower_right_y": "918", "words": "5.00 SR"}, {"upper_left_x": "119", "upper_left_y": "921", "lower_right_x": "250", "lower_right_y": "959", "words": "(50PCS)"}, {"upper_left_x": "55", "upper_left_y": "973", "lower_right_x": "106", "lower_right_y": "1003", "words": "5.9"}, {"upper_left_x": "128", "upper_left_y": "970", "lower_right_x": "216", "lower_right_y": "1009", "words": "TYPE:"}, {"upper_left_x": "244", "upper_left_y": "974", "lower_right_x": "267", "lower_right_y": "1003", "words": "2"}, {"upper_left_x": "338", "upper_left_y": "972", "lower_right_x": "415", "lower_right_y": "1005", "words": "TOTAL"}, {"upper_left_x": "668", "upper_left_y": "970", "lower_right_x": "754", "lower_right_y": "1003", "words": "10.40"}, {"upper_left_x": "48", "upper_left_y": "1035", "lower_right_x": "191", "lower_right_y": "1067", "words": "DISCOUNT:"}, {"upper_left_x": "679", "upper_left_y": "1036", "lower_right_x": "749", "lower_right_y": "1066", "words": "0.00"}, {"upper_left_x": "53", "upper_left_y": "1092", "lower_right_x": "258", "lower_right_y": "1133", "words": "ROUNDING ADJ"}, {"upper_left_x": "680", "upper_left_y": "1096", "lower_right_x": "748", "lower_right_y": "1126", "words": "0.00"}, {"upper_left_x": "51", "upper_left_y": "1150", "lower_right_x": "358", "lower_right_y": "1187", "words": "TOTAL INCLUSIVE GST:"}, {"upper_left_x": "645", "upper_left_y": "1152", "lower_right_x": "753", "lower_right_y": "1189", "words": "10.40"}, {"upper_left_x": "52", "upper_left_y": "1213", "lower_right_x": "149", "lower_right_y": "1249", "words": "CASH"}, {"upper_left_x": "645", "upper_left_y": "1211", "lower_right_x": "751", "lower_right_y": "1250", "words": "10.40"}, {"upper_left_x": "48", "upper_left_y": "1276", "lower_right_x": "332", "lower_right_y": "1328", "words": "GST SUMMARY"}, {"upper_left_x": "53", "upper_left_y": "1338", "lower_right_x": "132", "lower_right_y": "1374", "words": "CODE"}, {"upper_left_x": "168", "upper_left_y": "1339", "lower_right_x": "203", "lower_right_y": "1372", "words": "%"}, {"upper_left_x": "292", "upper_left_y": "1341", "lower_right_x": "420", "lower_right_y": "1373", "words": "NET AMT"}, {"upper_left_x": "515", "upper_left_y": "1341", "lower_right_x": "588", "lower_right_y": "1375", "words": "GST"}, {"upper_left_x": "641", "upper_left_y": "1341", "lower_right_x": "791", "lower_right_y": "1377", "words": "TOTAL(RM)"}, {"upper_left_x": "56", "upper_left_y": "1395", "lower_right_x": "106", "lower_right_y": "1430", "words": "SR"}, {"upper_left_x": "161", "upper_left_y": "1398", "lower_right_x": "213", "lower_right_y": "1433", "words": "SR"}, {"upper_left_x": "345", "upper_left_y": "1398", "lower_right_x": "411", "lower_right_y": "1434", "words": "9.81"}, {"upper_left_x": "518", "upper_left_y": "1397", "lower_right_x": "586", "lower_right_y": "1432", "words": "0.59"}, {"upper_left_x": "706", "upper_left_y": "1397", "lower_right_x": "789", "lower_right_y": "1434", "words": "10.40"}, {"upper_left_x": "54", "upper_left_y": "1450", "lower_right_x": "134", "lower_right_y": "1487", "words": "TOTAL"}, {"upper_left_x": "342", "upper_left_y": "1454", "lower_right_x": "411", "lower_right_y": "1484", "words": "9.81"}, {"upper_left_x": "518", "upper_left_y": "1453", "lower_right_x": "588", "lower_right_y": "1487", "words": "0.59"}, {"upper_left_x": "706", "upper_left_y": "1454", "lower_right_x": "788", "lower_right_y": "1486", "words": "10.40"}, {"upper_left_x": "50", "upper_left_y": "1534", "lower_right_x": "782", "lower_right_y": "1585", "words": "THANK YOU ! & PLEASE COME AGAIN !!"}, {"upper_left_x": "60", "upper_left_y": "1589", "lower_right_x": "772", "lower_right_y": "1623", "words": "GOODS SOLD ARE NOT RETURNABLE FOR REFUND OR EXCHANGE !!"}]}
```
##Multiple lines (Just for understanding - not to use as input)
```
{
   "img_height":1678,
   "img_width":884,
   "tokens":[
      {
         "upper_left_x":"46",
         "upper_left_y":"276",
         "lower_right_x":"780",
         "lower_right_y":"317",
         "words":"HON HWA HARDWARE TRADING"
      },
      {
         "upper_left_x":"122",
         "upper_left_y":"325",
         "lower_right_x":"699",
         "lower_right_y":"366",
         "words":"COMPANY REG. NO. : 001055194X"
      },
      {
         "upper_left_x":"45",
         "upper_left_y":"368",
         "lower_right_x":"762",
         "lower_right_y":"410",
         "words":"NO 37  JALAN MANIS 7  TAMAN SEGAR "
      },
      {
         "upper_left_x":"106",
         "upper_left_y":"412",
         "lower_right_x":"709",
         "lower_right_y":"454",
         "words":"56100 CHERAS  KUALA LUMPUR."
      },
      {
         "upper_left_x":"268",
         "upper_left_y":"457",
         "lower_right_x":"551",
         "lower_right_y":"491",
         "words":"+603-9130 2672"
      },
      {
         "upper_left_x":"208",
         "upper_left_y":"503",
         "lower_right_x":"601",
         "lower_right_y":"540",
         "words":"GST REG : 001125220352"
      },
      {
         "upper_left_x":"283",
         "upper_left_y":"562",
         "lower_right_x":"545",
         "lower_right_y":"604",
         "words":"TAX INVOICE"
      },
      {
         "upper_left_x":"50",
         "upper_left_y":"640",
         "lower_right_x":"249",
         "lower_right_y":"678",
         "words":"CB# : 87870"
      },
      {
         "upper_left_x":"418",
         "upper_left_y":"646",
         "lower_right_x":"792",
         "lower_right_y":"681",
         "words":"21/09/2017 10:20:37 AM"
      },
      {
         "upper_left_x":"50",
         "upper_left_y":"680",
         "lower_right_x":"250",
         "lower_right_y":"717",
         "words":"M# : C2 - 0"
      },
      {
         "upper_left_x":"51",
         "upper_left_y":"724",
         "lower_right_x":"172",
         "lower_right_y":"760",
         "words":"CASHIER"
      },
      {
         "upper_left_x":"203",
         "upper_left_y":"729",
         "lower_right_x":"318",
         "lower_right_y":"762",
         "words":"CASH1 -"
      },
      {
         "upper_left_x":"50",
         "upper_left_y":"785",
         "lower_right_x":"107",
         "lower_right_y":"822",
         "words":"QTY"
      },
      {
         "upper_left_x":"131",
         "upper_left_y":"782",
         "lower_right_x":"300",
         "lower_right_y":"824",
         "words":"DESCRIPTION"
      },
      {
         "upper_left_x":"546",
         "upper_left_y":"782",
         "lower_right_x":"628",
         "lower_right_y":"816",
         "words":"PRICE"
      },
      {
         "upper_left_x":"661",
         "upper_left_y":"783",
         "lower_right_x":"810",
         "lower_right_y":"821",
         "words":"TOTAL(RM)"
      },
      {
         "upper_left_x":"57",
         "upper_left_y":"831",
         "lower_right_x":"462",
         "lower_right_y":"868",
         "words":"0.9 3/4\" ALUMINIUM ROD"
      },
      {
         "upper_left_x":"558",
         "upper_left_y":"832",
         "lower_right_x":"631",
         "lower_right_y":"864",
         "words":"6.00"
      },
      {
         "upper_left_x":"680",
         "upper_left_y":"835",
         "lower_right_x":"813",
         "lower_right_y":"867",
         "words":"5.40 SR"
      },
      {
         "upper_left_x":"86",
         "upper_left_y":"880",
         "lower_right_x":"390",
         "lower_right_y":"915",
         "words":"5 PVC WALLPLUG"
      },
      {
         "upper_left_x":"562",
         "upper_left_y":"884",
         "lower_right_x":"627",
         "lower_right_y":"913",
         "words":"1.00"
      },
      {
         "upper_left_x":"684",
         "upper_left_y":"880",
         "lower_right_x":"814",
         "lower_right_y":"918",
         "words":"5.00 SR"
      },
      {
         "upper_left_x":"119",
         "upper_left_y":"921",
         "lower_right_x":"250",
         "lower_right_y":"959",
         "words":"(50PCS)"
      },
      {
         "upper_left_x":"55",
         "upper_left_y":"973",
         "lower_right_x":"106",
         "lower_right_y":"1003",
         "words":"5.9"
      },
      {
         "upper_left_x":"128",
         "upper_left_y":"970",
         "lower_right_x":"216",
         "lower_right_y":"1009",
         "words":"TYPE:"
      },
      {
         "upper_left_x":"244",
         "upper_left_y":"974",
         "lower_right_x":"267",
         "lower_right_y":"1003",
         "words":"2"
      },
      {
         "upper_left_x":"338",
         "upper_left_y":"972",
         "lower_right_x":"415",
         "lower_right_y":"1005",
         "words":"TOTAL"
      },
      {
         "upper_left_x":"668",
         "upper_left_y":"970",
         "lower_right_x":"754",
         "lower_right_y":"1003",
         "words":"10.40"
      },
      {
         "upper_left_x":"48",
         "upper_left_y":"1035",
         "lower_right_x":"191",
         "lower_right_y":"1067",
         "words":"DISCOUNT:"
      },
      {
         "upper_left_x":"679",
         "upper_left_y":"1036",
         "lower_right_x":"749",
         "lower_right_y":"1066",
         "words":"0.00"
      },
      {
         "upper_left_x":"53",
         "upper_left_y":"1092",
         "lower_right_x":"258",
         "lower_right_y":"1133",
         "words":"ROUNDING ADJ"
      },
      {
         "upper_left_x":"680",
         "upper_left_y":"1096",
         "lower_right_x":"748",
         "lower_right_y":"1126",
         "words":"0.00"
      },
      {
         "upper_left_x":"51",
         "upper_left_y":"1150",
         "lower_right_x":"358",
         "lower_right_y":"1187",
         "words":"TOTAL INCLUSIVE GST:"
      },
      {
         "upper_left_x":"645",
         "upper_left_y":"1152",
         "lower_right_x":"753",
         "lower_right_y":"1189",
         "words":"10.40"
      },
      {
         "upper_left_x":"52",
         "upper_left_y":"1213",
         "lower_right_x":"149",
         "lower_right_y":"1249",
         "words":"CASH"
      },
      {
         "upper_left_x":"645",
         "upper_left_y":"1211",
         "lower_right_x":"751",
         "lower_right_y":"1250",
         "words":"10.40"
      },
      {
         "upper_left_x":"48",
         "upper_left_y":"1276",
         "lower_right_x":"332",
         "lower_right_y":"1328",
         "words":"GST SUMMARY"
      },
      {
         "upper_left_x":"53",
         "upper_left_y":"1338",
         "lower_right_x":"132",
         "lower_right_y":"1374",
         "words":"CODE"
      },
      {
         "upper_left_x":"168",
         "upper_left_y":"1339",
         "lower_right_x":"203",
         "lower_right_y":"1372",
         "words":"%"
      },
      {
         "upper_left_x":"292",
         "upper_left_y":"1341",
         "lower_right_x":"420",
         "lower_right_y":"1373",
         "words":"NET AMT"
      },
      {
         "upper_left_x":"515",
         "upper_left_y":"1341",
         "lower_right_x":"588",
         "lower_right_y":"1375",
         "words":"GST"
      },
      {
         "upper_left_x":"641",
         "upper_left_y":"1341",
         "lower_right_x":"791",
         "lower_right_y":"1377",
         "words":"TOTAL(RM)"
      },
      {
         "upper_left_x":"56",
         "upper_left_y":"1395",
         "lower_right_x":"106",
         "lower_right_y":"1430",
         "words":"SR"
      },
      {
         "upper_left_x":"161",
         "upper_left_y":"1398",
         "lower_right_x":"213",
         "lower_right_y":"1433",
         "words":"SR"
      },
      {
         "upper_left_x":"345",
         "upper_left_y":"1398",
         "lower_right_x":"411",
         "lower_right_y":"1434",
         "words":"9.81"
      },
      {
         "upper_left_x":"518",
         "upper_left_y":"1397",
         "lower_right_x":"586",
         "lower_right_y":"1432",
         "words":"0.59"
      },
      {
         "upper_left_x":"706",
         "upper_left_y":"1397",
         "lower_right_x":"789",
         "lower_right_y":"1434",
         "words":"10.40"
      },
      {
         "upper_left_x":"54",
         "upper_left_y":"1450",
         "lower_right_x":"134",
         "lower_right_y":"1487",
         "words":"TOTAL"
      },
      {
         "upper_left_x":"342",
         "upper_left_y":"1454",
         "lower_right_x":"411",
         "lower_right_y":"1484",
         "words":"9.81"
      },
      {
         "upper_left_x":"518",
         "upper_left_y":"1453",
         "lower_right_x":"588",
         "lower_right_y":"1487",
         "words":"0.59"
      },
      {
         "upper_left_x":"706",
         "upper_left_y":"1454",
         "lower_right_x":"788",
         "lower_right_y":"1486",
         "words":"10.40"
      },
      {
         "upper_left_x":"50",
         "upper_left_y":"1534",
         "lower_right_x":"782",
         "lower_right_y":"1585",
         "words":"THANK YOU ! & PLEASE COME AGAIN !!"
      },
      {
         "upper_left_x":"60",
         "upper_left_y":"1589",
         "lower_right_x":"772",
         "lower_right_y":"1623",
         "words":"GOODS SOLD ARE NOT RETURNABLE FOR REFUND OR EXCHANGE !!"
      }
   ]
}
```