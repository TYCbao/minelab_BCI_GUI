# minelab_BCI_GUI
此程式用途為介面化繪製中央大學Minelab製作之19通道大腦拓樸圖以及分析想像運動之ERD/ERS
## main_GUI.py
主要執行程式。  
第一頁請輸入受試者編號並選擇功能，主要分析分別有Rest State、N-Back以及Motor Imagery  
選擇需分析之功能後，點選NEXT按鈕，進入第二頁選擇預分析資料檔案(資料夾)  
- Rest State : 選擇名為1.txt之EEG Raw Data檔案 
- N-Back : 選擇名為1.txt之EEG Raw Data檔案 
- Motor Imagery : 選擇含有為1.txt及tri_1.txt之資料夾(資料夾名稱預設為受試時間)
![image](https://github.com/user-attachments/assets/baf2905f-01b0-45bd-8e3a-346d9cba7b38)

## 結果
- Rest State與N-Back結果下方有FFT與Band Power按鈕，點選後會分別繪製EEG之FFT圖及Band Power分析。
- Motor Imagery 下方ERD/ERS按鈕，點選後繪製此次想像運動實驗之ERD/ERS圖
