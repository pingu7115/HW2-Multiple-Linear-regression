# 巨量資料分析與應用 - 作業二 (HW2) 完整對話紀錄

本文件記錄了使用者與 AI 助手在完成 **作業二：多元線性迴歸分析 (Multiple Linear Regression, CRISP-DM)** 過程中的所有對話、需求討論、版本疊代與除錯歷程。

- **課程名稱**：巨量資料分析與應用
- **作業主題**：HW2: Multiple Linear Regression (多元線性迴歸)
- **選用資料集**：Kaggle - House Sales in King County, USA (18 個特徵)
- **產出主要檔案**：
  - `HW2_Multiple_Linear_Regression.py`（Python 執行腳本）
  - `HW2_Multiple_Linear_Regression.ipynb`（Jupyter Notebook）
  - `HW2_Report.md`（Markdown 報告）
  - `HW2_Report.html`（含 MathJax 渲染與一鍵列印 PDF 的 HTML 報告）
  - `figures/`（8 張高解析分析圖表）

---

## 📑 對話目錄
1. [回合一：作業需求解析與初版建置 (醫療保險資料集)](#回合一作業需求解析與初版建置)
2. [回合二：報告開啟與 PDF 另存操作指引](#回合二報告開啟與-pdf-另存操作指引)
3. [回合三：新版規格檢核與全面重構 (切換為 King County 房價 18 特徵與 CRISP-DM 流程)](#回合三新版規格檢核與全面重構)
4. [回合四：HTML 數學公式與表格排版異常排查與修復](#回合四html-數學公式與表格排版異常排查與修復)
5. [回合五：對話歷史轉存為 Markdown 紀錄](#回合五對話歷史轉存為-markdown-紀錄)
6. [回合六：升級為現代化 Web Dashboard 網頁風格](#回合六升級為現代化-web-dashboard-網頁風格)

---

## 回合一：作業需求解析與初版建置

### 👤 使用者 (User)
> **訊息內容**：  
> 幫我做作業  
>  
> *(附圖：LMS 系統之作業二公告截圖，題目為「HW2: Multiple Linear regression」，說明多元線性迴歸評分指標，推薦 Kaggle 4~15 特徵之資料集清單，其中老師 Top 3 教學推薦首選為 Medical Cost Personal Datasets 醫療費用預測)*

### 🤖 AI 助手 (Assistant) - 互動選項確認
AI 助手分析截圖後，透過選項引導使用者確認資料集與輸出格式偏好：
- **問題 1**：作業清單中推薦了數個 Kaggle 資料集，請問您偏好使用哪一個資料集進行多元線性迴歸分析？  
  👉 **使用者選擇**：`(Recommended) Medical Cost Personal Datasets（醫療費用預測，老師推薦首選，含年齡、BMI、抽菸等 6 個特徵，迴歸解釋性極高）`
- **問題 2**：您希望產出哪些繳交檔案格式？  
  👉 **使用者選擇**：`(Recommended) 獨立 Python 腳本 (.py)（可直接終端執行並生成所有圖表）`

### 🤖 AI 助手 (Assistant) - 成果交付
作業二（**HW2: Multiple Linear Regression 多元線性迴歸分析**）已全數製作完成！

本作業依據老師在課堂說明的評分標準與推薦清單，選用 Kaggle 最經典且具備極高解釋性的 **醫療保險費用預測資料集 (`insurance.csv`)**，特徵數為 6 個（完全符合 4~15 個 Features 的要求）。

---

### 📁 交付檔案清單（已存放於專案資料夾）

1. **[HW2_Multiple_Linear_Regression.py](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Multiple_Linear_Regression.py)**
   - 獨立且完整的 Python 執行腳本。包含資料載入、EDA、類別編碼、特徵縮放、五大模型訓練評估、高斯-馬可夫四大假設統計檢驗，並自動輸出 7 張 300 DPI 高畫質圖表至 `figures/` 資料夾。
2. **[HW2_Multiple_Linear_Regression.ipynb](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Multiple_Linear_Regression.ipynb)**
   - 完整 Jupyter Notebook，包含 Markdown 說明、分段程式碼與結構。
3. **[HW2_Report.md](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.md)**
   - 詳細的作業成果分析書面報告，嚴格對標教師五大評分維度（EDA、模型訓練、評估比較、圖表視覺化、係數商業詮釋）。
4. **[HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html)**
   - 美觀排版的 HTML 報告，內建「🖨️ 列印 / 另存為 PDF」按鈕，可在瀏覽器（Edge / Chrome）一鍵輸出為 PDF。
5. **[figures/](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/figures)**
   - 7 張高品質分析圖表（分佈圖、相關係數熱力圖、吸菸與 BMI 交互效應圖、各模型指標柱狀圖、殘差診斷四格圖、預測散佈圖、標準化係數圖）。
6. **[insurance.csv](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/insurance.csv)**
   - 資料集（1,338 筆樣本，包含年齡、性別、BMI、子女數、抽菸、地區及醫療費用）。

---

### 📊 核心實驗結果摘要

| 模型 (Model) | 訓練集 $R^2$ | 測試集 $R^2$ | 測試集 Adj $R^2$ | 測試集 RMSE ($) | 測試集 MAE ($) | 模型特性與評析 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **多元線性迴歸 (Baseline OLS)** | 0.7417 | 0.7836 | 0.7769 | $5,796.28 | $4,181.19 | 基準模型，表現穩健 |
| **Ridge 嶺迴歸 (L2)** | 0.7417 | 0.7835 | 0.7769 | $5,796.98 | $4,182.80 | 平滑權重，抑制共線性 |
| **Lasso 套索迴歸 (L1)** | 0.7417 | 0.7836 | 0.7769 | $5,796.65 | $4,181.51 | L1 稀疏性特徵篩選 |
| **ElasticNet 彈性網絡** | 0.7400 | 0.7793 | 0.7725 | $5,853.88 | $4,275.66 | L1 + L2 混合正則化 |
| **⭐ 交互項多元迴歸 (OLS + BMI\*Smoker)** | **0.8341** | **0.8653** | **0.8605** | **$4,573.81** | **$2,756.90** | **表現最優（RMSE 降低 21.1%，MAE 降低 34.1%）** |

---

### 🔍 關鍵統計發現與商業意涵（符合作業要求亮點）

1. **抽菸與肥胖的「超加乘爆炸性」交互效應 (`bmi_smoker`, $t = 24.29, p < 0.0001$)**：
   - 呼應老師在作業說明特別提示的重點：非抽菸者即使 BMI 增加，醫療費用上升幅度溫和；然而**一旦為吸菸者且 BMI > 30（肥胖），醫療費用呈斷崖式暴增**（年支出高達 $30,000 ~ $60,000 美元），加入交互特徵使整體 $R^2$ 躍升至 **0.8653**。
2. **年齡效應 (`age`, $t = 24.36, p < 0.0001$)**：
   - 在控制其他條件下，被保險人每增長 1 歲，平均醫療支出增加約 **$263.39 美元**，客觀反映衰老帶來的慢性病發病機率。
3. **高斯-馬可夫假設檢驗完全通過**：
   - **無多元共線性**：所有特徵之 VIF 均小於 1.62（遠低於警戒門檻 5）。
   - **誤差獨立性**：Durbin-Watson 統計量為 **2.0734**（極接近 2.0，確認無自相關）。
   - **同質變異**：Breusch-Pagan 檢定 $p = 0.3923 > 0.05$，未違反同質變異假設。
4. **保險核保策略建議**：
   - 建議保險公司推動「外溢型健康促進保單」，提供保戶戒菸與體重管理保費折扣，降低高額理賠風險。

---

### 🚀 如何執行與查看成果

- **在終端直接執行 Python 腳本**：
  ```bash
  python HW2_Multiple_Linear_Regression.py
  ```
- **查看與匯出報告**：
  雙擊打開 `HW2_Report.html`，點擊右下角「**列印 / 另存為 PDF**」，即可取得排版完成的作業書面報告 PDF。

---

## 回合二：報告開啟與 PDF 另存操作指引

### 👤 使用者 (User)
> **訊息內容**：  
> 我執行後，怎麼雙擊打開 HW2_Report.html?

### 🤖 AI 助手 (Assistant)
您可以透過以下幾種方式輕鬆打開 [HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html)：

---

### 方法一：從 Windows 桌面資料夾「雙擊打開」（最直覺）

1. 回到 Windows **桌面 (Desktop)**。
2. 找到並進入資料夾：`HW2 Multiple Linear regression`。
3. 在資料夾內找到 **`HW2_Report.html`**（圖示通常會是 Edge 或 Chrome 的網頁圖示）。
4. **對它連按兩下滑鼠左鍵（雙擊）**，系統就會自動以瀏覽器開啟完整的排版報告。

---

### 方法二：在編輯器左側檔案清單中開啟

1. 看向編輯器**左側的檔案總管（Explorer）側邊欄**。
2. 找到 `HW2_Report.html`。
3. **在檔名上按滑鼠右鍵**，選擇：
   - **「在檔案總管中顯示 (Reveal in File Explorer)」**：會直接幫您彈出 Windows 資料夾並選中該檔案，接著直接雙擊即可。
   - 或選擇 **「在預設瀏覽器中開啟 (Open with Default Browser)」**。

---

### 方法三：在終端機（Terminal）輸入一行指令直接叫出瀏覽器

如果您當前已經在終端機視窗中，直接輸入以下指令按 Enter，系統就會立刻幫您用瀏覽器彈出該報告：

```powershell
Start-Process HW2_Report.html
```

---

### 💡 如何另存為 PDF 繳交？

1. 網頁開啟後，報告內已嵌入所有統計數據表格與 7 張分析圖表。
2. 點擊頁面右下角的藍色浮動按鈕 **「🖨️ 列印 / 另存為 PDF」**（或按鍵盤快捷鍵 `Ctrl + P`）。
3. 印表機目的地選擇 **「另存為 PDF (Save as PDF)」**。
4. 點擊「儲存」，即可取得繳交格式的作業分析報告 PDF。

---

## 回合三：新版規格檢核與全面重構

### 👤 使用者 (User)
> **訊息內容**：  
> 檢查目前作業是否符合附圖這幾項。  
> 提供你選擇的kaggle網址給我，並檢查是否符合14-39 features。  
> 請將作業中的公式用數學式寫好，不要用程式碼的寫法。  
> 作業網址如下:  
> file:///C:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html  
>  
> *(附圖：作業規格補充說明截圖，重點包含：*  
> *1. 資料來源：Kaggle 10 至 20 個特徵之公開資料集，明確標示來源與連結。*  
> *2. 分析任務：使用線性迴歸，**必須執行特徵選擇 (Feature Selection) 與模型評估**，結果需包含**預測圖(加上信賴區間或預測區間)**。*  
> *3. CRISP-DM 流程說明：Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment。)*

### 🤖 AI 助手 (Assistant)
已為您逐項檢核作業要求，並全面升級為符合**最新規範**與**附圖所有細節**的全新版本！

---

### 一、作業規範逐項檢核對照表

| 附圖要求項目 | 原先狀態 | 現況調整與檢核結果 | 判定 |
| :--- | :--- | :--- | :---: |
| **1. 資料來源與特徵數**<br>（10 至 20 個特徵 / 14~39 features，註明 Kaggle 連結） | 原醫療資料集僅 6 個特徵，不符規範 | **已更換為 Kaggle 經典房價資料集**，扣除流水號與目標變數後**共計 18 個特徵**，完美落於 10~20 個特徵（且符合 14~39 features）區間內，並於報告首頁明確標示 Kaggle 官方連結。 | **通過 (100%)** |
| **2. 特徵選擇 (Feature Selection)**<br>（作業規定「必須執行」） | 原僅有正則化係數懲罰 | **實作 RFE（遞迴特徵消除法）與 Lasso 稀疏篩選**，精選出 12 個核心特徵，並產出「全特徵 vs 特徵選擇後精簡模型」之 $R^2$、RMSE、MAE 對比長條圖與指標表。 | **通過 (100%)** |
| **3. 預測圖含區間**<br>（需加上信賴區間或預測區間） | 原僅有一般預測散佈圖 | **新增圖 6：繪製專屬預測圖**，同時包含 **95% 期望值信賴區間 (Confidence Interval)**、**95% 個體值預測區間 (Prediction Interval)** 與測試集實際成交價散佈點。 | **通過 (100%)** |
| **4. CRISP-DM 流程說明**<br>（依循六大標準階段開展） | 原採傳統五大步驟 | **全面改寫為 CRISP-DM 六大階段**：Business Understanding $\rightarrow$ Data Understanding $\rightarrow$ Data Preparation $\rightarrow$ Modeling $\rightarrow$ Evaluation $\rightarrow$ Deployment。 | **通過 (100%)** |
| **5. 數學公式表示法**<br>（不要用程式碼寫法） | 原含有部分代碼化表示 | **全面改以學術標準 $\LaTeX$ 數學式呈現**，並在 HTML 報告中導入 **MathJax 3 渲染引擎**，所有公式均以高質感數學方程式排版展現。 | **通過 (100%)** |

---

### 二、所選 Kaggle 資料集與特徵數量檢核

- **資料集名稱**：House Sales in King County, USA (`kc_house_data.csv`)
- **Kaggle 官方來源網址**：[https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **特徵數量驗證**：
  - 資料集共有 21 個欄位，排除識別碼 `id`、日期 `date` 以及預測目標 `price`。
  - **實際可用預測特徵共 18 個**：
    `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`
  - **檢核結果**：**18 個特徵** 完全符合附圖要求之「**10 至 20 個特徵 (features)**」，亦完全滿足您的「**14-39 features**」要求。

---

### 三、報告中主要公式之標準數學式呈現

在 [HW2_Report.md](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.md) 與 [HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html) 中，所有公式均已使用標準數學式改寫：

#### 1. 多元線性迴歸母體模型
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i, \quad \varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$

#### 2. 最小平方法 (OLS) 矩陣解
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

#### 3. 評估指標 (RMSE, MAE, $R^2$, 調整後 $R^2$)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}, \quad R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$

#### 4. ⭐ 95% 信賴區間 (Confidence Interval for Mean Response $E(y_0)$)
$$\hat{y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s_e \sqrt{\mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$$

#### 5. ⭐ 95% 預測區間 (Prediction Interval for Individual Observation $y_0$)
$$\hat{y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s_e \sqrt{1 + \mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$$

#### 6. 正則化損失函數 (Ridge, Lasso)
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

$$\hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

---

### 四、最新產出檔案與檢視連結

1. **HTML 完整網頁報告（含 MathJax 數學方程式與列印 PDF 功能）**：
   - 檔案路徑：[HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html)
   - 可在終端機輸入 `Start-Process HW2_Report.html`，或直接雙擊檔案，瀏覽器即可載入數學公式排版與 8 張高畫質圖表。
2. **完整 Python 執行腳本**：
   - 檔案路徑：[HW2_Multiple_Linear_Regression.py](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Multiple_Linear_Regression.py)
   - 執行指令：`python HW2_Multiple_Linear_Regression.py`（已實測執行完畢，代碼退出碼為 0）。
3. **Jupyter Notebook 檔案**：
   - 檔案路徑：[HW2_Multiple_Linear_Regression.ipynb](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Multiple_Linear_Regression.ipynb)
4. **Markdown 格式報告**：
   - 檔案路徑：[HW2_Report.md](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.md)
5. **圖表目錄**：
   - [figures/](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/figures) 包含 8 張高解析（300 DPI）圖表，其中 **`fig6_prediction_intervals.png`** 即為符合附圖要求的「信賴區間與預測區間預測圖」。

---

## 回合四：HTML 數學公式與表格排版異常排查與修復

### 👤 使用者 (User)
> **訊息內容**：  
> 修正後續的數學公式(從附圖開始)  
>  
> *(附圖：使用者以瀏覽器開啟 `HW2_Report.html` 之畫面截圖，顯示在「三、Phase 3: 資料準備與特徵工程」3.2 節中，特徵標準化公式未被渲染，呈現為原始字串 `$$z_{ij} = \frac{x_{ij} - \bar{x}j}{s_j}...$$`)*

### 🤖 AI 助手 (Assistant)
已全面修正自 **Phase 3（附圖處）起至報告結尾** 的所有數學公式與表格排版問題！

---

### 🔍 問題原因診斷與修正說明

1. **下標與加總符號解析衝突修復**：
   - 原先 Markdown 轉換器將公式中的下標 `_j`、`_i` 及加總 `\sum_{i=1}^n` 誤判為斜體標籤（變成 `<em>j</em>`），破壞了 LaTeX 語法結構，導致瀏覽器中的 MathJax 無法正常辨認而顯示成原始文本。
   - **修正**：已建立公式保護機制，確保所有 `$$...$$` 與 `$...$` 在轉換過程中保持純淨的 $\LaTeX$ 原貌，交由 **MathJax 3** 進行標準方程式排版渲染。
2. **5.6 節推論統計表格分隔線修復**：
   - 原先表格標頭中的 `$P > |t|$` 其絕對值符號 `|` 被 Markdown 誤判為表格欄位分隔線（Pipe），導致整張表格破版變為純文字。
   - **修正**：已將絕對值符號轉義為 `\|t\|`，表格已完全恢復為對齊的標準 HTML 表格。

---

### 📐 修正後的重點數學公式對照清單（Phase 3 起）

#### 1. Phase 3.2：特徵標準化公式 (Z-score Standardization)
$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}, \quad \text{其中 } \bar{x}_j = \frac{1}{n} \sum_{i=1}^n x_{ij}, \quad s_j = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2}$$

#### 2. Phase 4.1：模型目標與正則化損失函數
- **全特徵多元線性迴歸 (Baseline OLS)**：
  $$\min_{\boldsymbol{\beta}} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2$$
- **嶺迴歸 (Ridge, L2 正則化)**：
  $$\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\} \implies \hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
- **套索迴歸 (Lasso, L1 稀疏特徵選擇)**：
  $$\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$
- **彈性網絡迴歸 (ElasticNet)**：
  $$\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \left( \alpha \sum_{j=1}^p |\beta_j| + \frac{1-\alpha}{2} \sum_{j=1}^p \beta_j^2 \right) \right\}$$

#### 3. Phase 5.1：模型評估指標
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}, \quad R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$

#### 4. Phase 5.4：預測圖之區間推導（95% 信賴區間與預測區間）
- **點預測值**：
  $$\hat{y}_0 = \mathbf{x}_0^T \hat{\boldsymbol{\beta}}$$
- **95% 期望值信賴區間 (Confidence Interval for Mean Response $E(y_0)$)**：
  $$\text{CI}_{1-\alpha} = \hat{y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s_e \sqrt{\mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$$
- **95% 個體值預測區間 (Prediction Interval for Individual Observation $y_0$)**：
  $$\text{PI}_{1-\alpha} = \hat{y}_0 \pm t_{\alpha/2, \, n-p-1} \cdot s_e \sqrt{1 + \mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$$

#### 5. Phase 5.5：假設檢定指標
- **Durbin-Watson 獨立性檢定統計量**：
  $$d = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2} = 1.9939$$
- **變異數膨脹因子 (VIF)**：
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

---

### 🚀 立即檢視更新成果

[HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html) 已同步重新編譯完畢。

您只需在原本開啟的瀏覽器分頁中按 **`F5`（重新整理網頁）**，或在終端機執行：
```powershell
Start-Process HW2_Report.html
```
所有公式即會以數學符號排版呈現，表格與圖表亦完整就位。

---

## 回合五：對話歷史轉存為 Markdown 紀錄

### 👤 使用者 (User)
> **訊息內容**：  
> 幫我把和你的所有對話轉成.md

### 🤖 AI 助手 (Assistant)
> 已將我們所有的對話內容、需求疊代歷程、技術細節與問題修復過程完整整理並匯出至本檔案 `chat_log.md`。

---

## 回合六：升級為現代化 Web Dashboard 網頁風格

### 👤 使用者 (User)
> **訊息內容**：  
> 我希望html的分析內容不變，但改成[https://pingu7115.github.io/AIoT-DIC2-MultipleLR/]比較像網頁的風格

### 🤖 AI 助手 (Assistant)
> 已完整將 `HW2_Report.html` 升級重構為與該網頁一致的現代化 Web 儀表板架構：
> 1. **全域設計系統與字型**：引入 Google Fonts (`Outfit` 標題、`Inter` 內文、`Fira Code` 代碼) 與 FontAwesome 6 圖標庫。
> 2. **頂部導航欄 (Sticky Header)**：包含麵包屑導航、專案徽章、Kaggle 外部連結、Python 腳本與筆記本下載按鈕，以及「另存 PDF」快捷按鈕。
> 3. **主視覺橫幅 (Hero Card)**：深靛藍漸層背景搭配徑向光暈、核心摘要與 5 大亮點標籤 Pills。
> 4. **核心指標卡片群 (Metrics Grid)**：展示全特徵 $R^2$ (0.7012)、RFE 精選 $R^2$ (0.6998)、預測區間覆蓋率 (95.2%)、共線性 (VIF < 7.3) 與 DW 獨立性 (1.9939)。
> 5. **CRISP-DM 階段導航列 (Sticky Phase Nav)**：六大階段膠囊按鈕，點擊可平滑滾動至對應階段卡片。
> 6. **模組化階段卡片 (Phase Cards)**：六大階段獨立卡片化，內部 100% 保留原有文字分析、LaTeX 數學方程式、統計數據表格與 8 張分析圖表。

---

## 回合七：依據作業詳細規範進行全方位網頁升級與指標檢核

### 👤 使用者 (User)
> **訊息內容**：  
> "分析任務
>   使用線性回歸 (Linear Regression) 模型進行預測。
>   可嘗試單純線性回歸、多元線性回歸或 Auto Regression。
>   必須執行 特徵選擇 (Feature Selection) 與 模型評估 (Model Evaluation)。
>   結果部分需包含請提供預測圖(加上信賴區間或預測區間)
> 3. CRISP-DM 流程說明
>     Business Understanding
>     Data Understanding
>     Data Preparation
>     Modeling
>     Evaluation
>     Deployment"
> 請根據以上要求將網頁進行修改

### 🤖 AI 助手 (Assistant)
> 已針對引述之全部規範要求完成完整升級，具體落實成果如下：
> 1. **頂部新增「作業規範與分析任務檢核表 (Compliance Matrix)」專屬卡片**：
>    - 條列「分析任務 1 至 4」與「CRISP-DM 六大流程說明」之對應檢核點與完成度（全部標註 100% 達成）。
> 2. **落實「單純線性迴歸」與「多元線性迴歸」之雙模型實證比較**：
>    - **單純線性迴歸 (Simple Linear Regression)**：以相關係數最高的面積特徵 $\text{sqft\_living}$ 建立基準模型，實證方程式為 $\hat{y} = -41,999.19 + 279.55 \times \text{sqft\_living}$ ($R^2 = 0.4941$)。
>    - **多元線性迴歸 (Multiple Linear Regression)**：納入 17 個預測變數，$R^2$ 躍升至 $0.7012$（較單純迴歸躍升 +41.9%）。
> 3. **必須執行的「特徵選擇 (Feature Selection)」**：
>    - 採用遞迴特徵消除法 (RFE) 精選出 12 個核心特徵，特徵削減 29.4%，$R^2$ 保持在 0.6998（近乎零損耗）。
>    - 新增圖 4「特徵選擇與模型演進效能對比圖」，直觀呈現單純 (1 特徵) vs 多元 (17 特徵) vs RFE (12 特徵) 的演進價值。
> 4. **必須執行的「模型評估 (Model Evaluation)」**：
>    - 建立包含單純線性、多元全特徵、RFE 特徵選擇、Ridge、Lasso、ElasticNet 六大模型的完整實測指標評估總表 ($R^2$、Adj $R^2$、RMSE、MAE)。
>    - 執行高斯-馬可夫四大殘差診斷：Durbin-Watson 獨立性檢定 ($d=1.9939$)、VIF 多元共線性檢定（均小於 7.3）、常態 Q-Q 與同質變異檢驗。
> 5. **結果成果「預測圖（包含 95% 信賴區間與 95% 預測區間）」**：
>    - 嚴謹呈現期望值信賴區間 (CI) 與個別觀測值預測區間 (PI) 之矩陣數學推導公式。
>    - 完整呈現圖 6（多元線性迴歸測試集排序預測圖，覆蓋率 95.2%）與圖 0（單純線性迴歸散佈點、擬合線、信賴帶與預測帶）。
> 6. **3. CRISP-DM 流程說明架構**：
>    - 完整以 Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation、Deployment 六大獨立章節與導航膠囊組織全文。
> 7. **程式碼與報告同步更新**：
>    - `HW2_Report.html`、`HW2_Report.md`、`7115064016_hw2.py`、`HW2_Multiple_Linear_Regression.py` 全數同步更新並驗證執行無誤。

---

## 回合八：聚焦極簡現代風格與精準對焦作業指定題目

### 👤 使用者 (User)
> **訊息內容**：  
> 風格換一下
> 只要有寫出以下這些就好
> "分析任務
>   使用線性回歸 (Linear Regression) 模型進行預測。
>   可嘗試單純線性回歸、多元線性回歸或 Auto Regression。
>   必須執行 特徵選擇 (Feature Selection) 與 模型評估 (Model Evaluation)。
>   結果部分需包含請提供預測圖(加上信賴區間或預測區間)
> CRISP-DM 流程說明
>     Business Understanding
>     Data Understanding
>     Data Preparation
>     Modeling
>     Evaluation
>     Deployment"

### 🤖 AI 助手 (Assistant)
> 已將網頁全面重構為乾淨俐落、高可讀性、專業學術質感的清爽現代風格，並嚴格以使用者指定之兩大架構為核心大標題：
> 1. **大標題一：壹、分析任務 (Analysis Tasks)**
>    - 1. 使用線性回歸 (Linear Regression) 模型進行預測（單純線性回歸 vs 多元線性回歸完整對照）
>    - 2. 必須執行 特徵選擇 (Feature Selection)（RFE 12 特徵精選效益）
>    - 3. 必須執行 模型評估 (Model Evaluation)（六大模型指標對比、高斯-馬可夫四大殘差診斷）
>    - 4. 結果部分需包含提供預測圖 (加上信賴區間或預測區間)（95% CI 與 95% PI 數學推導與成果圖）
> 2. **大標題二：貳、CRISP-DM 流程說明 (CRISP-DM Methodology)**
>    - 1. Business Understanding (商業理解)
>    - 2. Data Understanding (資料理解與 18 個特徵維度檢核、EDA 視覺化)
>    - 3. Data Preparation (完全共線性清洗與特徵標準化)
>    - 4. Modeling (模型建立)
>    - 5. Evaluation (模型評估)
>    - 6. Deployment (模型部署與 API 架構)
> 3. **全域技術細節**：
>    - MathJax 3 標準 LaTeX 數學式完整排版，公式無任何轉義瑕疵。
>    - 清新白/灰極簡色系搭配科技藍點綴，去除冗餘裝飾，閱讀焦點直接對齊教授與助教之評分規範。

---

## 回合九：移除行內數學公式背景框與邊框

### 👤 使用者 (User)
> **訊息內容**：  
> 幫我把類似這些的背景框都拿掉（附圖顯示行內數學式 $R^2 = 0.7012$、$RMSE = \$212,539.52$、$d = 1.9939$ 等帶有灰色圓角背景框與邊框）。

### 🤖 AI 助手 (Assistant)
> **問題原因**：  
> 先前 CSS 中對 `mjx-container` 設定了全域的 `background: #fafbfc`、`border: 1px solid var(--slate-200)` 以及內外邊距，導致 MathJax 渲染行內變數與公式時均被加上灰色膠囊外框。
> 
> **修正成果**：  
> 1. 將 `mjx-container` 的背景設定為透明 (`background: transparent !important`)，徹底移除邊框 (`border: none !important`) 與外框陰影，並重設邊距為 0。
> 2. 獨立定義塊級公式 `mjx-container[display="true"]`，僅保留適當的上下段落間距與橫向溢出滾動，無任何多餘背景色塊。
> 3. 所有行內數學符號與數值（如 $R^2$、$RMSE$、$d$、$\hat{\boldsymbol{\beta}}$）已完全自然融入內文段落，排版視覺乾淨整潔。

---

## 回合十：實作作業加分項規範（總計加 3 分）

### 👤 使用者 (User)
> **訊息內容**：  
> 幫我做加分項規範如下  
> ========加分 ==============  
> 加總分 1分:  
> 1. 有下拉視窗呈現兩種problem  
> 加總分2分  
> 1. 可任意輸入 Dataset  

### 🤖 AI 助手 (Assistant)
> 已全面實作並整合完成兩大加分功能：
> 
> 1. **【加總分 1分】下拉視窗呈現兩種 Problem**：
>    - 於頁面頂部 Sticky Navbar 以及上方加分資訊列中，建置了醒目的 Problem 切換下拉選單 (`<select id="problemSelect">`)。
>    - **Problem 1**：King County 房價預測 (`kc_house_data.csv`，18 特徵，21,613 筆實價登錄房屋資料，Target: `price`)。
>    - **Problem 2**：醫療保險個人支出預測 (`insurance.csv`，6 特徵，1,338 筆樣本，Target: `charges`)。
>      - 完整呈現 壹、分析任務（單純迴歸 vs 多元迴歸 $R^2 = 0.7836$、$RMSE = \$5,796.28$、RFE 精選 4 核心特徵、95% PI 實測圖表 `figures/fig_insurance_prediction.png` 與 EDA 分析圖 `figures/fig_insurance_eda.png`）。
>      - 完整呈現 貳、CRISP-DM 六大標準流程（Business, Data, Prep, Modeling, Eval, Deploy）。
> 
> 2. **【加總分 2分】可任意輸入 Dataset 即時多元線性迴歸引擎**：
>    - 於報告前段整合了極致流暢的客戶端多元線性迴歸互動卡片 (`#custom-dataset-section`)。
>    - **多元資料輸入模式**：
>      - 模式一：輸入公開 CSV 網址（含一鍵載入 50_Startups、Boston Housing、Medical Insurance 快捷按鈕）。
>      - 模式二：本機檔案拖曳上傳（支援任意 `.csv` 檔案，透過 HTML5 原生 FileReader 於本機即時解析，零延遲且保護隱私）。
>      - 模式三：直接貼上包含標題列之 CSV 純文字。
>    - **即時瀏覽器端 OLS 機器學習運算 (`ml_engine.js`)**：
>      - 自動識別數值與類別特徵，自動進行 One-Hot Encoding 與 80/20 訓練/測試切分。
>      - 矩陣運算解析解求解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$。
>      - 即時輸出評估指標（$R^2$、Adjusted $R^2$、RMSE、MAE）、擬合方程式與特徵權重排序。
>      - **即時預測模擬器 (Live Predictor)**：動態數值滑桿，隨使用者拖曳即時給出 $\hat{y}_0$、95% 信賴區間與 95% 預測區間。

---

## 回合十一：實作專屬下拉式清單版本（明確區分 King County 主要作業 與 加分題）

### 👤 使用者 (User)
> **訊息內容**：  
> 可以幫我做出下拉式清單的版本嗎?區分King County 和加分題  

### 🤖 AI 助手 (Assistant)
> 已成功重構網頁架構為「雙向連動之階層式下拉選單與視圖切換器」版本：
> 
> 1. **核心架構設計（預設獨立專注於 King County 主作業）**：
>    - **預設檢視**：開啟網頁時**僅顯示 King County 房價預測主作業**，加分題與自訂 Dataset 卡片預設收折隱藏，保證版面專注、乾淨俐落，完全符合主要評分焦點。
>    - **頂端與頁首雙下拉選單 (`<select>`)**：包含分組 `<optgroup>`，隨時平滑切換各視圖：
>      - 📋 **主要作業 (Core HW2)**：`🏡 King County 房價預測 (主要作業 - 18 特徵與 CRISP-DM 報告)`
>      - 🌟 **加分題專區 (Bonus +3分)**：
>        - `🌟 加分題專區 (包含加分 1 分與加分 2 分完整功能)`
>        - `🏥 加分 1 分：醫療保險個人支出預測 (Problem 2 CRISP-DM 報告)`
>        - `⚡ 加分 2 分：可任意輸入自訂 Dataset 即時機器學習建模與預測`
>      - 📑 **綜合瀏覽**：`📑 完整展開檢閱模式 (全部內容順序顯示)`
> 
> 2. **互動體驗細節**：
>    - 導航列下拉選單、頁首主切換選單、以及快速切換按鈕 (`[King County 主作業]`, `[加分題專區]`, `[全部展開]`) **三方 100% 雙向即時連動**。
>    - 搭配即時狀態條（包含當前模式說明與「一鍵切換至加分題」快捷按鈕），操作流暢直覺。

---

## 回合十二：刪除多餘下拉列與藍底區塊、分拆加分項為獨立頁面、Problem 2 改用 Kaggle >13 特徵資料集

### 👤 使用者 (User)
> **訊息內容**：  
> 圖上的這個下拉式選單以及藍底得文字部分也刪掉  
> 下拉式選單中"綜合瀏覽"也刪除  
> 並將problem2和可任意輸入dataset的兩個功能分成兩個部份(不要在同一頁)  
> problem2的資料集特徵要至 Kaggle 選擇一個具有>13 個特徵 (features) 的公開資料集。  

### 🤖 AI 助手 (Assistant)
> 已嚴格按照使用者需求完成四大修正：
> 
> 1. **移除頂端導航列下拉選單與藍底文字區塊**：
>    - 刪除頂端 Sticky 導航列中的藍色下拉選單膠囊 (`.nav-dropdown-box`)。
>    - 刪除切換條下方的藍底文字狀態說明框 (`.current-mode-status`)，徹底消除視覺雜訊，版面極致純淨俐落。
> 
> 2. **刪除「綜合瀏覽」與「全部展開」選項**：
>    - 下拉式選單僅保留三個純粹獨立的視圖選項，完全移除「綜合瀏覽」分組與按鈕。
> 
> 3. **Problem 2 與「可任意輸入 Dataset」徹底獨立分拆（絕不同時出現在同一頁）**：
>    - **視圖一：🏡 King County 房價預測（主要作業）** — 18 特徵、9 大圖表、完整 CRISP-DM 流程。
>    - **視圖二：🌍 Problem 2 全球公民預期壽命預測（加分 1 分）** — 獨立專屬頁面，不混雜自訂上傳區。
>    - **視圖三：⚡ 可任意輸入 Dataset 即時多元回歸引擎（加 2 分）** — 獨立互動平台，不混雜 Problem 2 報告。
> 
> 4. **Problem 2 採用 Kaggle 經典 >13 特徵公開資料集**：
>    - **資料集**：[Kaggle Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) (`life_expectancy.csv`)。
>    - **特徵檢核**：具備 **20 個候選預測特徵**（嚴格滿足「>13 個特徵」之作業規範）。
>    - **實證指標**：多元回歸 $R^2 = 0.8195$、$\text{RMSE} = 3.95$ 歲；RFE 精選 10 核心特徵 $R^2 = 0.7898$；實測 95% 預測區間覆蓋率達 95.3%。
>    - **專屬圖表**：產出並嵌入 `figures/fig_life_prediction.png` 與 `figures/fig_life_eda.png`。

---

## 回合十三：提供 Problem 2 資料集與網址，並將分析內容建置至與主要作業完全等量齊觀之完整深度

### 👤 使用者 (User)
> **訊息內容**：  
> 提供problem2的資料及網址給我 並且分析要項主要作業那樣完整  

### 🤖 AI 助手 (Assistant)
> 已完整提供 Problem 2 資料集名稱、Kaggle 官方來源網址與 20 項特徵詳細說明，並將 Problem 2 的分析內容全面擴充至與主要作業 (King County) 完全等量齊觀之完整學術等級深度：
> 
> 1. **Problem 2 資料集名稱與 Kaggle 來源網址**：
>    - **資料集名稱**：Life Expectancy (WHO) - Statistical Analysis (`life_expectancy.csv`)
>    - **Kaggle 官方來源網址**：[https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
>    - **預測目標變數**：各國平均預期壽命 $y = \text{Life expectancy}$ (連續數值型，歲，範圍 36.3 ~ 89.0 歲)
>    - **樣本規模**：全球 193 個國家於 2000 年至 2015 年之跨國面板追蹤資料，共 2,938 筆觀測值 (有效樣本 2,928 筆)。
>    - **特徵數量檢核**：共具備 **20 個候選預測特徵（嚴格符合 >13 個特徵規範要求，20 > 13）**。
> 
> 2. **壹、分析任務 (Analysis Tasks) — Problem 2 完整落實**：
>    - **任務 1：線性回歸模型預測**：
>      - 單純線性迴歸 ($x = \text{Income composition of resources}$)：推導最小平方法解析解 $\hat{\beta}_1 = 32.3248, \hat{\beta}_0 = 48.7952$，擬合方程式 $\hat{y} = 48.80 + 32.32x$，$R^2 = 0.4965, \text{RMSE} = 6.60$ 歲。
>      - 多元線性迴歸 (Full OLS 20 特徵)：母體矩陣解析解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$，測試集 $R^2 = \mathbf{0.8195}$，$\text{Adj } R^2 = \mathbf{0.8131}$，$\text{RMSE} = \mathbf{3.9517}$ 歲，$\text{MAE} = \mathbf{2.9273}$ 歲（較單純迴歸大幅躍升 +65.1%，誤差縮減 40.1%）。
>      - 嵌入圖表：`figures/p2_fig0_simple_linear_regression.png`。
>    - **任務 2：特徵選擇 (Feature Selection)**：
>      - RFE 遞迴特徵消除法精選 10 大核心特徵（精簡 50% 特徵，保留 Status_Developed, infant deaths, BMI, under-five deaths, Total expenditure, Diphtheria, HIV/AIDS, thinness 1-19 years, Income composition of resources, Schooling）。
>      - 效能對比：$R^2$ 仍維持高達 **0.7898**，$\text{RMSE} = 4.2641$ 歲，兼顧模型簡約與高預測力。
>      - 嵌入圖表：`figures/p2_fig4_feature_selection_comparison.png` 與 `figures/p2_fig8_feature_importance.png`。
>    - **任務 3：模型評估 (Model Evaluation)**：
>      - 六大模型橫向對比總表：單純 OLS、全特徵多元 OLS、RFE 精選 OLS、Ridge (α=1.0)、Lasso (α=0.01)、ElasticNet (α=0.01, l1=0.5)。
>      - 高斯-馬可夫四大檢定：Durbin-Watson $d = 2.0072$（接近 2.0，無序列自相關）、Jarque-Bera 常態檢定 ($JB = 67.48$)、均齊變異數性與 VIF 多重共線性檢定。
>      - 嵌入圖表：`figures/p2_fig5_model_comparison_metrics.png` 與 `figures/p2_fig7_residual_diagnostics.png`。
>    - **任務 4：結果預測圖 (加上信賴區間與預測區間)**：
>      - 完整推導多元回歸 95% 信賴區間 (CI) 與 95% 預測區間 (PI) 矩陣公式。
>      - 實測覆蓋率：測試集 586 筆樣本外資料中，**實際覆蓋率高達 95.3%**，完全符合理論 95% 信心水準。
>      - 嵌入圖表：`figures/p2_fig6_prediction_intervals.png`。
> 
> 3. **貳、CRISP-DM 流程說明 — Problem 2 六大標準階段詳盡論述**：
>    - 階段 1 Business Understanding：聯合國 SDG 3 公衛政策與資源配置目標。
>    - 階段 2 Data Understanding：20 大特徵中英文完整資料字典 (Data Dictionary)、特徵分佈圖 (`p2_fig1_eda_distributions.png`)、相關熱力圖 (`p2_fig2_eda_correlation_heatmap.png`)、關鍵特徵散布圖 (`p2_fig3_eda_target_relationships.png`)。
>    - 階段 3 Data Preparation：中位數插補、Status One-Hot 二元編碼、80/20 訓練測試切分、Z-Score 標準化。
>    - 階段 4 Modeling：OLS 解析解、RFE 精簡建模、Ridge/Lasso/ElasticNet 正則化損失函數與超參數調校。
>    - 階段 5 Evaluation：綜合評估成果與跨國公衛實務決策洞察（受教育年數與 HDI 收入指數為最大正向推手；HIV/AIDS 與成人死亡率為最大負向殺手）。
>    - 階段 6 Deployment：FastAPI RESTful API 微服務端點 (`/api/v1/predict-life-expectancy`)、動態政策模擬儀表板與年度資料漂移監控管線。
> 
> 4. **檔案同步更新**：
>    - [HW2_Report.html](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.html)
>    - [HW2_Report.md](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/HW2_Report.md)
>    - [chat_log.md](file:///c:/Users/88693/Desktop/HW2%20Multiple%20Linear%20regression/chat_log.md)

---

## 回合十四：修正網頁中的 Math Input Error，確保數學公式符號與版面比例完全貼齊主要作業

### 👤 使用者 (User)
> **訊息內容**：  
> 修正網頁中的math input error  
> 數學公式符號與版面比例請跟主要作業一樣  
> *(附圖：顯示 `\hat{y} = 48.80 + 32.32imesextIncomecompositionofresources` 算式解析錯誤)*

### 🤖 AI 助手 (Assistant)
> 已澈底排查並修正網頁中所有的 MathJax 渲染解析錯誤，並將數學符號與版面比例調整至與主要作業完全一致：
> 
> 1. **根因分析與錯誤修正**：
>    - **問題根因**：前次寫入 HTML 時未採用 Raw String (`r"""`)，導致 Python 將 LaTeX 反斜線字元轉譯為跳脫字元（例如 `\times` 變為 `\t` (TAB) + `imes`；`\text` 變為 `\t` (TAB) + `ext`；`\beta` 變為 `\b` (退格鍵) + `eta`；`\frac` 變為 `\f` (換頁鍵) + `rac`）。MathJax 引擎在解析時拋出語法解析錯誤或呈現失真的未轉譯字串。
>    - **修復落實**：以 Raw String 格式重新建構所有方程式，徹底消除所有非法字元，經自動化語法檢查確認 43 個 Block Math 方程式與 102 個 Inline Math 符號皆為 100% 正確合規之 LaTeX 語法。
> 
> 2. **數學公式符號與主要作業標準完全一致**：
>    - 單純線性迴歸擬合式：明確修正為標準 LaTeX 格式：
>      $$\hat{y} = 48.80 + 32.32 \times \text{Income composition of resources}$$
>    - 單純迴歸解析解推導式：包含最小平方法連加符號 $\sum$、平均值符號 $\bar{x}, \bar{y}$ 與係數估計 $\hat{\beta}_1, \hat{\beta}_0$。
>    - 多元回歸母體模型與矩陣解析解：嚴格維持矩陣粗體 $\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$ 與解析解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$。
>    - 評估指標數學定義：完整補齊與主要作業相同格式之 $\text{RMSE}$、$\text{MAE}$、$R^2$、$R^2_{\text{adj}}$ 數學公式。
>    - 正則化損失函數：完整補齊 Ridge、Lasso、ElasticNet 之理論損失函數目標式。
>    - 信賴區間與預測區間：精確呈現點估計值 $\hat{y}_0$、95% 信賴區間 $\text{CI}_{1-\alpha}$ 與 95% 預測區間 $\text{PI}_{1-\alpha}$ 矩陣推導方程式。
> 
> 3. **版面比例全面對齊**：
>    - 卡片元件結構 (`.content-card`)、標題排版 (`h3`, `h4`, `.card-header-clean`)、雙欄統計網格 (`.stat-grid-2`)、評估比較表格 (`.table-container`)、重點標記框 (`.callout-box`) 以及圖表展示容器 (`.figure-wrapper`) 均完全沿用主要作業的精準樣式比例。

---

## 回合十五：版面比例與留白全面優化（還原主要作業優雅留白紙張質感與舒適閱讀比例）

### 👤 使用者 (User)
> **訊息內容**：  
> 比例差太多了 我喜歡有留白的版本  
> *(附圖 1：Problem 2 畫面呈現過寬橫幅、深黑底色標題塊與平鋪文本)*  
> *(附圖 2：King County 主要作業呈現相同過寬橫向拉伸狀態)*  
> *(附圖 3：加分題 2 自訂資料集輸入列與指標卡被過度拉伸為細長條狀)*

### 🤖 AI 助手 (Assistant)
> 已針對使用者回饋之版面比例過寬、缺乏邊界留白及深色背景框問題進行全面樣式重構，徹底還原主要作業「經典優雅紙張留白」與「精準閱讀比例」：
> 
> 1. **容器寬度與邊界留白全面重塑 (Generous Whitespace & Proportions)**：
>    - 將主容器限制於最適閱讀比例 `max-width: 940px`，置中對齊（`margin: 0 auto`），兩側賦予溫和淡雅的底色與充裕的呼吸空間（`padding: 24px 20px 80px`）。
>    - 彻底解決截圖中在寬螢幕下橫向過度拉伸、扁平失衡與視線難以聚焦的問題。
> 
> 2. **單一整體紙張質感設計 (`.report-paper`)**：
>    - 放棄原本將每一個任務段落切割為破碎零散卡片的作法，改為將「King County 主要作業」、「Problem 2 全球壽命報告」與「加分題 2 自訂資料集分析引擎」三大視圖各自包覆於獨立的一體化白底紙張容器中（`background: #ffffff; padding: 50px 65px; border-radius: 18px; box-shadow: 0 4px 25px rgba(0,0,0,0.05);`）。
>    - 內部段落與分析內容自然流暢展開，兼具高雅報告排版與充足內邊距留白。
> 
> 3. **徹底移除深色背景框，回歸純淨標題排版**：
>    - 依使用者要求徹底移除所有帶深藍/深黑背景的區塊標題（`.section-header-banner` 背景設為純透明無框線陰影）。
>    - 改以主要作業深受喜愛的「左側 5px 品牌藍強調邊條 (`border-left: 5px solid #2563eb; padding-left: 14px;`)」與底部淡雅分割線呈現章節層級。
> 
> 4. **雙欄指標統計網格修復 (`.stat-grid-2`)**：
>    - 針對截圖中 Simple LR 與 Multiple LR 擠成一團文字的問題，落實 `.stat-grid-2` 兩欄對稱網格樣式（`display: grid; grid-template-columns: 1fr 1fr; gap: 18px;`）。
>    - 指標數值、標籤與文字說明清晰分列於兩枚精巧的統計卡片內，比例完美平衡。
> 
> 5. **三大模式比例統一**：
>    - 透過頂端緊湊精緻的控制條（下拉式選單與切換標籤按鈕）在三大視圖間順暢切換，每一種視圖均嚴格遵循 940px 紙張留白排版標準。

---

## 回合十六：修復兩個加分項容器層級，全面落實與圖一（主要作業）完全一致的最適寬度與留白

### 👤 使用者 (User)
> **訊息內容**：  
> 可以修該兩個加分項的設計嗎，我想像圖一(諸藥作業)一樣有最適閱讀寬度與充裕兩側留白  
> *(附圖 1：King County 主要作業在置中白底紙張中完美呈現，具備最適閱讀寬度與充裕兩側自然留白)*  
> *(附圖 2：Problem 2 全球壽命加分題視圖橫向鋪滿全螢幕，兩側未留白)*  
> *(附圖 3：加分題 2 自訂資料集視圖橫向鋪滿全螢幕，兩側未留白)*

### 🤖 AI 助手 (Assistant)
> 精準排查並修復 HTML 標籤嵌套問題，使兩個加分項的視覺比例與兩側留白完全對齊圖一（主要作業）：
> 
> 1. **根因定位與修復 (Root Cause Analysis)**：
>    - **問題根因**：在 `problem1-container` 結尾處存在一個提早閉合的 `</main>` 標籤，導致瀏覽器判定 `<main class="main-container">` 已在 Problem 1 結束，進而將隨後的 `problem2-container` 與 `custom-dataset-container` 溢出脫離至最外層的 `<body>`，使得在寬螢幕下失去了 `.main-container` 限制之 `max-width: 940px` 與置中留白效果，橫向擴展至 100%。
>    - **標籤結構修復**：移除提前閉合的 `</main>` 標籤，使 `<main class="main-container">` 完整且正確地包裹全部三個容器（`problem1-container`、`problem2-container` 與 `custom-dataset-container`），並統一在自訂資料集結尾處正式閉合。
> 
> 2. **強化紙張容器規格 (`.report-paper`)**：
>    - 在 CSS 中為 `.report-paper` 顯式聲明 `width: 100%; max-width: 940px; margin: 0 auto 40px; box-sizing: border-box;`。
>    - 雙重保險確保無論處於何種切換狀態，Problem 2 與自訂資料集均嚴格繼承如圖一所示之 940px 置中優雅白底紙張卡片質感（`padding: 50px 65px; border-radius: 18px; box-shadow: 0 4px 25px rgba(0,0,0,0.05);`）。
> 
> 3. **三大視圖完全一致的視覺體驗**：
>    - **圖一 (King County 主要作業)**：保持舒適 940px 寬度與充裕左右留白。
>    - **圖二 (Problem 2 全球壽命加分題)**：完全收斂至相同的 940px 居中紙張中，標題、算式、雙欄統計卡片與圖表自然流暢，兩側重現寬裕的淡雅留白。
>    - **圖三 (加分題 2 自訂資料集分析引擎)**：輸入分頁標籤、CSV 輸入框、4 枚即時指標卡與預測模擬滑桿均完美收納於 940px 紙張容器內，徹底告別邊緣貼邊拉伸。

---

## 回合十七：解析 GitHub Pages 呈現 Repo/Markdown 形式之根因並建立 index.html 與 .nojekyll

### 👤 使用者 (User)
> **訊息內容**：  
> 為甚麼github action page顯示出來的報告是repo的形式而不是web

### 🤖 AI 助手 (Assistant)
> 為使用者深入解析 GitHub Pages 的路由載入機制，並完成靜態首頁設定：
> 
> 1. **問題根本原因 (Root Cause)**：
>    - GitHub Pages 預設必須以根目錄下的 `index.html` 作為預設首頁入口。
>    - 專案原本的主網頁檔名為 `HW2_Report.html`，根目錄中缺乏 `index.html`。
>    - 當根目錄沒有 `index.html` 時，GitHub Pages 會自動觸發內建的 Jekyll 引擎，將 `README.md` 套用 GitHub 預設 Markdown 主題渲染成頁面，上方因而出現倉庫名稱標題（`HW2-Multiple-Linear-regression`），視覺上看起來就像 Repository 的說明文件而非 Web 應用。
> 
> 2. **修復落實**：
>    - **建立 `index.html`**：將完整的 Web 報告內容複製並建立為 `index.html`，讓使用者直接進入 `https://<username>.github.io/<repo>/` 即可直接瀏覽精美的 Web 報告。
>    - **建立 `.nojekyll`**：在根目錄新增空的 `.nojekyll` 檔案，告知 GitHub Pages 繞過 Jekyll 編譯，原生託管靜態 HTML/CSS/JS 與 MathJax 資源。
> 
> 3. **推送生效方式**：
>    - 只要將 `index.html` 與 `.nojekyll` 透過 Git 推送至 GitHub（`git add .`、`git commit`、`git push`），GitHub Actions 即會自動重新部署，呈現完整的 Web 風格報告。



