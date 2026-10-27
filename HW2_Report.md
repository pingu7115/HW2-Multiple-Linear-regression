# 作業二：多元線性迴歸分析與 CRISP-DM 預測模型報告
## HW2: Multiple Linear Regression & CRISP-DM Analysis Report

- **課程名稱**：巨量資料分析與應用 (Big Data Analytics & Applications)
- **主要作業 (Problem 1)**：House Sales in King County, USA (`kc_house_data.csv`)，18 特徵，21,613 筆樣本，目標：房屋價格 $y = \text{price}$
- **加分題 1 (Problem 2)**：Life Expectancy (WHO) (`life_expectancy.csv`)，**20 特徵 (嚴格符合 >13 特徵規範)**，2,938 筆樣本，目標：公民平均壽命 $y = \text{Life expectancy}$
  - **Kaggle 來源網址**：[https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **加分題 2 (Custom Dataset)**：可任意輸入 Dataset 即時多元線性迴歸引擎 (線上網址、檔案上傳、貼上純文字、動態滑桿預測)

---

## 🌐 Live Demo & Interactive Web Application

🐙 **GitHub Repository**: **[https://github.com/pingu7115/HW2-Multiple-Linear-regression](https://github.com/pingu7115/HW2-Multiple-Linear-regression)**  
🔗 **Live Demo (GitHub Pages)**: **[https://pingu7115.github.io/HW2-Multiple-Linear-regression/](https://pingu7115.github.io/HW2-Multiple-Linear-regression/)**

[![作業二：多元線性迴歸互動式 Web 分析報告與預測模擬平台](dashboard_demo.png)](https://pingu7115.github.io/HW2-Multiple-Linear-regression/)

> 💡 **點擊上方預覽圖或 Live Demo 連結，即可直接於瀏覽器啟動互動式 Web 分析報告！**  
> 內建「**King County 主要作業**」、「**加分題 1：Problem 2 全球壽命預測**」與「**加分題 2：可任意輸入 Dataset 即時多元迴歸引擎**」三大獨立分頁檢閱模式，支援 LaTeX 數學方程式、多模型橫向評估與動態數值滑桿預測模擬。

---

# 🏡 第一部分：主要作業 (Problem 1: King County 房價預測)

## 壹、分析任務 (Analysis Tasks)

### 1. 使用線性回歸 (Linear Regression) 模型進行預測
- **單純線性迴歸 (Simple LR: sqft_living)**：
  - 擬合方程式：$\hat{y} = -41,999.19 + 279.55 \times \text{sqft\_living}$
  - 測試集指標：$R^2 = 0.4941$，$RMSE = \$276,559.17$，$MAE = \$177,867.54$。
  - 圖 0：單純線性迴歸房價預測圖 (包含 95% 信賴區間與 95% 預測區間)
- **多元線性迴歸 (Multiple LR: Full 17 Features)**：
  - 理論矩陣解：$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
  - 測試集指標：$R^2 = 0.7012$，$RMSE = \$212,539.52$，$MAE = \$127,493.34$。
  - 效益比較：決定係數大幅提升 **+41.9%**，均方根誤差縮減 **\$64,020 美元**。

### 2. 必須執行 特徵選擇 (Feature Selection)
- **RFE 遞迴特徵消除法精選 12 個特徵**：
  $$\text{Selected} = \{\text{bedrooms}, \text{bathrooms}, \text{sqft\_living}, \text{waterfront}, \text{view}, \text{condition}, \text{grade}, \text{sqft\_above}, \text{yr\_built}, \text{zipcode}, \text{lat}, \text{long}\}$$
- **剔除次要特徵**：`sqft_lot`, `floors`, `yr_renovated`, `sqft_living15`, `sqft_lot15`。
- **效益**：特徵數量精簡 29.4%，$R^2$ 仍維持高達 **0.6998**，兼顧精度與奧坎剃刀原則。

### 3. 必須執行 模型評估 (Model Evaluation)
- **六大模型測試集比較**：
  | 模型名稱 | 特徵數 | 測試集 $R^2$ | 測試集 Adj $R^2$ | RMSE (\$) | MAE (\$) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | 單純線性迴歸 (sqft_living) | 1 | 0.4941 | 0.4940 | \$276,559.17 | \$177,867.54 |
  | 多元線性迴歸 (全特徵) | 17 | **0.7012** | **0.7000** | **\$212,539.52** | \$127,493.34 |
  | RFE 精選模型 (Top 12) | 12 | **0.6998** | **0.6990** | \$213,033.84 | \$127,884.28 |
  | Ridge 嶺迴歸 (L2) | 17 | 0.7012 | 0.7000 | \$212,547.03 | \$127,474.60 |
  | Lasso 套索迴歸 (L1) | 17 | 0.7011 | 0.6999 | \$212,567.97 | \$127,456.87 |
  | ElasticNet 彈性網絡 | 17 | 0.6983 | 0.6971 | \$213,560.47 | **\$126,380.60** |
- **高斯-馬可夫四大檢定**：
  - Durbin-Watson 無自相關統計量 $d = 1.9939$（接近 2.0）。
  - 變異數膨脹因子 VIF $< 7.3$，無嚴重共線性。

### 4. 結果部分需包含提供預測圖 (加上信賴區間或預測區間)
- 95% 信賴區間：$\text{CI}_{0.95} = \hat{y}_0 \pm t_{0.025, n-p-1} s \sqrt{\mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$
- 95% 預測區間：$\text{PI}_{0.95} = \hat{y}_0 \pm t_{0.025, n-p-1} s \sqrt{1 + \mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$
- 實測覆蓋率：**95.2%** 測試樣本落入預測區間帶內。

## 貳、CRISP-DM 流程說明 (Problem 1)
1. **Business Understanding**：消除房屋交易資訊不對稱，提供房仲與金融機構具信賴/預測區間之估價依據。
2. **Data Understanding**：21,613 筆，18 個候選特徵，繪製特徵分佈 (圖 1)、相關矩陣 (圖 2)、關聯散佈圖 (圖 3)。
3. **Data Preparation**：清洗完全共線特徵 `sqft_basement`，80/20 訓練測試切分，Z-Score 標準化。
4. **Modeling**：最小平方法解析解、RFE 特徵消除、Ridge/Lasso/ElasticNet 正則化。
5. **Evaluation**：RFE 精選 12 特徵達成甜點平衡 ($R^2 = 0.6998$)，95% PI 覆蓋率 95.2%。
6. **Deployment**：FastAPI 微服務端點 `/api/v1/predict-price`，Airflow MLOps 漂移監控管線。

---

# 🌍 第二部分：加分題 1 (Problem 2: WHO 全球公民預期壽命預測)

## 壹、資料集基本資訊與 Kaggle 來源
- **資料集名稱**：Life Expectancy (WHO) - Statistical Analysis (`life_expectancy.csv`)
- **Kaggle 官方來源網址**：[https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **特徵數量規範核可**：**20 個候選預測特徵（嚴格符合 >13 個特徵之作業規範，20 > 13）**
- **預測目標變數**：各國平均預期壽命 $y = \text{Life expectancy}$ (連續數值型，歲，範圍 36.3 ~ 89.0 歲)
- **樣本規模**：全球 193 個國家，2000 年至 2015 年跨國面板資料，共 2,938 筆觀測值 (有效樣本 2,928 筆)。

### 20 個預測特徵清單與資料字典 (Data Dictionary)
1. `Year` (年份，2000-2015)
2. `Status_Developed` (國家發展狀態：Developed 1, Developing 0)
3. `Adult Mortality` (成人死亡率：每千人中 15-60 歲死亡機率)
4. `infant deaths` (嬰兒死亡數：每千名活產嬰兒之死亡人數)
5. `Alcohol` (純酒精消費量：人均每年純酒精公升數)
6. `percentage expenditure` (醫療衛生支出佔人均 GDP 百分比 %)
7. `Hepatitis B` (B型肝炎疫苗覆蓋率 %)
8. `Measles` (麻疹通報確診數)
9. `BMI` (國民平均身體質量指數)
10. `under-five deaths` (五歲以下兒童死亡數)
11. `Polio` (小兒麻痺疫苗接種率 %)
12. `Total expenditure` (政府衛生支出佔政府總支出比 %)
13. `Diphtheria` (白喉三合一疫苗接種率 %)
14. `HIV/AIDS` (愛滋病毒死亡率)
15. `GDP` (人均國內生產毛額，美元)
16. `Population` (國家總人口數)
17. `thinness 1-19 years` (青少年身體過輕率 %)
18. `thinness 5-9 years` (學齡兒童身體過輕率 %)
19. `Income composition of resources` (資源收入構成指數，HDI 核心分項 [0, 1])
20. `Schooling` (平均受教育年數，年)

---

## 貳、分析任務 (Analysis Tasks) — Problem 2

### 1. 使用線性回歸 (Linear Regression) 模型進行預測
- **單純線性迴歸 (Simple Linear Regression)**：
  - 自變數：相關係數最高特徵 $x = \text{Income composition of resources}$ ($r = 0.72$)
  - OLS 解析解：
    $$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = 32.3248, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 48.7952$$
    $$\hat{y} = 48.80 + 32.32 \times \text{Income composition of resources}$$
  - 測試集表現：$R^2 = 0.4965$，$RMSE = 6.5992$ 歲，$MAE = 4.4567$ 歲。
  - 圖 P2-0：單純線性迴歸預測圖與 95% 信賴/預測區間 (`figures/p2_fig0_simple_linear_regression.png`)
- **多元線性迴歸 (Multiple Linear Regression: Full 20 Features)**：
  - 理論矩陣解：$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
  - 測試集表現：$R^2 = \mathbf{0.8195}$，$\text{Adj } R^2 = \mathbf{0.8131}$，$\text{RMSE} = \mathbf{3.9517}$ 歲，$\text{MAE} = \mathbf{2.9273}$ 歲。
  - 效益比較：決定係數相較於單純線性迴歸大幅躍升 **+65.1%**，均方根誤差大幅縮減 **2.65 歲 (-40.1%)**，多維衛生指標解釋力極強！

### 2. 必須執行 特徵選擇 (Feature Selection)
- **RFE 遞迴特徵消除法篩選 10 個核心特徵**：
  $$\text{Selected (10)} = \{\text{Status\_Developed}, \text{infant deaths}, \text{BMI}, \text{under-five deaths}, \text{Total expenditure}, \text{Diphtheria}, \text{HIV/AIDS}, \text{thinness 1-19 years}, \text{Income composition of resources}, \text{Schooling}\}$$
- **剔除次要特徵**：`Year`, `Adult Mortality`, `Alcohol`, `percentage expenditure`, `Hepatitis B`, `Measles`, `Polio`, `GDP`, `Population`, `thinness 5-9 years`。
- **特徵選擇前後效能對比**：
  - 全特徵 (20 Feat)：$R^2 = 0.8195$，$RMSE = 3.9517$ 歲，$MAE = 2.9273$ 歲
  - RFE 精選 (10 Feat)：$R^2 = 0.7898$，$RMSE = 4.2641$ 歲，$MAE = 3.1812$ 歲
  - 決策權衡：特徵數量精簡 50%，$R^2$ 僅微降 0.03，RMSE 僅微增 0.31 歲，大幅降低國際普查維護負擔。
- 圖 P2-4：特徵選擇前後效能對比長條圖 (`figures/p2_fig4_feature_selection_comparison.png`)
- 圖 P2-8：20 特徵標準化回歸係數 (Beta Coefficients β*) 排序圖 (`figures/p2_fig8_feature_importance.png`)

### 3. 必須執行 模型評估 (Model Evaluation)
- **六大模型實測評估比較表**：
  | 模型名稱 (Model) | 特徵數量 | 測試集 $R^2$ | 測試集 Adj $R^2$ | RMSE (歲) | MAE (歲) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | 單純線性迴歸 (Income) | 1 | 0.4965 | 0.4956 | 6.5992 | 4.4567 |
  | 多元線性迴歸 (Full OLS 20 特徵) 🏆 | 20 | **0.8195** | **0.8131** | **3.9517** | **2.9273** |
  | RFE 特徵選擇迴歸 (10 特徵) | 10 | 0.7898 | 0.7861 | 4.2641 | 3.1812 |
  | Ridge 嶺迴歸 (L2, α=1.0) | 20 | 0.8192 | 0.8128 | 3.9542 | 2.9268 |
  | Lasso 套索迴歸 (L1, α=0.01) | 20 | 0.8187 | 0.8123 | 3.9606 | 2.9288 |
  | ElasticNet 彈性網絡 (α=0.01, l1=0.5) | 20 | 0.8147 | 0.8082 | 4.0033 | 2.9678 |
- 圖 P2-5：六大模型 $R^2$ 與 RMSE 評估長條圖 (`figures/p2_fig5_model_comparison_metrics.png`)
- **高斯-馬可夫四大檢定**：
  1. 殘差常態性：Jarque-Bera 統計量 $JB = 67.48$ ($p = 2.22 \times 10^{-15}$)，中心區段高度常態。
  2. 殘差獨立性：Durbin-Watson 統計量 $d = \mathbf{2.0072}$（極為接近 2.0，確認無一階序列自相關）。
  3. 均齊變異數性：Scale-Location 圖呈橫向平滑帶狀，無嚴重發散異質變異。
  4. 多重共線性：排除重合特徵後矩陣條件數良好，各特徵 VIF 受控。
- 圖 P2-7：高斯-馬可夫四大殘差診斷組合圖 (`figures/p2_fig7_residual_diagnostics.png`)

### 4. 結果部分需包含提供預測圖 (加上信賴區間或預測區間)
- 95% 信賴區間：$\text{CI}_{0.95} = \hat{y}_0 \pm t_{0.025, n-p-1} s \sqrt{\mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$
- 95% 預測區間：$\text{PI}_{0.95} = \hat{y}_0 \pm t_{0.025, n-p-1} s \sqrt{1 + \mathbf{x}_0^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_0}$
- 實測覆蓋率：在 586 筆測試樣本中，**實測覆蓋率高達 95.3%**，完全符合 95% 統計信心水準。
- 圖 P2-6：多元線性迴歸 95% 預測區間圖 (`figures/p2_fig6_prediction_intervals.png`)

---

## 參、CRISP-DM 流程說明 (Problem 2)
1. **Business Understanding (商業理解)**：
   - 聚焦聯合國 SDG 3 (良好健康與福祉)，協助各國公衛部門與 WHO 量化評估教育投資、疫苗普及、傳染病防治對預期壽命之邊際效益，優化跨國有限預算分配。
2. **Data Understanding (資料理解)**：
   - 2,938 筆跨國觀測值，20 個預測特徵與預期壽命目標變數。
   - 繪製特徵分佈 (圖 P2-1)、20 特徵相關矩陣 (圖 P2-2)、關鍵公衛指標散佈圖 (圖 P2-3)。
3. **Data Preparation (資料準備)**：
   - 剔除 10 筆目標變數缺失樣本，中位數插補各國通報延遲特徵。
   - 國家發展狀態 `Status` One-Hot 編碼為 `Status_Developed` (0/1)。
   - 80/20 訓練測試切分 (`random_state=42`)，Z-Score 標準化轉換。
4. **Modeling (模型建立)**：
   - 普通最小平方法 (OLS) 解析解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$。
   - RFE 遞迴特徵消除精選 10 大關鍵指標。
   - Ridge (L2)、Lasso (L1)、ElasticNet 損失函數與超參數調校。
5. **Evaluation (模型評估)**：
   - 全特徵多元回歸達成 $R^2 = 0.8195$、$\text{RMSE} = 3.95$ 歲；95% 預測區間覆蓋率達 95.3%。
   - 公衛實證洞察：教育受教年數與 HDI 收入指數為最關鍵推動力；愛滋病與成人死亡率為主要剝奪因子。
6. **Deployment (模型部署)**：
   - 建置 FastAPI 微服務端點 `/api/v1/predict-life-expectancy`，提供各國輸入公衛數據即時試算壽命與 95% 預測區間。
   - 規劃年度資料漂移監控管線與政策模擬儀表板。

---

# ⚡ 第三部分：加分題 2 (可任意輸入 Dataset 即時多元回歸引擎)

- **獨立專屬展示頁面**：完全自成一體，與主要作業及 Problem 2 分頁隔離。
- **多元輸入模式**：支援輸入線上 CSV 網址（含 50_Startups、Medical Insurance、Boston Housing 快捷按鈕）、上傳本機 `.csv` 檔案（離線原生解析）、貼上 CSV 純文字。
- **瀏覽器原生機器學習運算引擎 (`ml_engine.js`)**：
  - 自動欄位探勘、類別型變數 One-Hot Encoding、80/20 訓練/測試集自動切割。
  - 線性代數矩陣求逆解析解求解 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$。
  - 即時呈現決定係數 $R^2$、調整後 $R^2$、RMSE、MAE、擬合數學方程式與特徵重要度權重列表。
  - **即時互動預測模擬器 (Live Predictor)**：拖曳數值滑桿，即時給出預測值、95% 信賴區間與 95% 預測區間！
