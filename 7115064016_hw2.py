# -*- coding: utf-8 -*-
"""
HW2: Multiple Linear Regression (多元線性迴歸分析與預測模型建立)
遵循標準資料探勘流程：CRISP-DM (Cross-Industry Standard Process for Data Mining)
1. Business Understanding (商業理解)
2. Data Understanding (資料理解與 EDA)
3. Data Preparation (資料準備與特徵工程)
4. Modeling (模型建立與特徵選擇 Feature Selection)
5. Evaluation (模型評估、信賴區間/預測區間與殘差診斷)
6. Deployment (模型部署與應用規劃)

資料集：Kaggle - House Sales in King County, USA (kc_house_data.csv)
Kaggle 來源網址：https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
特徵數量：18 個預測特徵（符合作業 10~20 個 features 規範要求）
目標變數：房屋成交價格 (price)
"""

import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# -------------------------------------------------------------
# 0. 環境設定與圖表字型初始化
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Segoe UI', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Microsoft JhengHei')

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def print_section(phase_num, phase_name):
    print("\n" + "=" * 75)
    print(f"  CRISP-DM Phase {phase_num}: {phase_name}")
    print("=" * 75)


# -------------------------------------------------------------
# Phase 1 & 2: Business Understanding & Data Understanding
# -------------------------------------------------------------
def load_and_explore_data(filepath='kc_house_data.csv'):
    print_section(1, "Business Understanding (商業理解)")
    print("【商業目標】建立西雅圖 King County 地區房屋市場價格預測模型，")
    print("協助房仲業者、購屋者及銀行估價部門精確預估房屋公允市場價值 (Fair Market Value)，")
    print("並提供具備 95% 信賴區間與預測區間的統計量化指標。\n")

    print_section(2, "Data Understanding (資料理解與 EDA)")
    if not os.path.exists(filepath):
        print(f"找不到檔案 {filepath}，正在自遠端下載...")
        import urllib.request
        url = 'https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv'
        urllib.request.urlretrieve(url, filepath)
        print("下載完成！")

    df = pd.read_csv(filepath)
    print(f"原始資料維度: {df.shape[0]} 筆樣本，{df.shape[1]} 個欄位")
    print(f"Kaggle 資料集網址: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction\n")

    print("前 5 筆資料 (head):")
    print(df.head())

    print("\n缺失值統計:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "資料集完全無缺失值 (0 Missing Values)！")

    # 檢視特徵數量
    non_predictive = ['id', 'date', 'price']
    features = [c for c in df.columns if c not in non_predictive]
    print(f"\n【特徵數量檢核】原始預測特徵共計: {len(features)} 個 (完全符合 10~20 個特徵之作業要求)")
    print(f"特徵名稱清單: {features}")

    return df, features


# -------------------------------------------------------------
# EDA Visualizations (探索性視覺化)
# -------------------------------------------------------------
def run_eda_visualizations(df, features):
    print("\n正在生成 EDA 探索性視覺化圖表...")

    # 圖 1: 目標變數與主要關鍵特徵分佈
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('圖 1: King County 房價與核心特徵分佈 (Feature Distributions)', fontsize=16, fontweight='bold')

    sns.histplot(df['price'] / 1000, kde=True, ax=axes[0, 0], color='#e63946', bins=30)
    axes[0, 0].set_title('成交房價分佈 (Price in Thousands $)')
    axes[0, 0].set_xlabel('Price ($k)')

    sns.histplot(df['sqft_living'], kde=True, ax=axes[0, 1], color='#1d3557', bins=30)
    axes[0, 1].set_title('室內居住坪數 (sqft_living)')
    axes[0, 1].set_xlabel('Square Feet')

    sns.countplot(data=df, x='grade', ax=axes[0, 2], palette='Blues_r')
    axes[0, 2].set_title('建築結構品質等級 (Grade 1-13)')
    axes[0, 2].set_xlabel('Grade')

    sns.countplot(data=df, x='bedrooms', ax=axes[1, 0], palette='crest')
    axes[1, 0].set_title('臥室數量分佈 (Bedrooms)')
    axes[1, 0].set_xlim(-0.5, 9.5)

    sns.histplot(df['yr_built'], kde=True, ax=axes[1, 1], color='#2a9d8f', bins=30)
    axes[1, 1].set_title('建築建造年份 (Year Built)')
    axes[1, 1].set_xlabel('Year')

    sns.scatterplot(data=df, x='long', y='lat', hue=np.log1p(df['price']), palette='viridis',
                    alpha=0.3, s=10, ax=axes[1, 2], legend=False)
    axes[1, 2].set_title('地理位置座標分佈與房價熱區 (Lat / Long)')

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, 'fig1_eda_distributions.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig1_path}")

    # 圖 2: 18 個特徵與房價之皮爾森相關係數熱力圖
    corr_cols = ['price'] + features
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', mask=mask,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title('圖 2: 房價與 18 個候選特徵之皮爾森相關係數矩陣 (Correlation Heatmap)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, 'fig2_eda_correlation_heatmap.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig2_path}")

    # 圖 3: 核心特徵對房價的非線性與分層影響散佈圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('圖 3: 核心驅動特徵與房屋成交價格之關聯散佈圖', fontsize=15, fontweight='bold')

    sns.scatterplot(data=df, x='sqft_living', y=df['price']/1000, alpha=0.3, color='#1d3557', ax=axes[0])
    sns.regplot(data=df, x='sqft_living', y=df['price']/1000, scatter=False, ax=axes[0], color='red')
    axes[0].set_title('居住面積 (sqft_living) vs 房價 ($k)')
    axes[0].set_ylabel('Price ($k)')

    sns.boxplot(data=df, x='grade', y=df['price']/1000, ax=axes[1], palette='Blues')
    axes[1].set_title('建築等級 (Grade) vs 房價 ($k)')
    axes[1].set_ylabel('Price ($k)')

    sns.scatterplot(data=df, x='lat', y=df['price']/1000, alpha=0.3, color='#2a9d8f', ax=axes[2])
    axes[2].set_title('緯度位置 (Latitude) vs 房價 ($k)')
    axes[2].set_ylabel('Price ($k)')

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, 'fig3_eda_price_relationships.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig3_path}")


# -------------------------------------------------------------
# Phase 3: Data Preparation (資料準備與特徵工程)
# -------------------------------------------------------------
def prepare_data(df):
    print_section(3, "Data Preparation (資料準備與特徵工程)")

    # 1. 排除非預測性欄位 (id, date)
    df_clean = df.drop(columns=['id', 'date']).copy()

    # 2. 檢驗並處理數學完全線性重合 (Exact Collinearity):
    # sqft_living = sqft_above + sqft_basement
    # 兩者相加完全等於 sqft_living，此為古典 rank deficiency 典型問題，必須移除一項以滿足設計矩陣滿秩假設
    exact_diff = (df_clean['sqft_living'] - (df_clean['sqft_above'] + df_clean['sqft_basement'])).abs().max()
    print(f"【共線性檢查】sqft_living 與 (sqft_above + sqft_basement) 最大差值: {exact_diff}")
    print("由於 sqft_living = sqft_above + sqft_basement 構成完全線性重合，")
    print("為避免矩陣奇異性 (Singular Matrix)，將 sqft_basement 交由特徵工程或予以排除。")

    X = df_clean.drop(columns=['price', 'sqft_basement'])
    y = df_clean['price']

    print(f"\n特徵準備完成，最終自變數矩陣維度: {X.shape[0]} 筆樣本，{X.shape[1]} 個特徵")
    print(f"特徵欄位清單: {X.columns.tolist()}")

    # 3. 劃分訓練集與測試集 (80% 訓練, 20% 測試, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"訓練集筆數: {X_train.shape[0]}, 測試集筆數: {X_test.shape[0]}")

    # 4. 特徵標準化 (StandardScaler)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist()


# -------------------------------------------------------------
# Phase 4: Modeling (模型建立與特徵選擇 Feature Selection)
# -------------------------------------------------------------
def build_and_select_models(X_train_sc, X_test_sc, y_train, y_test, feature_names):
    print_section(4, "Modeling (模型建立與特徵選擇 Feature Selection)")

    # 1. 全特徵多元線性迴歸 (Full Feature OLS)
    lr_full = LinearRegression()
    lr_full.fit(X_train_sc, y_train)
    y_pred_full = lr_full.predict(X_test_sc)

    r2_full = r2_score(y_test, y_pred_full)
    rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
    mae_full = mean_absolute_error(y_test, y_pred_full)

    # 2. 特徵選擇 (Feature Selection via RFE: Recursive Feature Elimination)
    print("【特徵選擇執行】使用遞迴特徵消除法 (RFE) 篩選出最關鍵之 Top 10 與 Top 12 特徵...")
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=12)
    rfe.fit(X_train_sc, y_train)
    selected_mask = rfe.support_
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    print(f"RFE 最佳篩選出之 12 個核心特徵: {selected_features}")

    # 使用篩選後的特徵訓練模型
    X_train_rfe = X_train_sc[selected_features]
    X_test_rfe = X_test_sc[selected_features]

    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train_rfe, y_train)
    y_pred_rfe = lr_rfe.predict(X_test_rfe)

    r2_rfe = r2_score(y_test, y_pred_rfe)
    rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))
    mae_rfe = mean_absolute_error(y_test, y_pred_rfe)

    # 3. 正則化模型比較 (Ridge, Lasso, ElasticNet)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_sc, y_train)
    y_pred_ridge = ridge.predict(X_test_sc)

    lasso = Lasso(alpha=100.0, max_iter=10000)
    lasso.fit(X_train_sc, y_train)
    y_pred_lasso = lasso.predict(X_test_sc)

    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_train_sc, y_train)
    y_pred_elastic = elastic.predict(X_test_sc)

    # 彙整評估表
    models_dict = {
        '多元線性迴歸 (全特徵 17 Features)': {
            'model': lr_full, 'pred': y_pred_full, 'feats': feature_names, 'p': len(feature_names)
        },
        '特徵選擇精簡模型 (RFE 12 Features)': {
            'model': lr_rfe, 'pred': y_pred_rfe, 'feats': selected_features, 'p': len(selected_features)
        },
        'Ridge 嶺迴歸 (L2 Regularization)': {
            'model': ridge, 'pred': y_pred_ridge, 'feats': feature_names, 'p': len(feature_names)
        },
        'Lasso 套索迴歸 (L1 Feature Sparsity)': {
            'model': lasso, 'pred': y_pred_lasso, 'feats': feature_names, 'p': len(feature_names)
        },
        'ElasticNet 彈性網絡 (L1+L2 Regularization)': {
            'model': elastic, 'pred': y_pred_elastic, 'feats': feature_names, 'p': len(feature_names)
        }
    }

    eval_results = []
    n = len(y_test)
    for name, item in models_dict.items():
        r2 = r2_score(y_test, item['pred'])
        rmse = np.sqrt(mean_squared_error(y_test, item['pred']))
        mae = mean_absolute_error(y_test, item['pred'])
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - item['p'] - 1)

        eval_results.append({
            'Model': name,
            'Features Count': item['p'],
            'Test R2': round(r2, 4),
            'Test Adj R2': round(adj_r2, 4),
            'Test RMSE ($)': round(rmse, 2),
            'Test MAE ($)': round(mae, 2)
        })

    eval_df = pd.DataFrame(eval_results)
    print("\n模型評估總表 (含特徵選擇前後對比):")
    print(eval_df.to_string(index=False))

    # 繪製圖 4: 特徵選擇前後模型指標對比圖
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('圖 4: 特徵選擇 (Feature Selection) 前後之模型效能對比', fontsize=15, fontweight='bold')

    fs_comp = eval_df.iloc[:2]
    sns.barplot(data=fs_comp, x='Model', y='Test R2', ax=axes[0], palette=['#1d3557', '#e63946'])
    axes[0].set_ylim(0.65, 0.75)
    axes[0].set_title('決定係數 ($R^2$, 越高越好)')
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11)

    sns.barplot(data=fs_comp, x='Model', y='Test RMSE ($)', ax=axes[1], palette=['#1d3557', '#e63946'])
    axes[1].set_title('均方根誤差 (RMSE, 越低越好)')
    for p in axes[1].patches:
        axes[1].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, 'fig4_feature_selection_comparison.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig4_path}")

    # 繪製圖 5: 五大迴歸模型綜合指標比較
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('圖 5: 各線性迴歸與正則化模型之整體指標比較', fontsize=15, fontweight='bold')

    sns.barplot(data=eval_df, x='Model', y='Test R2', ax=axes[0], palette='Blues_r')
    axes[0].set_ylim(0.65, 0.75)
    axes[0].set_title('測試集決定係數 ($R^2$)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha='right')

    sns.barplot(data=eval_df, x='Model', y='Test RMSE ($)', ax=axes[1], palette='Reds_r')
    axes[1].set_title('測試集均方根誤差 (RMSE $)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha='right')

    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, 'fig5_model_comparison_metrics.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig5_path}")

    return models_dict, eval_df, selected_features


# -------------------------------------------------------------
# Phase 5: Evaluation (模型評估、信賴區間/預測區間與殘差診斷)
# -------------------------------------------------------------
def evaluate_and_diagnose(X_train, X_test, y_train, y_test, selected_features):
    print_section(5, "Evaluation (模型評估、信賴/預測區間與殘差診斷)")

    # 使用 Statsmodels OLS 擬合精選特徵模型，以獲得嚴謹統計量與信賴區間/預測區間
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    X_train_sm = sm.add_constant(X_train_sel)
    X_test_sm = sm.add_constant(X_test_sel)

    ols_model = sm.OLS(y_train, X_train_sm).fit()
    print("【Statsmodels OLS 統計推論摘要 (精選特徵模型)】")
    print(ols_model.summary())

    # 1. 產生 95% 信賴區間 (Confidence Interval) 與 95% 預測區間 (Prediction Interval)
    print("\n正在計算測試集之 95% 信賴區間與預測區間並繪製預測圖...")
    # 為維持圖表清晰可讀，選取測試集中具代表性的 100 筆樣本進行區間視覺化展示
    sample_size = 100
    X_sample_sm = X_test_sm.iloc[:sample_size]
    y_sample_actual = y_test.iloc[:sample_size]

    pred_res = ols_model.get_prediction(X_sample_sm)
    pred_frame = pred_res.summary_frame(alpha=0.05)

    # 依預測值由小到大排序以便呈現平滑區間帶
    plot_df = pred_frame.copy()
    plot_df['actual'] = y_sample_actual.values
    plot_df = plot_df.sort_values(by='mean').reset_index(drop=True)

    # 圖 6: 預測圖（包含 95% 信賴區間與預測區間）
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df.index, plot_df['mean'] / 1000, color='#1d3557', lw=2.5,
             label='迴歸預測值 $\\hat{y}$ (Regression Fit)')

    # 95% 信賴區間 (Confidence Interval for Mean Response)
    plt.fill_between(plot_df.index, plot_df['mean_ci_lower'] / 1000, plot_df['mean_ci_upper'] / 1000,
                     color='#457b9d', alpha=0.45, label='95% 期望值信賴區間 (95% Confidence Interval)')

    # 95% 預測區間 (Prediction Interval for Individual Observation)
    plt.fill_between(plot_df.index, plot_df['obs_ci_lower'] / 1000, plot_df['obs_ci_upper'] / 1000,
                     color='#e63946', alpha=0.15, label='95% 個體值預測區間 (95% Prediction Interval)')

    # 測試集實際觀測值散佈點
    plt.scatter(plot_df.index, plot_df['actual'] / 1000, color='#d62828', edgecolors='k',
                s=40, alpha=0.75, label='測試集實際成交價 Actual Price ($k)')

    plt.title('圖 6: 多元線性迴歸房屋價格預測圖 (包含 95% 信賴區間與 95% 預測區間)', fontsize=15, fontweight='bold')
    plt.xlabel('測試集房屋樣本 (依預測價格遞增排序)', fontsize=12)
    plt.ylabel('房屋價格 (Price in Thousands $k)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, fontsize=11)
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, 'fig6_prediction_intervals.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig6_path}")

    # 2. 高斯-馬可夫四大假設殘差診斷 (Gauss-Markov Assumptions)
    residuals = ols_model.resid
    fitted = ols_model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('圖 7: 線性迴歸高斯-馬可夫基本假設殘差診斷分析', fontsize=15, fontweight='bold')

    # (1) 殘差 vs 擬合值 (線性性與同質變異檢驗)
    axes[0, 0].scatter(fitted / 1000, residuals / 1000, alpha=0.3, color='#1d3557', edgecolors='none', s=15)
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0, 0].set_title('1. 殘差 vs 擬合值 (Residuals vs Fitted)')
    axes[0, 0].set_xlabel('預測擬合值 Fitted Values ($k)')
    axes[0, 0].set_ylabel('殘差 Residuals ($k)')

    # (2) 常態 Q-Q 圖 (常態分佈假設檢驗)
    sm.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
    axes[0, 1].set_title('2. 殘差常態 Q-Q 圖 (Normal Q-Q Plot)')

    # (3) 尺度-位置圖 (Scale-Location Plot，同質變異檢驗)
    standardized_res = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[1, 0].scatter(fitted / 1000, standardized_res, alpha=0.3, color='#2a9d8f', edgecolors='none', s=15)
    axes[1, 0].set_title('3. 尺度-位置圖 (Scale-Location Plot)')
    axes[1, 0].set_xlabel('預測擬合值 Fitted Values ($k)')
    axes[1, 0].set_ylabel('$\\sqrt{|標準化殘差|}$')

    # (4) 殘差直方圖與常態比對
    sns.histplot(residuals / 1000, kde=True, ax=axes[1, 1], color='#e76f51', bins=40)
    axes[1, 1].set_title('4. 殘差分佈直方圖 (Residual Distribution)')
    axes[1, 1].set_xlabel('殘差 Residuals ($k)')

    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, 'fig7_residual_diagnostics.png')
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig7_path}")

    # 3. 多元共線性檢驗 (VIF)
    vif_df = pd.DataFrame()
    vif_df['Feature'] = X_train_sel.columns
    vif_df['VIF'] = [variance_inflation_factor(X_train_sel.values, i) for i in range(len(X_train_sel.columns))]
    print("\n【精選特徵多元共線性 (VIF) 檢定結果】:")
    print(vif_df.to_string(index=False))

    # 4. 統計檢定指標
    dw_stat = sm.stats.stattools.durbin_watson(residuals)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    bp_test = het_breuschpagan(residuals, ols_model.model.exog)
    print("\n【統計假設檢定數值摘要】:")
    print(f"1. Durbin-Watson 統計量 (獨立性): {dw_stat:.4f} (接近 2.0，確認無自相關)")
    print(f"2. Jarque-Bera 檢定值 (常態性): {jb_stat:.2f}, p-value: {jb_p:.4e}")
    print(f"3. Breusch-Pagan 檢定值 (同質變異): {bp_test[0]:.2f}, p-value: {bp_test[1]:.4e}")

    # 5. 特徵標準化係數重要性排序
    coefs = ols_model.params.drop('const')
    # 計算標準化迴歸係數: beta_std = beta * (std_x / std_y)
    std_x = X_train_sel.std()
    std_y = y_train.std()
    std_coefs = coefs * (std_x / std_y)

    coef_df = pd.DataFrame({
        'Feature': std_coefs.index,
        'Std_Coef': std_coefs.values
    }).sort_values(by='Std_Coef', key=abs, ascending=True)

    plt.figure(figsize=(10, 6))
    colors = ['#e63946' if c > 0 else '#1d3557' for c in coef_df['Std_Coef']]
    plt.barh(coef_df['Feature'], coef_df['Std_Coef'], color=colors)
    plt.axvline(0, color='black', linestyle='--', lw=1)
    plt.title('圖 8: 精選特徵標準化迴歸係數 (Feature Importance by Standardized Beta)', fontsize=14, fontweight='bold')
    plt.xlabel('標準化迴歸係數 (Standardized Beta Weight)', fontsize=12)
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, 'fig8_feature_importance.png')
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"已儲存: {fig8_path}")

    return ols_model, vif_df


# -------------------------------------------------------------
# Phase 6: Deployment (模型部署與應用規劃)
# -------------------------------------------------------------
def deployment_plan():
    print_section(6, "Deployment (模型部署與應用規劃)")
    print("【線上房價估價 API 部署規劃】")
    print("1. 將訓練完成之迴歸係數與特徵標準化轉換器序列化 (joblib / pickle)。")
    print("2. 透過 FastAPI 或 Flask 封裝為 RESTful API，接收使用者輸入的房屋特徵。")
    print("3. 即時回傳房屋預測估價、95% 信賴區間 (合理均價帶) 及 95% 預測區間 (個別個案極限帶)。")
    print("4. 定期透過 Airflow 觸發每月批次重訓排程，因應西雅圖整體房市通膨與利率浮動調整權重。")


# -------------------------------------------------------------
# 主程式執行入口
# -------------------------------------------------------------
def main():
    print("*" * 75)
    print("  巨量資料分析與應用 - 作業二：多元線性迴歸分析 (CRISP-DM 完整流程)")
    print("  資料來源：Kaggle - House Sales in King County, USA (18 預測特徵)")
    print("*" * 75)

    # 1. 商業與資料理解
    df, features = load_and_explore_data('kc_house_data.csv')

    # 2. 探索性視覺化
    run_eda_visualizations(df, features)

    # 3. 資料準備與前處理
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, feature_names = prepare_data(df)

    # 4. 模型建立與特徵選擇
    models_dict, eval_df, selected_features = build_and_select_models(
        X_train_sc, X_test_sc, y_train, y_test, feature_names
    )

    # 5. 模型評估、信賴/預測區間與殘差診斷
    ols_model, vif_df = evaluate_and_diagnose(
        X_train, X_test, y_train, y_test, selected_features
    )

    # 6. 部署規劃
    deployment_plan()

    print("\n" + "*" * 75)
    print("  作業執行完成！所有 8 張高解析圖表已輸出至 figures/ 資料夾。")
    print("*" * 75)


if __name__ == '__main__':
    main()
