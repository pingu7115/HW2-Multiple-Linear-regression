# -*- coding: utf-8 -*-
"""
Generates all 9 academic-grade visualization figures for Problem 2: WHO Life Expectancy Dataset.
Features: 20 features (> 13 features requirement).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 150

os.makedirs('figures', exist_ok=True)

# 1. Load and prepare dataset
df = pd.read_csv('life_expectancy.csv')
df.columns = df.columns.str.strip()
df.rename(columns={'thinness  1-19 years': 'thinness 1-19 years'}, inplace=True)

# Drop missing target
df = df.dropna(subset=['Life expectancy']).copy()

# Feature transformation
df['Status_Developed'] = (df['Status'] == 'Developed').astype(int)

candidate_features = [
    'Year', 'Status_Developed', 'Adult Mortality', 'infant deaths', 'Alcohol',
    'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 'under-five deaths',
    'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP',
    'Population', 'thinness 1-19 years', 'thinness 5-9 years',
    'Income composition of resources', 'Schooling'
]

print(f"Number of predictive features: {len(candidate_features)} (Requirement > 13: {len(candidate_features) > 13})")

# Median imputation for predictor features
for col in candidate_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

X = df[candidate_features].copy()
y = df['Life expectancy'].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression: best correlated feature (Income composition of resources)
corr = df[candidate_features].apply(lambda c: c.corr(df['Life expectancy'])).abs().sort_values(ascending=False)
best_feature = 'Income composition of resources'

slr = LinearRegression()
slr.fit(X_train[[best_feature]], y_train)
y_pred_slr = slr.predict(X_test[[best_feature]])
slr_r2 = r2_score(y_test, y_pred_slr)
slr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_slr))
slr_mae = mean_absolute_error(y_test, y_pred_slr)
slr_slope = slr.coef_[0]
slr_intercept = slr.intercept_

# Full Multiple Linear Regression (OLS)
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
mlr_r2 = r2_score(y_test, y_pred_mlr)
mlr_adj_r2 = 1 - (1 - mlr_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
mlr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
mlr_mae = mean_absolute_error(y_test, y_pred_mlr)

# RFE: 10 features
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe.fit(X_train, y_train)
rfe_features = [f for f, s in zip(candidate_features, rfe.support_) if s]
mlr_rfe = LinearRegression()
mlr_rfe.fit(X_train[rfe_features], y_train)
y_pred_rfe = mlr_rfe.predict(X_test[rfe_features])
rfe_r2 = r2_score(y_test, y_pred_rfe)
rfe_adj_r2 = 1 - (1 - rfe_r2) * (len(y_test) - 1) / (len(y_test) - len(rfe_features) - 1)
rfe_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rfe))
rfe_mae = mean_absolute_error(y_test, y_pred_rfe)

# Regularized Models (Scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_adj_r2 = 1 - (1 - ridge_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_adj_r2 = 1 - (1 - lasso_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
y_pred_elastic = elastic.predict(X_test_scaled)
elastic_r2 = r2_score(y_test, y_pred_elastic)
elastic_adj_r2 = 1 - (1 - elastic_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
elastic_rmse = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
elastic_mae = mean_absolute_error(y_test, y_pred_elastic)

# Residual Diagnostics
residuals = y_test - y_pred_mlr
durbin_watson = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
jb_stat, jb_p = stats.jarque_bera(residuals)

print(f"SLR (Income): R2={slr_r2:.4f}, RMSE={slr_rmse:.4f}, MAE={slr_mae:.4f}")
print(f"Full OLS (20 features): R2={mlr_r2:.4f}, Adj_R2={mlr_adj_r2:.4f}, RMSE={mlr_rmse:.4f}, MAE={mlr_mae:.4f}")
print(f"RFE (10 features): R2={rfe_r2:.4f}, Adj_R2={rfe_adj_r2:.4f}, RMSE={rfe_rmse:.4f}, MAE={rfe_mae:.4f}")
print(f"Ridge: R2={ridge_r2:.4f}, RMSE={ridge_rmse:.4f}, MAE={ridge_mae:.4f}")
print(f"Lasso: R2={lasso_r2:.4f}, RMSE={lasso_rmse:.4f}, MAE={lasso_mae:.4f}")
print(f"ElasticNet: R2={elastic_r2:.4f}, RMSE={elastic_rmse:.4f}, MAE={elastic_mae:.4f}")
print(f"Durbin-Watson: {durbin_watson:.4f}, JB Stat: {jb_stat:.4f}, p: {jb_p:.4e}")
print(f"RFE Selected Features: {rfe_features}")

# -------------------------------------------------------------
# FIGURE 0: Simple Linear Regression with 95% CI & 95% PI
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
x_vals = np.linspace(X[best_feature].min(), X[best_feature].max(), 300)
y_fitted = slr_intercept + slr_slope * x_vals

# Analytical SE for Simple LR CI & PI
n_slr = len(X_train)
x_train_mean = X_train[best_feature].mean()
ss_x = np.sum((X_train[best_feature] - x_train_mean) ** 2)
s_slr = np.sqrt(np.sum((y_train - slr.predict(X_train[[best_feature]])) ** 2) / (n_slr - 2))
t_val = stats.t.ppf(0.975, df=n_slr - 2)

ci_se = s_slr * np.sqrt(1.0 / n_slr + (x_vals - x_train_mean) ** 2 / ss_x)
pi_se = s_slr * np.sqrt(1.0 + 1.0 / n_slr + (x_vals - x_train_mean) ** 2 / ss_x)

plt.scatter(X_test[best_feature], y_test, alpha=0.35, color='#2563eb', edgecolors='none', s=28, label='Test Data Points (n=586)')
plt.plot(x_vals, y_fitted, color='#dc2626', linewidth=2.5, label=f'OLS Line: y = {slr_intercept:.2f} + {slr_slope:.2f}x')
plt.fill_between(x_vals, y_fitted - t_val * ci_se, y_fitted + t_val * ci_se, color='#3b82f6', alpha=0.3, label='95% Confidence Interval (CI)')
plt.plot(x_vals, y_fitted - t_val * pi_se, color='#d97706', linestyle='--', linewidth=1.5, label='95% Prediction Interval (PI) Lower')
plt.plot(x_vals, y_fitted + t_val * pi_se, color='#d97706', linestyle='--', linewidth=1.5, label='95% Prediction Interval (PI) Upper')

plt.title('Problem 2: Simple Linear Regression (Income Composition of Resources vs Life Expectancy)', fontsize=13, fontweight='bold', pad=14)
plt.xlabel('Income Composition of Resources (Human Development Index Component, 0 - 1)', fontsize=11, labelpad=8)
plt.ylabel('Life Expectancy (Years)', fontsize=11, labelpad=8)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/p2_fig0_simple_linear_regression.png', dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 1: EDA Distributions
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
dist_features = ['Life expectancy', 'Income composition of resources', 'Schooling', 'Adult Mortality', 'HIV/AIDS', 'BMI']
colors = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2']

for idx, (col, color) in enumerate(zip(dist_features, colors)):
    ax = axes[idx // 3, idx % 3]
    sns.histplot(df[col], kde=True, ax=ax, color=color, alpha=0.6, edgecolor='white')
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='#1e293b', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#e11d48', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax.set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Problem 2: Exploratory Data Analysis - Feature Distributions (WHO Life Expectancy)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/p2_fig1_eda_distributions.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 2: Correlation Heatmap
# -------------------------------------------------------------
plt.figure(figsize=(15, 12))
corr_matrix = df[candidate_features + ['Life expectancy']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt='.2f', annot_kws={"size": 7.5})
plt.title('Problem 2: Pearson Correlation Matrix Heatmap (20 Predictors + Life Expectancy)', fontsize=14, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig('figures/p2_fig2_eda_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 3: Key Target Relationships
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
rel_features = [
    ('Income composition of resources', '#2563eb'),
    ('Adult Mortality', '#dc2626'),
    ('Schooling', '#059669'),
    ('HIV/AIDS', '#7c3aed')
]

for idx, (f_col, clr) in enumerate(rel_features):
    ax = axes[idx // 2, idx % 2]
    # Scatter colored by Status
    scatter = ax.scatter(df[f_col], df['Life expectancy'], c=df['Status_Developed'], cmap='coolwarm', alpha=0.5, s=26, edgecolors='none')
    # Trendline
    sns.regplot(data=df, x=f_col, y='Life expectancy', scatter=False, ax=ax, color='#1e293b', line_kws={'linewidth': 2})
    ax.set_title(f'Life Expectancy vs {f_col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(f_col, fontsize=10.5)
    ax.set_ylabel('Life Expectancy (Years)', fontsize=10.5)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Problem 2: Life Expectancy vs Key Socio-Economic & Health Indicators', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/p2_fig3_eda_target_relationships.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 4: Feature Selection Comparison (Full 20 vs RFE 10)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_names = ['R-Squared (R²)', 'RMSE (Years)', 'MAE (Years)']
full_vals = [mlr_r2, mlr_rmse, mlr_mae]
rfe_vals = [rfe_r2, rfe_rmse, rfe_mae]

for i, (m_name, f_val, r_val) in enumerate(zip(metrics_names, full_vals, rfe_vals)):
    ax = axes[i]
    bars = ax.bar(['Full Model\n(20 Features)', 'RFE Selected\n(10 Features)'], [f_val, r_val], color=['#2563eb', '#10b981'], width=0.55, edgecolor='black', linewidth=0.8)
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title(m_name, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(f_val, r_val) * 1.25)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.suptitle('Problem 2: Feature Selection Performance Comparison (Full 20 Features vs RFE 10 Features)', fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('figures/p2_fig4_feature_selection_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 5: Model Comparison Across 6 Algorithms
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
model_names = ['Simple LR\n(1 Feat)', 'Full OLS\n(20 Feat)', 'RFE OLS\n(10 Feat)', 'Ridge (L2)\n(α=1.0)', 'Lasso (L1)\n(α=0.01)', 'ElasticNet\n(α=0.01)']
r2_scores = [slr_r2, mlr_r2, rfe_r2, ridge_r2, lasso_r2, elastic_r2]
rmse_scores = [slr_rmse, mlr_rmse, rfe_rmse, ridge_rmse, lasso_rmse, elastic_rmse]
colors_bar = ['#94a3b8', '#2563eb', '#059669', '#7c3aed', '#d97706', '#0891b2']

# Bar 1: R2
bars1 = ax1.bar(model_names, r2_scores, color=colors_bar, width=0.6, edgecolor='black', linewidth=0.8)
for b in bars1:
    h = b.get_height()
    ax1.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax1.set_title('Test Set Explanatory Power (R²)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_ylim(0, 1.0)
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

# Bar 2: RMSE
bars2 = ax2.bar(model_names, rmse_scores, color=colors_bar, width=0.6, edgecolor='black', linewidth=0.8)
for b in bars2:
    h = b.get_height()
    ax2.annotate(f'{h:.2f} yrs', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax2.set_title('Test Set Prediction Error (RMSE, Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE (Years)', fontsize=11)
ax2.set_ylim(0, max(rmse_scores) * 1.2)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.suptitle('Problem 2: Model Evaluation & Performance Benchmark Across 6 Regression Models', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/p2_fig5_model_comparison_metrics.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 6: Actual vs Predicted with 95% Prediction Interval
# -------------------------------------------------------------
# Calculate OLS Matrix Standard Error for Multiple Regression PI
X_train_mat = np.column_stack([np.ones(len(X_train)), X_train.values])
X_test_mat = np.column_stack([np.ones(len(X_test)), X_test.values])
n_train_mat = len(X_train)
p_mat = X_train.shape[1]
sigma_hat = mlr_rmse  # sample residual std
xtx_inv = np.linalg.pinv(X_train_mat.T @ X_train_mat)

leverage_test = np.sum((X_test_mat @ xtx_inv) * X_test_mat, axis=1)
pi_margin = 1.96 * sigma_hat * np.sqrt(1.0 + leverage_test)

pi_lower = y_pred_mlr - pi_margin
pi_upper = y_pred_mlr + pi_margin
coverage = np.mean((y_test.values >= pi_lower) & (y_test.values <= pi_upper)) * 100.0

plt.figure(figsize=(10, 7))
# Sort for clean visualization
sort_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sort_idx]
y_pred_sorted = y_pred_mlr[sort_idx]
pi_lower_sorted = pi_lower[sort_idx]
pi_upper_sorted = pi_upper[sort_idx]

# Subsample 150 points for visual clarity
sub_idx = np.linspace(0, len(sort_idx) - 1, 140, dtype=int)

plt.scatter(range(len(sub_idx)), y_test_sorted[sub_idx], color='#1e293b', s=26, label='Actual Life Expectancy (Years)', zorder=4)
plt.plot(range(len(sub_idx)), y_pred_sorted[sub_idx], color='#2563eb', linewidth=2, label=f'OLS Predicted Value (R² = {mlr_r2:.4f})', zorder=3)
plt.fill_between(range(len(sub_idx)), pi_lower_sorted[sub_idx], pi_upper_sorted[sub_idx], color='#93c5fd', alpha=0.45, label=f'95% Prediction Interval (Actual Coverage: {coverage:.1f}%)', zorder=2)

plt.title('Problem 2: Multiple Linear Regression (Full 20 Features) 95% Prediction Interval', fontsize=13, fontweight='bold', pad=14)
plt.xlabel('Test Samples (Sorted by Observed Life Expectancy)', fontsize=11, labelpad=8)
plt.ylabel('Life Expectancy (Years)', fontsize=11, labelpad=8)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/p2_fig6_prediction_intervals.png', dpi=200)
plt.close()

# -------------------------------------------------------------
# FIGURE 7: 4-Panel Residual Diagnostics (Gauss-Markov)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Residuals vs Fitted
ax1 = axes[0, 0]
ax1.scatter(y_pred_mlr, residuals, alpha=0.4, color='#2563eb', edgecolors='none', s=28)
ax1.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
sns.regplot(x=y_pred_mlr, y=residuals, scatter=False, ax=ax1, lowess=True, color='#f59e0b', line_kws={'linewidth': 2})
ax1.set_title('Residuals vs Fitted (Homoscedasticity)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Fitted Values (Years)', fontsize=10.5)
ax1.set_ylabel('Residuals (e_i)', fontsize=10.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Normal Q-Q Plot
ax2 = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.get_lines()[0].set_markerfacecolor('#059669')
ax2.get_lines()[0].set_markeredgecolor('none')
ax2.get_lines()[0].set_alpha(0.5)
ax2.get_lines()[1].set_color('#dc2626')
ax2.get_lines()[1].set_linewidth(1.8)
ax2.set_title(f'Normal Q-Q Plot (Jarque-Bera p = {jb_p:.2e})', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel 3: Scale-Location Plot
ax3 = axes[1, 0]
std_residuals = residuals / np.std(residuals)
sqrt_abs_res = np.sqrt(np.abs(std_residuals))
ax3.scatter(y_pred_mlr, sqrt_abs_res, alpha=0.4, color='#7c3aed', edgecolors='none', s=28)
sns.regplot(x=y_pred_mlr, y=sqrt_abs_res, scatter=False, ax=ax3, lowess=True, color='#dc2626', line_kws={'linewidth': 2})
ax3.set_title('Scale-Location (Spread-Location)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Fitted Values (Years)', fontsize=10.5)
ax3.set_ylabel('√|Standardized Residuals|', fontsize=10.5)
ax3.grid(True, linestyle='--', alpha=0.5)

# Panel 4: Histogram of Residuals with Normal Curve
ax4 = axes[1, 1]
sns.histplot(residuals, kde=True, ax=ax4, color='#2563eb', stat="density", alpha=0.5, edgecolor='white')
x_norm = np.linspace(residuals.min(), residuals.max(), 200)
ax4.plot(x_norm, stats.norm.pdf(x_norm, loc=np.mean(residuals), scale=np.std(residuals)), color='#dc2626', linewidth=2, label='Theoretical Normal')
ax4.set_title(f'Residual Distribution (Mean={np.mean(residuals):.3f}, DW d={durbin_watson:.3f})', fontsize=12, fontweight='bold')
ax4.set_xlabel('Residuals (e_i)', fontsize=10.5)
ax4.set_ylabel('Density', fontsize=10.5)
ax4.legend(fontsize=9.5)
ax4.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Problem 2: Gauss-Markov Residual Diagnostics (Multiple Linear Regression)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/p2_fig7_residual_diagnostics.png', dpi=200, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# FIGURE 8: Standardized Coefficient Importance Ranking
# -------------------------------------------------------------
# Standardized coefficients
std_scaler_x = StandardScaler()
X_std = std_scaler_x.fit_transform(X)
std_model = LinearRegression().fit(X_std, y)
coef_df = pd.DataFrame({
    'Feature': candidate_features,
    'Std_Coef': std_model.coef_
}).sort_values(by='Std_Coef', key=abs, ascending=True)

plt.figure(figsize=(10, 8))
colors_coef = ['#dc2626' if c < 0 else '#2563eb' for c in coef_df['Std_Coef']]
bars = plt.barh(coef_df['Feature'], coef_df['Std_Coef'], color=colors_coef, edgecolor='black', linewidth=0.6, height=0.65)
for b in bars:
    w = b.get_width()
    offset = 0.08 if w >= 0 else -0.08
    ha = 'left' if w >= 0 else 'right'
    plt.annotate(f'{w:.3f}', xy=(w + offset, b.get_y() + b.get_height() / 2), ha=ha, va='center', fontsize=9, fontweight='bold')

plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.title('Problem 2: Standardized Feature Importance (Beta Coefficients)', fontsize=13, fontweight='bold', pad=14)
plt.xlabel('Standardized Regression Coefficient (β*)', fontsize=11, labelpad=8)
plt.xlim(coef_df['Std_Coef'].min() * 1.25, coef_df['Std_Coef'].max() * 1.25)
plt.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
plt.savefig('figures/p2_fig8_feature_importance.png', dpi=200)
plt.close()

print("All 9 Problem 2 figures successfully created in figures/ directory!")
