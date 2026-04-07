# ====================================================
# 门店活动效果全维度分析总脚本（内置完整规模+类型分层）
# 功能：原始数据→自动分层→输出明细表→数据匹配→全维度统计分析
# 修改版：补充EDA、修正层级原则、增强诊断、汇总报告、logging
# 新增：多因素ANOVA方差齐性检验（自动选择普通/Welch ANOVA）
#       回归残差方差齐性检验（Breusch-Pagan）
# ====================================================
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, levene, shapiro
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import os
import pingouin as pg
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import logging

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("new.new.total.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ========== 统一配置区域（一键修改所有路径/参数） ==========
# 活动数据
INPUT_FILE = "./guoba.sale.xlsx"
FILE_TYPE = "excel"
SHEET_NAME = "Sheet1"
COL_STORE = "门店ID"
COL_REGION = "区域名称"
COL_PRE = "活动前"
COL_POST = "活动中"
COL_DAYS = "天数"

# 原始未分层数据
STORE_BASIC_RAW = "./门店基础信息.xlsx"
STORE_SALE_RAW = "./门店数据01.csv"
COL_OPEN_YEAR = "开业日期"
COL_ACTUAL = "actual"

# 输出
OUTPUT_DIR = "output"
CONVERT_TO_DAILY = True
SMALL_SAMPLE_THRESHOLD = 30

# 回归基准组（人工指定）—— 实际代码中未使用，保留备用
BASE_REGION = "环粤区域"
BASE_SCALE = 1
BASE_TYPE = 1

# 分层通用参数
BASE_DATE = datetime(2025, 7, 1)
SCALE_TIER_3 = True
TYPE_TIER_3 = True

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. 活动数据加载（修复Excel引擎） ==========
def load_and_preprocess(file_path, file_type, sheet_name=None):
    if file_type == "excel":
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    elif file_type == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("仅支持excel/csv")
    
    required_cols = [COL_STORE, COL_REGION, COL_PRE, COL_POST]
    if CONVERT_TO_DAILY:
        required_cols.append(COL_DAYS)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列：{missing}")
    
    if CONVERT_TO_DAILY:
        days = df[COL_DAYS]
        df["pre_daily"] = df[COL_PRE] / days
        df["post_daily"] = df[COL_POST] / days
    else:
        df["pre_daily"] = df[COL_PRE]
        df["post_daily"] = df[COL_POST]
    
    df["diff"] = df["post_daily"] - df["pre_daily"]
    df = df.drop_duplicates(subset=[COL_STORE], keep="first")
    df = df.dropna(subset=["diff", COL_REGION]).reset_index(drop=True)
    return df

# ========== 3. 【整合】规模分层全套功能 ==========
def auto_scale_stratify():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    INPUT_FILE = STORE_SALE_RAW
    Q1_3TIER, Q2_3TIER = 0.2, 0.8
    Q_2TIER = 0.7
    TIER_LABELS_3 = {1: "小店", 2: "中店", 3: "大店"}
    TIER_LABELS_2 = {0: "小店", 1: "大店"}

    # 加载
    if INPUT_FILE.endswith(".xlsx"):
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    elif INPUT_FILE.endswith(".csv"):
        df = pd.read_csv(INPUT_FILE)
    else:
        raise ValueError()

    store_agg = df.groupby("门店ID").agg(
        累计实收=("actual", "sum"), 区域=("region", "first")
    ).reset_index()
    store_agg = store_agg[store_agg["累计实收"] > 0].copy()

    # 分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    sns.histplot(store_agg["累计实收"], kde=True, ax=ax1, color="#2E86AB")
    sns.boxplot(y=store_agg["累计实收"], ax=ax2, color="#A23B72")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/门店累计实收分布.png", dpi=300)
    plt.close()

    # 分层
    if SCALE_TIER_3:
        q_low = np.percentile(store_agg["累计实收"], Q1_3TIER*100)
        q_high = np.percentile(store_agg["累计实收"], Q2_3TIER*100)
        conditions = [
            store_agg["累计实收"] <= q_low,
            (store_agg["累计实收"]>q_low)&(store_agg["累计实收"]<q_high),
            store_agg["累计实收"] >= q_high
        ]
        store_agg["规模编码"] = np.select(conditions, [1,2,3])
        store_agg["规模等级"] = store_agg["规模编码"].map(TIER_LABELS_3)
    else:
        q = np.percentile(store_agg["累计实收"], Q_2TIER*100)
        conditions = [store_agg["累计实收"]<=q, store_agg["累计实收"]>q]
        store_agg["规模编码"] = np.select(conditions, [0,1])
        store_agg["规模等级"] = store_agg["规模编码"].map(TIER_LABELS_2)

    # 验证
    groups = []
    tiers = [1,2,3] if SCALE_TIER_3 else [0,1]
    for t in tiers:
        groups.append(store_agg[store_agg["规模编码"]==t]["累计实收"].values)
    f_stat, p_val = f_oneway(*groups)

    plt.figure(figsize=(8,5))
    order = ["小店","中店","大店"] if SCALE_TIER_3 else ["小店","大店"]
    sns.boxplot(x="规模等级", y="累计实收", data=store_agg, order=order, palette="Set2")
    plt.title("规模分层验证")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/门店规模分层验证.png", dpi=300)
    plt.close()

    # 导出
    detail = store_agg[["门店ID","区域","累计实收","规模编码","规模等级"]]
    detail.rename(columns={"门店ID": COL_STORE}, inplace=True)
    detail.to_excel(f"{OUTPUT_DIR}/门店规模分层明细表.xlsx", index=False, engine='openpyxl')
    logging.info("规模分层完成")
    return detail

# ========== 4. 【整合】开业类型分层全套功能 ==========
def auto_type_stratify():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TIER3_THRESHOLD_1 = 1
    TIER3_THRESHOLD_2 = 3
    TIER2_THRESHOLD = 1
    TIER_LABELS_3 = {1: "新店", 2: "次新店", 3: "成熟老店"}
    TIER_LABELS_2 = {0: "新店", 1: "老店"}

    df = pd.read_excel(STORE_BASIC_RAW, engine='openpyxl')
    df["开业日期"] = pd.to_datetime(df["开业日期"], errors="coerce")
    df = df.dropna(subset=["开业日期"]).copy()
    df["开业时长_年"] = (BASE_DATE - df["开业日期"]).dt.days / 365.25
    df = df[df["开业时长_年"] >= 0].copy()

    # 分布图
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
    sns.histplot(df["开业时长_年"], kde=True, ax=ax1, color="#2E86AB")
    sns.boxplot(y=df["开业时长_年"], ax=ax2, color="#A23B72")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/门店开业时长分布.png", dpi=300)
    plt.close()

    # 分层
    if TYPE_TIER_3:
        conditions = [
            df["开业时长_年"] <= TIER3_THRESHOLD_1,
            (df["开业时长_年"]>TIER3_THRESHOLD_1)&(df["开业时长_年"]<=TIER3_THRESHOLD_2),
            df["开业时长_年"] > TIER3_THRESHOLD_2
        ]
        df["门店类型编码"] = np.select(conditions, [1,2,3])
        df["门店类型"] = df["门店类型编码"].map(TIER_LABELS_3)
    else:
        conditions = [df["开业时长_年"]<=TIER2_THRESHOLD, df["开业时长_年"]>TIER2_THRESHOLD]
        df["门店类型编码"] = np.select(conditions, [0,1])
        df["门店类型"] = df["门店类型编码"].map(TIER_LABELS_2)

    # 验证
    groups = []
    tiers = [1,2,3] if TYPE_TIER_3 else [0,1]
    for t in tiers:
        groups.append(df[df["门店类型编码"]==t]["开业时长_年"].values)
    f_oneway(*groups)

    plt.figure(figsize=(8,5))
    order = ["新店","次新店","成熟老店"] if TYPE_TIER_3 else ["新店","老店"]
    sns.boxplot(x="门店类型", y="开业时长_年", data=df, order=order, palette="Set2")
    plt.title("类型分层验证")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/门店类型分层验证.png", dpi=300)
    plt.close()

    # 导出
    df.rename(columns={"店号": COL_STORE}, inplace=True)
    detail = df[[COL_STORE, "开业日期", "开业时长_年", "门店类型编码", "门店类型"]]
    detail.to_excel(f"{OUTPUT_DIR}/门店类型分层明细表.xlsx", index=False, engine='openpyxl')
    logging.info("类型分层完成")
    return detail

# ========== 5. 自动合并匹配（自动删除无效门店） ==========
def merge_all_data(activity_df):
    scale_df = auto_scale_stratify()
    type_df = auto_type_stratify()
    basic_df = pd.merge(scale_df, type_df, on=COL_STORE, how="inner")
    merged_df = pd.merge(activity_df, basic_df, on=COL_STORE, how="inner")

    logging.info(f"匹配结果：原始活动门店 {len(activity_df)}，有效匹配 {len(merged_df)}，自动删除 {len(activity_df)-len(merged_df)}")
    return merged_df

# ========== 6. 统计分析辅助函数 ==========
def normality_test_small_sample(df):
    logging.info("小样本正态性检验")
    regions = df[COL_REGION].value_counts()
    small = regions[regions<30].index
    for r in small:
        s = df[df[COL_REGION]==r]["diff"]
        p = stats.shapiro(s)[1]
        logging.info(f"{r}: p={p:.4f}, {'正态' if p>0.05 else '非正态'}")

def robust_log_transform(df):
    df["diff_log"] = np.sign(df["diff"]) * np.log1p(np.abs(df["diff"]))
    return df

def compute_region_stats(g):
    n = len(g)
    if n<2: return [np.nan]*7
    t, p = stats.ttest_rel(g["post_daily"], g["pre_daily"])
    return [n, g["pre_daily"].mean(), g["post_daily"].mean(),
            g["diff"].mean(), g["diff"].std(), t, p]

def compute_all_and_group(df):
    cols = ["样本量","活动前","活动后","日均提升","标准差","t值","p值"]
    res = df.groupby(COL_REGION).apply(compute_region_stats).apply(pd.Series)
    res.columns = cols
    res.to_excel(f"{OUTPUT_DIR}/区域统计汇总.xlsx", engine='openpyxl')
    return res

def visualize(df):
    long = df.melt(id_vars=[COL_STORE, COL_REGION], value_vars=["pre_daily","post_daily"])
    long.columns = [COL_STORE, COL_REGION, "阶段", "日均销量"]
    long["阶段"] = long["阶段"].map({"pre_daily":"活动前","post_daily":"活动后"})
    plt.figure(figsize=(12,6))
    sns.boxplot(x=COL_REGION, y="日均销量", hue="阶段", data=long)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/活动前后对比.png", dpi=300)
    plt.close()

# ========== 7. 多因素ANOVA（增强版：自动检验方差齐性并选择方法） ==========
def three_way_anova_main_effect(df):
    logging.info("三因素主效应ANOVA（含方差齐性检验）")
    
    # 将分类变量转为category类型（确保ANOVA正确分组）
    df[COL_REGION] = df[COL_REGION].astype('category')
    df["规模编码"] = df["规模编码"].astype('category')
    df["门店类型编码"] = df["门店类型编码"].astype('category')
    
    # 检查分组样本量
    for col in [COL_REGION, "规模编码", "门店类型编码"]:
        cnt = df.groupby(col)['diff_log'].count()
        logging.info(f"{col} 样本量：{cnt.to_dict()}")
    
    # 1. 方差齐性检验（Levene检验，按三个因素的交互分组）
    # 创建交互分组列
    df["interaction_group"] = (df[COL_REGION].astype(str) + "_" + 
                                df["规模编码"].astype(str) + "_" + 
                                df["门店类型编码"].astype(str))
    groups = [group["diff_log"].values for name, group in df.groupby("interaction_group") if len(group) >= 2]
    if len(groups) >= 2:
        stat_levene, p_levene = levene(*groups)
        logging.info(f"Levene方差齐性检验（按交互分组）: p={p_levene:.4f}")
    else:
        p_levene = 0.0
        logging.warning("交互分组数量不足，无法进行Levene检验，默认方差不齐")
    
    # 2. 根据方差齐性选择ANOVA方法
    if p_levene < 0.05:
        logging.warning("方差不齐（p<0.05），使用Welch ANOVA（不假设方差齐性）")
        # Welch ANOVA（pingouin的welch_anova）
        welch_result = pg.welch_anova(dv="diff_log", between=[COL_REGION, "规模编码", "门店类型编码"], data=df)
        logging.info("Welch ANOVA结果：")
        logging.info(welch_result[["Source", "F", "p-unc"]].round(4))
        # 保存结果（Welch ANOVA无偏η²，只保存F和p）
        welch_result.to_excel(f"{OUTPUT_DIR}/三因素Welch_ANOVA结果.xlsx", index=False, engine='openpyxl')
        # 为了保持输出格式一致，创建一个简化的DataFrame用于后续汇总
        main = welch_result.copy()
        main["partial_eta2"] = np.nan  # Welch ANOVA不提供偏η²
    else:
        logging.info("方差齐性通过（p>=0.05），使用普通ANOVA")
        a = pg.anova(dv="diff_log", between=[COL_REGION, "规模编码", "门店类型编码"], data=df, detailed=True)
        ss_residual = a[a["Source"] == "Residual"]["SS"].values[0]
        main = a[a["Source"].isin([COL_REGION, "规模编码", "门店类型编码"])].copy()
        main["partial_eta2"] = main["SS"] / (main["SS"] + ss_residual)
        logging.info("三因素ANOVA主效应结果（含偏η²）：")
        logging.info(main[["Source", "F", "p_unc", "partial_eta2"]].round(4))
        main.to_excel(f"{OUTPUT_DIR}/三因素ANOVA结果.xlsx", engine='openpyxl')
    
    # 清理临时列
    df.drop("interaction_group", axis=1, inplace=True)
    
    return main

# ========== 8. EDA（探索性数据分析） ==========
def exploratory_analysis(df):
    logging.info("开始EDA（探索性数据分析）")
    # 增量分布
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(df["diff"], kde=True)
    plt.title("增量（diff）整体分布")
    plt.subplot(1,2,2)
    sns.boxplot(x=COL_REGION, y="diff", data=df)
    plt.xticks(rotation=45)
    plt.title("各区域增量分布")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/EDA_增量分布.png", dpi=300)
    plt.close()
    
    # 活动前销量 vs 增量散点图
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="pre_daily", y="diff", hue=COL_REGION, data=df)
    plt.title("活动前日均销量 vs 增量")
    plt.savefig(f"{OUTPUT_DIR}/EDA_活动前vs增量.png", dpi=300)
    plt.close()
    
    # 相关性热图
    num_cols = ["pre_daily", "post_daily", "diff", "规模编码", "门店类型编码"]
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("数值变量相关性热图")
    plt.savefig(f"{OUTPUT_DIR}/EDA_相关性热图.png", dpi=300)
    plt.close()
    
    # 方差齐性检验（Levene）
    groups = [group["diff"].values for name, group in df.groupby(COL_REGION)]
    stat, p_levene = levene(*groups)
    logging.info(f"Levene方差齐性检验: p={p_levene:.4f}，{'方差齐' if p_levene>0.05 else '方差不齐'}")

# ========== 9. 多元线性回归（增强诊断：增加Breusch-Pagan方差齐性检验） ==========
def multiple_linear_regression_main_effect(df, sig_level=0.05):
    logging.info("多元线性回归分析开始")
    # 预处理
    df_reg = df[["diff_log", COL_REGION, "规模编码", "门店类型编码"]].dropna().copy()
    df_reg["diff_log"] = pd.to_numeric(df_reg["diff_log"], errors="coerce")
    df_reg["规模编码"] = pd.to_numeric(df_reg["规模编码"], errors="coerce")
    df_reg["门店类型编码"] = pd.to_numeric(df_reg["门店类型编码"], errors="coerce")
    df_reg = df_reg.dropna(subset=["diff_log", "规模编码", "门店类型编码"])
    y = df_reg["diff_log"].astype(float)
    total_mean = y.mean()
    logging.info(f"diff_log总均值 = {total_mean:.4f}")
    
    # 哑变量
    region_dummies = pd.get_dummies(df_reg[COL_REGION], prefix="region", drop_first=False).astype(float)
    X_main = pd.concat([region_dummies, df_reg[["规模编码", "门店类型编码"]].astype(float)], axis=1).astype(float)
    
    # 交互项
    X_interact = X_main.copy()
    for region_col in region_dummies.columns:
        X_interact[f"{region_col}_×_规模编码"] = X_interact[region_col] * df_reg["规模编码"].values
        X_interact[f"{region_col}_×_门店类型编码"] = X_interact[region_col] * df_reg["门店类型编码"].values
    X_interact = X_interact.astype(float)
    
    # 交互整体检验
    sm_model_main = OLS(y, X_main).fit()
    sm_model_interact = OLS(y, X_interact).fit()
    anova_result = sm_model_main.compare_f_test(sm_model_interact)
    f_stat_interact, p_val_interact = anova_result[0], anova_result[1]
    logging.info(f"交互效应整体检验: F={f_stat_interact:.4f}, p={p_val_interact:.4f}")
    
    # 层级原则：交互显著时强制保留所有主效应
    if p_val_interact < sig_level:
        logging.info("交互项整体显著，强制保留所有主效应，并筛选显著交互项")
        main_effect_vars = X_main.columns.tolist()
        interact_pvals = sm_model_interact.pvalues
        sig_interact_vars = interact_pvals[interact_pvals < sig_level].index.tolist()
        final_vars = list(set(main_effect_vars + sig_interact_vars))
        X_final = X_interact[final_vars]
    else:
        logging.info("交互项整体不显著，仅保留显著主效应")
        main_pvals = sm_model_main.pvalues
        sig_main_vars = main_pvals[main_pvals < sig_level].index.tolist()
        if not sig_main_vars:
            sig_main_vars = region_dummies.columns.tolist()
        X_final = X_main[sig_main_vars]
    
    X_final = X_final.astype(float)
    final_model = OLS(y, X_final).fit()
    robust_model = final_model.get_robustcov_results(cov_type='HC3')
    logging.info("已使用HC3稳健标准误修正异方差影响")
    
    # ========== 模型诊断 ==========
    logging.info("===== 模型诊断 =====")
    logging.info(f"R² = {final_model.rsquared.round(4)}（解释了{final_model.rsquared*100:.1f}%的差异），调整R² = {final_model.rsquared_adj.round(4)}")
    f_value = final_model.fvalue
    f_pvalue = final_model.f_pvalue
    logging.info(f"回归方程F检验: F = {f_value:.4f}, p = {f_pvalue:.4f}")
    
    # 2. 多重共线性（VIF）
    X_vif = sm.add_constant(X_final)
    vif_df = pd.DataFrame()
    vif_df["变量名"] = X_vif.columns
    vif_df["VIF值"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif_df = vif_df.round(2)
    logging.info("VIF共线性诊断：\n" + vif_df.to_string(index=False))
    vif_df.to_excel(f"{OUTPUT_DIR}/VIF共线性诊断结果.xlsx", index=False, engine='openpyxl')
    
    # 3. 残差分析
    residuals = final_model.resid
    res_mean = round(residuals.mean(), 4)
    res_std = round(residuals.std(), 4)
    logging.info(f"残差均值 = {res_mean}，残差标准差 = {res_std}")
    
    # 3.1 残差正态性检验（Shapiro-Wilk）
    shapiro_stat, shapiro_p = shapiro(residuals)
    logging.info(f"残差正态性检验（Shapiro-Wilk）: p={shapiro_p:.4f}，{'正态' if shapiro_p>0.05 else '非正态'}")
    
    # 3.2 残差独立性检验（Durbin-Watson）
    dw = durbin_watson(residuals)
    logging.info(f"Durbin-Watson统计量: {dw:.3f}（理想值≈2，1.5~2.5可接受）")
    
    # 3.3 残差方差齐性检验（Breusch-Pagan）
    # 注意：het_breuschpagan需要原始模型的自变量矩阵（含常数项）
    print("X_final shape:", X_final.shape)
    print("X_final columns:", X_final.columns.tolist())
    print("final_model.model.exog shape:", final_model.model.exog.shape)
    X_with_const = sm.add_constant(X_final)   # 添加常数列，形状变为 (1217, 13)
    bp_test = het_breuschpagan(residuals, X_with_const)
    bp_lm, bp_lm_p, bp_f, bp_f_p = bp_test
    logging.info(f"Breusch-Pagan方差齐性检验: p={bp_lm_p:.4f}，{'方差齐' if bp_lm_p>0.05 else '方差不齐'}")
    
    # 残差图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    plt.title("残差分布直方图")
    plt.subplot(1,2,2)
    plt.scatter(final_model.fittedvalues, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("残差 vs 拟合值")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/残差诊断图.png", dpi=300)
    plt.close()
    
    # ========== 结果整理 ==========
    final_result = pd.DataFrame({
    "变量名": X_final.columns,
    "系数（相对总均值）": robust_model.params.round(4),
    "P值": robust_model.pvalues.round(4),
    "是否显著": robust_model.pvalues < sig_level
    })
    final_result["变量类型"] = final_result["变量名"].apply(
        lambda name: "区域主效应" if name.startswith("region_") and "_×_" not in name 
        else ("基础属性主效应" if name in ["规模编码", "门店类型编码"] 
              else ("交互效应" if "_×_" in name else "其他"))
    )
    final_result.to_excel(f"{OUTPUT_DIR}/多元回归结果（精简版_总均值基准）.xlsx", index=False, engine='openpyxl')
    
    # 区域边际贡献
    region_result = final_result[final_result["变量类型"] == "区域主效应"].copy()
    if not region_result.empty:
        region_result["实际区域名"] = region_result["变量名"].str.replace("region_", "")
        region_result["区域均值（总均值+系数）"] = (total_mean + region_result["系数（相对总均值）"]).round(4)
        region_result.to_excel(f"{OUTPUT_DIR}/区域边际贡献（精简版）.xlsx", index=False, engine='openpyxl')
        logging.info("区域边际贡献汇总：\n" + region_result[["实际区域名","系数（相对总均值）","P值","区域均值（总均值+系数）"]].to_string(index=False))
    
    return final_model, final_result, region_result

# ========== 10. 汇总报告（将所有关键结果合并到一个Excel） ==========
def generate_summary_report(df, anova_df, reg_result, region_contrib, vif_df):
    with pd.ExcelWriter(f"{OUTPUT_DIR}/全维度分析汇总报告.xlsx") as writer:
        compute_all_and_group(df).to_excel(writer, sheet_name="区域统计")
        if isinstance(anova_df, pd.DataFrame):
            anova_df.to_excel(writer, sheet_name="三因素ANOVA")
        reg_result.to_excel(writer, sheet_name="回归系数")
        if region_contrib is not None:
            region_contrib.to_excel(writer, sheet_name="区域边际贡献")
        vif_df.to_excel(writer, sheet_name="VIF诊断")
    logging.info("汇总报告已生成：全维度分析汇总报告.xlsx")

# ========== 11. 主函数 ==========
def main():
    logging.info("="*60)
    logging.info("门店活动全维度分析启动")
    logging.info("="*60)
    
    # 加载活动数据
    activity_df = load_and_preprocess(INPUT_FILE, FILE_TYPE, SHEET_NAME)
    # 匹配分层数据
    merged_df = merge_all_data(activity_df)
    
    # 小样本正态性检验（可选）
    normality_test_small_sample(merged_df)
    
    # EDA（新增）
    exploratory_analysis(merged_df)
    
    # 对数变换
    merged_df = robust_log_transform(merged_df)
    
    # 区域统计汇总
    compute_all_and_group(merged_df)
    
    # 可视化活动前后对比
    visualize(merged_df)
    
    # 三因素ANOVA（自动检验方差齐性并选择方法）
    anova_result_df = three_way_anova_main_effect(merged_df)
    
    # 多元线性回归（含完整诊断）
    final_model, reg_result, region_contrib = multiple_linear_regression_main_effect(merged_df)
    
    # 读取VIF结果（已保存）
    vif_df = pd.read_excel(f"{OUTPUT_DIR}/VIF共线性诊断结果.xlsx")
    
    # 生成汇总报告
    generate_summary_report(merged_df, anova_result_df, reg_result, region_contrib, vif_df)
    
    logging.info("✅ 全部完成！结果已保存至 output 文件夹，日志见 analysis.log")

if __name__ == "__main__":
    main()