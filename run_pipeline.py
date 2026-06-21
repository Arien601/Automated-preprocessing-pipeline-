import pandas as pd
import numpy as np
import os

def run_pipeline():
    print("1. 开始读取并对齐多源原始数据...")
    try:
        # 这里对应你上传的真实文件
        fred = pd.read_csv('raw_fred.csv')
        jpm = pd.read_csv('raw_jpm.csv')
        vix = pd.read_csv('raw_vix.csv')
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}. 请确保 raw_fred.csv, raw_jpm.csv, raw_vix.csv 已上传。")
        return

    # 日期标准化
    fred['Date'] = pd.to_datetime(fred['date'], errors='coerce')
    jpm['Date'] = pd.to_datetime(jpm['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    vix['Date'] = pd.to_datetime(vix['Date'], utc=True).dt.tz_localize(None).dt.normalize()

    fred.set_index('Date', inplace=True)
    jpm.set_index('Date', inplace=True)
    vix.set_index('Date', inplace=True)

    # 以 JPM 为主轴合并数据
    df = pd.DataFrame(index=jpm.index)
    df['JPM_Close'] = jpm['Close']
    df['JPM_Volume'] = jpm['Volume']
    df['JPM_Dividends'] = jpm['Dividends']
    df['VIX_Close'] = vix['Close']
    df = df.join(fred['Treasury_3M']).rename(columns={'Treasury_3M': 'Treasury_Rate'})

    # --- 1. 数据预处理 ---
    df['Treasury_Rate'] = df['Treasury_Rate'].ffill()
    df['VIX_Close'] = df['VIX_Close'].interpolate(method='time')
    df.dropna(subset=['JPM_Close', 'VIX_Close', 'Treasury_Rate'], inplace=True)

    def rolling_winsorize(series, window=63, multiplier=1.5):
        Q1 = series.rolling(window=window, min_periods=21).quantile(0.25)
        Q3 = series.rolling(window=window, min_periods=21).quantile(0.75)
        IQR = Q3 - Q1
        return series.clip(lower=Q1 - multiplier * IQR, upper=Q3 + multiplier * IQR)

    for col in ['JPM_Close', 'VIX_Close', 'Treasury_Rate']:
        df[col] = rolling_winsorize(df[col])

    print("2. 开始执行高级特征工程...")
    
    # --- Traditional 特征 ---
    df['JPM_Log_Return'] = np.log(df['JPM_Close'] / df['JPM_Close'].shift(1))
    df['JPM_Vol_21d'] = df['JPM_Log_Return'].rolling(window=21).std() * np.sqrt(252)
    
    # 补全：真实股息增长
    div_ffill = df['JPM_Dividends'].replace(0, np.nan).ffill()
    df['Dividend_Growth'] = div_ffill.pct_change(fill_method=None).fillna(0)

    # --- Advanced 特征 ---
    df['VIX_Log_Return'] = np.log(df['VIX_Close'] / df['VIX_Close'].shift(1))
    df['VIX_JPM_Corr_21d'] = df['JPM_Log_Return'].rolling(window=21).corr(df['VIX_Log_Return'])
    df['Rate_Momentum_63d'] = df['Treasury_Rate'] - df['Treasury_Rate'].rolling(window=63).mean()
    
    # 补全：真实情感得分 (0-1)
    roll_min = df['VIX_Close'].rolling(window=252, min_periods=21).min()
    roll_max = df['VIX_Close'].rolling(window=252, min_periods=21).max()
    df['Sentiment_Score'] = (1 - (df['VIX_Close'] - roll_min) / (roll_max - roll_min)).fillna(0.5)

    # --- 顶级文献衍生特征 ---
    df['Vol_x_Rate'] = df['JPM_Vol_21d'] * df['Treasury_Rate']
    df['Ret_x_Rate'] = df['JPM_Log_Return'] * df['Treasury_Rate']
    df['JPM_Ret_Sq'] = df['JPM_Log_Return'] ** 2
    df['JPM_Vol_EWMA'] = df['JPM_Log_Return'].ewm(span=33).std() * np.sqrt(252)

    # --- 3. 终极截断与保存 ---
    df.dropna(inplace=True)
    
    feature_columns = [
        'JPM_Log_Return', 'JPM_Vol_21d', 'Dividend_Growth', 
        'VIX_JPM_Corr_21d', 'Rate_Momentum_63d', 'Sentiment_Score',
        'Vol_x_Rate', 'Ret_x_Rate', 'JPM_Ret_Sq', 'JPM_Vol_EWMA'
    ]
    
    final_df = df[feature_columns].copy()
    for col in final_df.columns:
        final_df[col] = rolling_winsorize(final_df[col])

    # ====== 核心修改点：强制绝对路径 ======
    # 如果在 GitHub Actions 中，自动获取根目录；如果在本地，获取当前目录
    workspace_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())
    save_path = os.path.join(workspace_path, 'jpm_features_final.parquet')
    
    final_df.to_parquet(save_path)
    
    print(f"3. ✅ 成功导出特征矩阵到绝对路径: {save_path}")
    print(f"   📊 最终特征数: {len(final_df.columns)} | 数据集形状: {final_df.shape}")

if __name__ == "__main__":
    run_pipeline()
