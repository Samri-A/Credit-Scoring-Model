import pandas as pd
from sklearn.pipeline import make_pipeline , Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , FunctionTransformer , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError as e: 
        print(e)
    except:
        print("Could not upload the data")

        
# Function to compute Weight of Evidence
# Source: 
# http://www.sanaitics.com/UploadedFiles/html_files/1770WoE_RvsPython.html

def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Bin Values': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & 
                    (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & 
                    (dataset[target] == 1)].count()[feature]
        }) 
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv

def rsm(df , months=3):
 
    snapshot_date = df["TransactionStartTime"].max()
    
    recent_period = snapshot_date - relativedelta(months=months)
    
    df_recent = df[df["TransactionStartTime"] >= recent_period]
    rfm = df_recent.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "CustomerId": "count",
        "Amount": "sum" 
    })
    rfm.columns = ["Recency", "Frequency", "Monetary" ]
    return rfm.reset_index()



def top(x):
    return x.mode().iloc[0]



def wraggle(df):
    #remove outliner in ammount

    q05 = df["Amount"].quantile(0.1)
    q95 = df["Amount"].quantile(0.9)

    # Keep only the middle 80% of data
    df = df[(df["Amount"] >= q05) & (df["Amount"] <= q95)]

    # Normalization
    min_max_scaler = MinMaxScaler()
    df["Amount"] = min_max_scaler.fit_transform(df[["Amount"]])
    
    # normalize the date column and extract dates
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour

    #sort the data based on date
    df.sort_values("TransactionStartTime", inplace=True)

    # compute the amount of transaction
    num_df = df.groupby("CustomerId").agg({
    "Amount": ["sum", "mean", "count", "std"] ,
     "FraudResult": "max"
    }).reset_index()

    cat_df = df.groupby("CustomerId").agg({
     "ProductCategory" :top,
     "ChannelId" :top,
     "ProviderId" :top,
    }).reset_index()
    
    num_df.columns = [ "CustomerId", "Total_Amount" , "Average_Amount" , "Transaction_Count" , "Std_Amount" , "Has_Fraud"] 
    
    processed_df = pd.merge(num_df, cat_df , on ="CustomerId" , how = "inner")

    processed_df["Std_Amount"] = processed_df["Std_Amount"].apply(lambda x: 0 if pd.isna(x) else x)

    rsm_df = rsm(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rsm_df.drop(columns="CustomerId"))
    kmeans = KMeans(n_clusters=3, random_state=42)
    rsm_df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = rsm_df.groupby("Cluster")[["Frequency", "Monetary", "Recency"]].mean()
    sorted_clusters = cluster_summary.sort_values(by=["Frequency", "Monetary", "Recency"], ascending=[True, True, False])
    high_risk_cluster = sorted_clusters.index[0]

    rsm_df["is_high_risk"] = (rsm_df["Cluster"] == high_risk_cluster).astype(int)
     
    processed_df = pd.merge(processed_df, rsm_df[["is_high_risk" , "CustomerId"]] , on="CustomerId" , how="inner")

    processed_copy_df = processed_df.copy()
    # Create binned versions of continuous numeric features
    processed_copy_df['Total_Amount_bin'] = pd.qcut(processed_copy_df['Total_Amount'], 10, duplicates='drop')
    processed_copy_df['Average_Amount_bin'] = pd.qcut(processed_copy_df['Average_Amount'], 10, duplicates='drop')
    processed_copy_df['Transaction_Count_bin'] = pd.qcut(processed_copy_df['Transaction_Count'], 10, duplicates='drop')
    processed_copy_df['Std_Amount_bin'] = pd.qcut(processed_copy_df['Std_Amount'], 10, duplicates='drop')
    
    # Combine all into a single processed_copy_dfFrame for IV calculation
    processed_copy_df_woe = processed_copy_df[['Total_Amount_bin', 'Average_Amount_bin', 'Transaction_Count_bin',
                     'Std_Amount_bin', 'ProductCategory', 'ChannelId', 'ProviderId', 'is_high_risk']]
    lst = []
    IV_df = pd.DataFrame(columns=['Variable', 'IV'])
    
    for col in processed_copy_df.columns:
        if col == 'is_high_risk':
            continue
        df_iv, iv = calculate_woe_iv(processed_copy_df, col, 'is_high_risk')
        lst.append(df_iv)
    
        
        IV_df = pd.concat([
            IV_df,
            pd.DataFrame([{"Variable": col, "IV": iv}])
        ], ignore_index=True)
        cols_to_drop = []
        for i in IV_df["IV"]:
            if i < 0.3:
                col= IV_df[IV_df["IV"] == i]["Variable"].values[0]
                if col != 'Average_Amount_bin':
                    cols_to_drop.append(col)
                
        processed_df.drop(columns=cols_to_drop)
    return processed_df






numerical_features = ["Total_Amount", "Average_Amount", "Transaction_Count", "Std_Amount"]
categorical_features = [ "is_high_risk"]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # or 'median'
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    #('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])



my_pipeline = make_pipeline(
    FunctionTransformer(wraggle, validate=False),
    preprocess  )