import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score, roc_curve, auc
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Conv2D, Flatten, LSTM, GRU, Attention
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import joblib
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Streamlit Configuration
st.set_page_config(page_title="AutoMLX", page_icon="🧠", layout="wide")

# Title
st.title("AutoMLX🧠")
st.write(":rainbow[ Data Scientist's Portal]")
st.write("### Your Ultimate Platform for Machine Learning, Deep Learning and Data Visualization")

# Step 1: Problem Definition
st.write("### Step 1: Problem Definition")
problem = st.text_area("Define your problem statement:", help="State the problem or task you are trying to solve.")
if problem:
    st.info(f"Problem Statement: {problem}")

# Step 2: Data Acquisition
st.write("### Step 2: Data Acquisition")
file = st.file_uploader("Upload your dataset (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"], help="Upload your dataset to start the analysis.")

# Initialize variables
model = None
X_test = None
y_test = None
predictions = None

if file:
    try:
        # Load dataset
        if file.name.endswith("csv"):
            data = pd.read_csv(file)
        elif file.name.endswith("xlsx"):
            data = pd.read_excel(file)
        else:
            data = pd.read_json(file)

        if data.empty:
            st.error("The dataset is empty. Please upload a valid dataset.")
        else:
            st.success("File uploaded successfully!")
            st.dataframe(data.head())

            # Tabs for workflow
            tabs = st.tabs(["📊 Data Overview", "🧹 Data Cleaning", "🔬 Feature Engineering",
                            "📊 EDA", "🤖 ML Modeling", "🧠 DL Modeling"])

            # Tab 1: Data Overview
# Tab 1: Data Overview
# Tab 1: Data Overview
            with tabs[0]:
                st.subheader(":rainbow[Data Overview]")

                # Display Basic Information
                try:
                    st.markdown("### Dataset Summary")
                    st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
                    st.write("**Memory Usage:** {:.2f} MB".format(data.memory_usage(deep=True).sum() / (1024 ** 2)))
                except Exception as e:
                    st.warning(f"Error in Dataset Summary: {e}")

                # Data Types Overview
                try:
                    st.markdown("### Data Types")
                    dtype_info = pd.DataFrame({
                        "Column": data.columns,
                        "Data Type": data.dtypes,
                        "Non-Null Count": data.notnull().sum(),
                        "Null Count": data.isnull().sum(),
                        "Unique Values": data.nunique()
                    }).set_index("Column")
                    st.dataframe(dtype_info)
                except Exception as e:
                    st.warning(f"Error in Data Types Overview: {e}")

                # Missing Values Overview
                try:
                    st.markdown("### Missing Values Overview")
                    missing_values = data.isnull().sum()
                    missing_percent = (missing_values / len(data)) * 100
                    missing_df = pd.DataFrame({
                        "Missing Values": missing_values,
                        "Percent (%)": missing_percent
                    }).sort_values(by="Missing Values", ascending=False)
                    st.dataframe(missing_df)
                except Exception as e:
                    st.warning(f"Error in Missing Values Overview: {e}")

                # Statistical Summary
                try:
                    st.markdown("### Statistical Summary (Numerical Data)")
                    st.dataframe(data.describe().T)
                except Exception as e:
                    st.warning(f"Error in Statistical Summary: {e}")

                # Advanced Insights: Unique and Zero Value Counts
                try:
                    st.markdown("### Advanced Insights")
                    
                    st.markdown("#### Unique Values Per Column")
                    unique_values = pd.DataFrame(data.nunique(), columns=["Unique Values"]).sort_values(by="Unique Values", ascending=False)
                    st.dataframe(unique_values)
                    
                    st.markdown("#### Zero Values Count")
                    zero_counts = (data == 0).sum()
                    zero_df = pd.DataFrame({"Zero Count": zero_counts}).sort_values(by="Zero Count", ascending=False)
                    st.dataframe(zero_df)
                except Exception as e:
                    st.warning(f"Error in Advanced Insights: {e}")

                # Correlation Matrix

                # Visualizations
                try:
                    st.markdown("### Visualizations")

                    # Missing Values Heatmap
                    st.markdown("#### Missing Values Heatmap")
                    if data.isnull().sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(data.isnull(), cbar=False, cmap="coolwarm", linewidths=0.5, linecolor="gray")
                        st.pyplot(fig)
                    else:
                        st.info("No missing values to display in heatmap.")
                except Exception as e:
                    st.warning(f"Error in Missing Values Heatmap: {e}")

                # Distribution of Numerical Data
                try:
                    st.markdown("#### Distribution of Numerical Columns")
                    numeric_cols = data.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, len(numeric_cols) * 3))
                        if len(numeric_cols) == 1:
                            sns.histplot(data[numeric_cols[0]], kde=True, color="skyblue", ax=axes)
                        else:
                            for i, col in enumerate(numeric_cols):
                                sns.histplot(data[col], kde=True, color="skyblue", ax=axes[i])
                        st.pyplot(fig)
                    else:
                        st.info("No numerical columns available for distribution analysis.")
                except Exception as e:
                    st.warning(f"Error in Distribution of Numerical Columns: {e}")

                # Value Counts for Categorical Columns
                try:
                    st.markdown("#### Top Categories for Categorical Columns")
                    cat_cols = data.select_dtypes(include='object').columns
                    for col in cat_cols:
                        st.markdown(f"**{col}**")
                        st.dataframe(data[col].value_counts().head(10))
                except Exception as e:
                    st.warning(f"Error in Top Categories for Categorical Columns: {e}")





            # Tab 2: Data Cleaning
            # Tab 2: Data Cleaning
            with tabs[1]:
                st.subheader(":rainbow[Data Cleaning]")

                # Display Original Data
                st.markdown("### Original Dataset")
                st.dataframe(data)

                try:
                    # Handle Missing Values
                    st.markdown("### Missing Values Handling")
                    missing_strategy = st.radio("Choose a strategy for handling missing values:", 
                                                ["Mean", "Median", "Most Frequent", "Remove Rows", "Custom Value"])
                    custom_value = None
                    if missing_strategy == "Custom Value":
                        custom_value = st.text_input("Enter custom value for missing data:")
                    if st.button("Handle Missing Values"):
                        if missing_strategy == "Remove Rows":
                            data = data.dropna()
                        else:
                            strategy = custom_value if custom_value else missing_strategy.lower()
                            imputer = SimpleImputer(strategy=strategy)
                            data.iloc[:, :] = imputer.fit_transform(data)
                        st.success("Missing values handled successfully!")
                        st.dataframe(data)
                except Exception as e:
                    st.warning(f"An error occurred while handling missing values: {e}")

                try:
                    # Handle Duplicate Rows
                    st.markdown("### Duplicate Rows")
                    if st.checkbox("Check for duplicate rows"):
                        duplicates = data.duplicated().sum()
                        st.write(f"Found {duplicates} duplicate rows.")
                        if duplicates > 0 and st.button("Remove Duplicates"):
                            data = data.drop_duplicates()
                            st.success("Duplicate rows removed!")
                            st.dataframe(data)
                except Exception as e:
                    st.warning(f"An error occurred while handling duplicate rows: {e}")

                try:
                    # Handle Outliers
                    st.markdown("### Outlier Detection and Handling")
                    numeric_columns = data.select_dtypes(include='number').columns
                    if len(numeric_columns) > 0:
                        outlier_column = st.selectbox("Select a column for outlier handling:", numeric_columns)
                        outlier_strategy = st.radio("Choose a strategy:", ["Remove", "Cap"])
                        threshold = st.slider("Threshold (number of standard deviations):", 1, 5, 3)
                        if st.button("Handle Outliers"):
                            mean, std = data[outlier_column].mean(), data[outlier_column].std()
                            lower_bound, upper_bound = mean - threshold * std, mean + threshold * std
                            if outlier_strategy == "Remove":
                                data = data[(data[outlier_column] >= lower_bound) & (data[outlier_column] <= upper_bound)]
                            elif outlier_strategy == "Cap":
                                data[outlier_column] = data[outlier_column].clip(lower_bound, upper_bound)
                            st.success("Outliers handled successfully!")
                            st.dataframe(data)
                    else:
                        st.info("No numeric columns available for outlier handling.")
                except Exception as e:
                    st.warning(f"An error occurred while handling outliers: {e}")

                try:
                    # Rename Columns
                    st.markdown("### Column Renaming")
                    if st.checkbox("Rename Columns"):
                        column_names = st.text_area("Enter new column names (comma-separated):", value=", ".join(data.columns))
                        if st.button("Apply Column Renaming"):
                            new_names = [name.strip() for name in column_names.split(',')]
                            if len(new_names) == len(data.columns):
                                data.columns = new_names
                                st.success("Columns renamed successfully!")
                                st.dataframe(data)
                            else:
                                st.warning("Number of new names must match the number of columns!")
                except Exception as e:
                    st.warning(f"An error occurred while renaming columns: {e}")

                try:
                    # Data Type Conversion
                    st.markdown("### Data Type Conversion")
                    selected_column = st.selectbox("Select a column for type conversion:", data.columns)
                    new_dtype = st.radio("Choose the new data type:", ["int", "float", "string", "datetime"])
                    if st.button("Convert Data Type"):
                        try:
                            if new_dtype == "int":
                                data[selected_column] = data[selected_column].astype(int)
                            elif new_dtype == "float":
                                data[selected_column] = data[selected_column].astype(float)
                            elif new_dtype == "string":
                                data[selected_column] = data[selected_column].astype(str)
                            elif new_dtype == "datetime":
                                data[selected_column] = pd.to_datetime(data[selected_column])
                            st.success(f"Data type of '{selected_column}' converted to {new_dtype} successfully!")
                            st.dataframe(data)
                        except Exception as inner_e:
                            st.warning(f"Error converting data type: {inner_e}")
                except Exception as e:
                    st.warning(f"An error occurred while converting data types: {e}")

                try:
                    # Remove Unwanted Characters
                    st.markdown("### Remove Unwanted Characters")
                    column_to_clean = st.selectbox("Select a column to clean:", data.columns)
                    unwanted_chars = st.text_input("Enter unwanted characters (comma-separated):")
                    if st.button("Clean Column"):
                        for char in unwanted_chars.split(','):
                            data[column_to_clean] = data[column_to_clean].str.replace(char.strip(), '')
                        st.success(f"Unwanted characters removed from column '{column_to_clean}'!")
                        st.dataframe(data)
                except Exception as e:
                    st.warning(f"An error occurred while removing unwanted characters: {e}")

                # Final Cleaned Dataset Display
                st.markdown("### Final Cleaned Dataset")
                st.dataframe(data)
                df=data.copy()



            # Tab 3: Feature Engineering
            with tabs[2]:
                st.subheader(":rainbow[Feature Engineering]")

                try:
                    # Scaling Features
                    if st.checkbox("Scale Features"):
                        scale_method = st.selectbox("Choose a scaling method:", ["Standardization (Z-Score)", "Normalization (Min-Max)"])
                        try:
                            scaler = StandardScaler() if scale_method == "Standardization (Z-Score)" else MinMaxScaler()
                            num_cols = data.select_dtypes(include=[np.number]).columns
                            if len(num_cols) > 0:
                                data[num_cols] = scaler.fit_transform(data[num_cols])
                                st.success(f"Features scaled using {scale_method} successfully!")
                                st.dataframe(data)
                            else:
                                st.warning("No numerical columns available for scaling.")
                        except Exception as e:
                            st.warning(f"Error in Scaling Features: {e}")

                    # Encoding Categorical Features
                    if st.checkbox("Encode Categorical Data"):
                        encoding_method = st.selectbox("Choose an encoding method:", ["Label Encoding", "One-Hot Encoding"])
                        try:
                            cat_cols = data.select_dtypes(include=["object"]).columns
                            if len(cat_cols) > 0:
                                if encoding_method == "Label Encoding":
                                    for col in cat_cols:
                                        data[col] = LabelEncoder().fit_transform(data[col])
                                    st.success("Categorical data encoded using Label Encoding successfully!")
                                elif encoding_method == "One-Hot Encoding":
                                    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
                                    st.success("Categorical data encoded using One-Hot Encoding successfully!")
                                st.dataframe(data)
                            else:
                                st.warning("No categorical columns available for encoding.")
                        except Exception as e:
                            st.warning(f"Error in Encoding Categorical Data: {e}")

                    # Handling Outliers
                    if st.checkbox("Handle Outliers"):
                        try:
                            outlier_method = st.selectbox("Choose a method for handling outliers:", ["Remove Outliers (IQR)", "Cap Outliers"])
                            num_cols = data.select_dtypes(include=[np.number]).columns
                            if len(num_cols) > 0:
                                if outlier_method == "Remove Outliers (IQR)":
                                    Q1 = data[num_cols].quantile(0.25)
                                    Q3 = data[num_cols].quantile(0.75)
                                    IQR = Q3 - Q1
                                    data = data[~((data[num_cols] < (Q1 - 1.5 * IQR)) | (data[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
                                    st.success("Outliers removed successfully using IQR method!")
                                elif outlier_method == "Cap Outliers":
                                    for col in num_cols:
                                        Q1 = data[col].quantile(0.25)
                                        Q3 = data[col].quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - 1.5 * IQR
                                        upper_bound = Q3 + 1.5 * IQR
                                        data[col] = np.clip(data[col], lower_bound, upper_bound)
                                    st.success("Outliers capped successfully!")
                                st.dataframe(data)
                            else:
                                st.warning("No numerical columns available for outlier handling.")
                        except Exception as e:
                            st.warning(f"Error in Handling Outliers: {e}")

                    # Feature Selection
                    if st.checkbox("Select Important Features"):
                        try:
                            target_col = st.selectbox("Choose the target column for feature selection:", data.columns)
                            X = data.drop(columns=[target_col])
                            y = data[target_col]
                            selector = SelectKBest(score_func=f_classif if y.dtypes == 'O' else f_regression, k='all')
                            selector.fit(X, y)
                            scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_}).sort_values(by='Score', ascending=False)
                            st.markdown("### Feature Importance Scores")
                            st.dataframe(scores)
                        except Exception as e:
                            st.warning(f"Error in Feature Selection: {e}")

                    # Polynomial Features
                    if st.checkbox("Generate Polynomial Features"):
                        try:
                            degree = st.slider("Choose polynomial degree:", 2, 5, 2)
                            poly = PolynomialFeatures(degree=degree, include_bias=False)
                            num_cols = data.select_dtypes(include=[np.number]).columns
                            if len(num_cols) > 0:
                                poly_features = poly.fit_transform(data[num_cols])
                                poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols))
                                data = pd.concat([data.reset_index(drop=True), poly_df], axis=1)
                                st.success(f"Polynomial features up to degree {degree} generated successfully!")
                                st.dataframe(data)
                            else:
                                st.warning("No numerical columns available for generating polynomial features.")
                        except Exception as e:
                            st.warning(f"Error in Generating Polynomial Features: {e}")

                    # Correlation Matrix Visualization
                    if st.checkbox("Show Correlation Matrix"):
                        try:
                            st.markdown("### Correlation Matrix")
                            corr = data.corr()
                            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Correlation Matrix")
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.warning(f"Error in Correlation Matrix Visualization: {e}")
            
                except Exception as e:
                    st.warning(f"An error occurred while performing feature engineering: {e}")

                    
                    
                    

           

# Tab 4: EDA (Exploratory Data Analysis)
            with tabs[3]:
                st.subheader(":rainbow[Exploratory Data Analysis]")
                
            
                try:
                    # User selects chart type
                    chart_type = st.radio("Choose a chart type:", [
                        "Histogram", "Boxplot", "Scatter Plot", "Heatmap", "Pairplot", 
                        "Correlation Matrix", "Violin Plot", "Countplot", "Bar Plot", 
                        "Pie Chart", "Density Plot", "Line Plot", "Area Plot", 
                        "Hexbin Plot", "Radar Chart", 
                        "3D Scatter Plot", "3D Surface Plot", "3D Histogram", "3D Line Plot", 
                        "3D Scatter Matrix"], key="eda_chart_type")
                    
                    # Histogram with Plotly
                    if chart_type == "Histogram":
                        col = st.selectbox("Select a column for the histogram:", data.columns, key="eda_hist_column")
                        fig = px.histogram(data, x=col, nbins=30, title=f"Histogram of {col}")
                        st.plotly_chart(fig)

                    # Boxplot with Plotly
                    elif chart_type == "Boxplot":
                        col = st.selectbox("Select a column for the boxplot:", data.columns, key="eda_box_column")
                        fig = px.box(data, y=col, title=f"Boxplot of {col}")
                        st.plotly_chart(fig)

                    # Scatter Plot with Plotly
                    elif chart_type == "Scatter Plot":
                        x_col = st.selectbox("Select the X-axis column:", data.columns, key="eda_scatter_x")
                        y_col = st.selectbox("Select the Y-axis column:", data.columns, key="eda_scatter_y")
                        fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatter Plot of {x_col} vs {y_col}")
                        st.plotly_chart(fig)

                    # Heatmap with Plotly
                    elif chart_type == "Heatmap":
                        fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap")
                        st.plotly_chart(fig)

                    # Pairplot with Seaborn
                    elif chart_type == "Pairplot":
                        cols = st.multiselect("Select columns for pairplot:", data.columns)
                        if len(cols) > 1:
                            fig = sns.pairplot(data[cols])
                            st.pyplot(fig)
                        else:
                            st.warning("Please select more than one column for pairplot.")

                    # Correlation Matrix with Seaborn
                    elif chart_type == "Correlation Matrix":
                        corr = data.corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)
                        st.pyplot(fig)

                    # Violin Plot with Plotly
                    elif chart_type == "Violin Plot":
                        col = st.selectbox("Select a column for the violin plot:", data.columns, key="eda_violin_column")
                        fig = px.violin(data, y=col, box=True, points="all", title=f"Violin Plot of {col}")
                        st.plotly_chart(fig)

                    # Countplot with Seaborn
                    elif chart_type == "Countplot":
                        col = st.selectbox("Select a categorical column for countplot:", data.select_dtypes(include=["object"]).columns)
                        fig = px.histogram(data, x=col, title=f"Countplot of {col}")
                        st.plotly_chart(fig)

                    # Bar Plot with Plotly
                    elif chart_type == "Bar Plot":
                        col = st.selectbox("Select a categorical column for bar plot:", data.select_dtypes(include=["object"]).columns)
                        bar_data = data[col].value_counts().reset_index()
                        bar_data.columns = [col, 'Count']
                        fig = px.bar(bar_data, x=col, y="Count", title=f"Bar Plot of {col}")
                        st.plotly_chart(fig)

                    # Pie Chart with Plotly
                    elif chart_type == "Pie Chart":
                        col = st.selectbox("Select a categorical column for pie chart:", data.select_dtypes(include=["object"]).columns)
                        pie_data = data[col].value_counts().reset_index()
                        pie_data.columns = [col, 'Count']
                        fig = px.pie(pie_data, names=col, values="Count", title=f"Pie Chart of {col}")
                        st.plotly_chart(fig)

                    # Density Plot (KDE) with Seaborn
                    elif chart_type == "Density Plot":
                        col = st.selectbox("Select a numerical column for density plot:", data.select_dtypes(include=[np.number]).columns)
                        fig = plt.figure(figsize=(10, 6))
                        sns.kdeplot(data[col])
                        st.pyplot(fig)

                    # Line Plot with Matplotlib
                    elif chart_type == "Line Plot":
                        col = st.selectbox("Select a column for the line plot:", data.select_dtypes(include=[np.number]).columns)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(data[col])
                        ax.set_title(f"Line Plot of {col}")
                        st.pyplot(fig)

                    # Area Plot with Pandas
                    elif chart_type == "Area Plot":
                        col = st.selectbox("Select a column for the area plot:", data.select_dtypes(include=[np.number]).columns)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        data[col].plot(kind="area", ax=ax)
                        ax.set_title(f"Area Plot of {col}")
                        st.pyplot(fig)

                    # Hexbin Plot with Matplotlib
                    elif chart_type == "Hexbin Plot":
                        x_col = st.selectbox("Select the X-axis column for Hexbin plot:", data.columns, key="eda_hexbin_x")
                        y_col = st.selectbox("Select the Y-axis column for Hexbin plot:", data.columns, key="eda_hexbin_y")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        hb = ax.hexbin(data[x_col], data[y_col], gridsize=30, cmap='Blues')
                        fig.colorbar(hb, ax=ax)
                        ax.set_title(f"Hexbin Plot of {x_col} vs {y_col}")
                        st.pyplot(fig)


                    # Radar Chart (Custom Plotly Radar)
                    elif chart_type == "Radar Chart":
                        categories = st.multiselect("Select columns for Radar chart:", data.columns)
                        if len(categories) > 1:
                            radar_data = data[categories].mean().reset_index()
                            radar_data.columns = ["Category", "Value"]
                            fig = px.line_polar(radar_data, r="Value", theta="Category", line_close=True, title="Radar Chart")
                            st.plotly_chart(fig)
                        
                    # 3D Scatter Plot with Plotly
                    elif chart_type == "3D Scatter Plot":
                        x_col = st.selectbox("Select the X-axis column for 3D scatter plot:", data.columns, key="eda_3dscatter_x")
                        y_col = st.selectbox("Select the Y-axis column for 3D scatter plot:", data.columns, key="eda_3dscatter_y")
                        z_col = st.selectbox("Select the Z-axis column for 3D scatter plot:", data.columns, key="eda_3dscatter_z")
                        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, title="3D Scatter Plot")
                        st.plotly_chart(fig)

                    # 3D Surface Plot with Plotly
                    elif chart_type == "3D Surface Plot":
                        x_col = st.selectbox("Select the X-axis column for 3D surface plot:", data.columns, key="eda_3dsurface_x")
                        y_col = st.selectbox("Select the Y-axis column for 3D surface plot:", data.columns, key="eda_3dsurface_y")
                        z_col = st.selectbox("Select the Z-axis column for 3D surface plot:", data.columns, key="eda_3dsurface_z")
                        fig = go.Figure(data=[go.Surface(z=data[z_col].values.reshape((len(data), 1)),
                                                        x=data[x_col],
                                                        y=data[y_col])])
                        fig.update_layout(title="3D Surface Plot")
                        st.plotly_chart(fig)

                    # 3D Histogram with Plotly
                    elif chart_type == "3D Histogram":
                        x_col = st.selectbox("Select the X-axis column for 3D histogram:", data.columns, key="eda_3dhist_x")
                        y_col = st.selectbox("Select the Y-axis column for 3D histogram:", data.columns, key="eda_3dhist_y")
                        z_col = st.selectbox("Select the Z-axis column for 3D histogram:", data.columns, key="eda_3dhist_z")
                        fig = px.histogram_3d(data, x=x_col, y=y_col, z=z_col, title="3D Histogram")
                        st.plotly_chart(fig)

                    # 3D Line Plot (custom with Plotly)
                    elif chart_type == "3D Line Plot":
                        x_col = st.selectbox("Select the X-axis column for 3D line plot:", data.columns, key="eda_3dline_x")
                        y_col = st.selectbox("Select the Y-axis column for 3D line plot:", data.columns, key="eda_3dline_y")
                        z_col = st.selectbox("Select the Z-axis column for 3D line plot:", data.columns, key="eda_3dline_z")
                        fig = go.Figure(data=[go.Scatter3d(x=data[x_col], y=data[y_col], z=data[z_col], mode='lines')])
                        fig.update_layout(title="3D Line Plot")
                        st.plotly_chart(fig)

                except Exception as e:
                    st.warning(f"An error occurred while generating the plot: {e}")


            # Tab 5: ML Modeling
           
# Tab 5: ML Modeling
            

# Tab 5: ML Modeling
            with tabs[4]:
                st.subheader(":rainbow[Machine Learning Modeling]")

                try:
                    model_type = st.radio("Choose a model type:", ["Supervised", "Unsupervised", "Semi-Supervised"], key="ml_model_type")

                    if model_type == "Supervised":
                        task_type = st.radio("Choose a task:", ["Classification", "Regression"], key="ml_task_type")
                        target = st.selectbox("Select the target variable:", options=["None"] + list(data.columns), key="ml_target")
                        features = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target], key="ml_features")
                       
                        # User input for test size
                        test_size = st.slider("Select test size:", 0.1, 0.9, 0.2)

                        if features:
                            X = data[features]
                            if target != "None":
                                y = data[target]
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                                # Supervised Classification
                                if task_type == "Classification":
                                    clf_model = st.selectbox("Choose a Classification Model:", ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Naive Bayes", "Decision Tree"])
                                    
                                    if clf_model == "Logistic Regression":
                                        model = LogisticRegression(max_iter=10000)
                                        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
                                    elif clf_model == "Random Forest":
                                        model = RandomForestClassifier()
                                        param_grid = {
                                            'n_estimators': [10, 50, 100, 200],
                                            'max_depth': [5, 10, 20],
                                            'min_samples_split': [2, 5, 10]
                                        }
                                    elif clf_model == "XGBoost":
                                        model = xgb.XGBClassifier()
                                        param_grid = {
                                            'n_estimators': [100, 200],
                                            'learning_rate': [0.01, 0.1, 0.2],
                                            'max_depth': [3, 6, 10]
                                        }
                                    elif clf_model == "SVM":
                                        model = SVC()
                                        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                                    elif clf_model == "Naive Bayes":
                                        model = GaussianNB()
                                        param_grid = {}
                                    elif clf_model == "Decision Tree":
                                        model = DecisionTreeClassifier()
                                        param_grid = {
                                            'max_depth': [5, 10, 20],
                                            'min_samples_split': [2, 5, 10]
                                        }

                                    # Hyperparameter tuning using GridSearchCV
                                    if param_grid:
                                        grid_search = GridSearchCV(model, param_grid, cv=5)
                                        grid_search.fit(X_train, y_train)
                                        model = grid_search.best_estimator_
                                        st.write(f"Best Parameters: {grid_search.best_params_}")

                                    # Model training and evaluation
                                    model.fit(X_train, y_train)
                                    st.success(f"{clf_model} trained successfully!")
                                    train_predictions = model.predict(X_train)
                                    test_predictions = model.predict(X_test)

                                    train_accuracy = accuracy_score(y_train, train_predictions)
                                    test_accuracy = accuracy_score(y_test, test_predictions)

                                    st.write(f"Training Accuracy: {train_accuracy:.2f}")
                                    st.write(f"Test Accuracy: {test_accuracy:.2f}")

                                # Supervised Regression
                                elif task_type == "Regression":
                                    reg_model = st.selectbox("Choose a Regression Model:", ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "SVR", "Decision Tree Regressor"])

                                    if reg_model == "Linear Regression":
                                        model = LinearRegression()
                                        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
                                    elif reg_model == "Random Forest Regressor":
                                        model = RandomForestRegressor()
                                        param_grid = {
                                            'n_estimators': [10, 50, 100, 200],
                                            'max_depth': [5, 10, 20],
                                            'min_samples_split': [2, 5, 10]
                                        }
                                    elif reg_model == "XGBoost Regressor":
                                        model = xgb.XGBRegressor()
                                        param_grid = {
                                            'n_estimators': [100, 200],
                                            'learning_rate': [0.01, 0.1, 0.2],
                                            'max_depth': [3, 6, 10]
                                        }
                                    elif reg_model == "SVR":
                                        model = SVR()
                                        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                                    elif reg_model == "Decision Tree Regressor":
                                        model = DecisionTreeRegressor()
                                        param_grid = {
                                            'max_depth': [5, 10, 20],
                                            'min_samples_split': [2, 5, 10]
                                        }

                                    # Hyperparameter tuning using GridSearchCV
                                    if param_grid:
                                        grid_search = GridSearchCV(model, param_grid, cv=5)
                                        grid_search.fit(X_train, y_train)
                                        model = grid_search.best_estimator_
                                        st.write(f"Best Parameters: {grid_search.best_params_}")

                                    # Model training and evaluation
                                    model.fit(X_train, y_train)
                                    st.success(f"{reg_model} trained successfully!")
                                    train_predictions = model.predict(X_train)
                                    test_predictions = model.predict(X_test)

                                    train_mse = mean_squared_error(y_train, train_predictions)
                                    test_mse = mean_squared_error(y_test, test_predictions)

                                    st.write(f"Training Mean Squared Error: {train_mse:.2f}")
                                    st.write(f"Test Mean Squared Error: {test_mse:.2f}")

                                # Save the model
                                model_filename = "model.pkl"
                                joblib.dump(model, model_filename)
                                with open(model_filename, "rb") as file:
                                    st.download_button("Download Pretrained Model", file, file_name=model_filename)

                    elif model_type == "Unsupervised":
                        task_type = st.radio("Choose an Unsupervised Task:", ["Clustering", "Dimensionality Reduction"], key="ml_unsupervised_task")
                        
                        X = data.select_dtypes(include=[np.number])  # Select numeric columns only

                        if task_type == "Clustering":
                            cluster_model = st.selectbox("Choose a Clustering Model:", ["K-Means", "DBSCAN", "Agglomerative Clustering"])
                            
                            if cluster_model == "K-Means":
                                n_clusters = st.number_input("Enter number of clusters:", min_value=2, max_value=10, value=3)
                                model = KMeans(n_clusters=n_clusters)
                            elif cluster_model == "DBSCAN":
                                eps = st.slider("Enter epsilon:", min_value=0.1, max_value=5.0, value=0.5)
                                min_samples = st.slider("Enter min_samples:", min_value=1, max_value=10, value=5)
                                model = DBSCAN(eps=eps, min_samples=min_samples)
                            elif cluster_model == "Agglomerative Clustering":
                                n_clusters = st.number_input("Enter number of clusters:", min_value=2, max_value=10, value=3)
                                model = AgglomerativeClustering(n_clusters=n_clusters)

                            model.fit(X)
                            st.success(f"{cluster_model} trained successfully!")
                            st.write(f"Cluster Labels: {model.labels_}")

                        elif task_type == "Dimensionality Reduction":
                            dr_model = st.selectbox("Choose a Dimensionality Reduction Model:", ["PCA", "t-SNE"])
                            
                            if dr_model == "PCA":
                                n_components = st.slider("Enter number of components:", min_value=2, max_value=10, value=2)
                                model = PCA(n_components=n_components)
                            elif dr_model == "t-SNE":
                                n_components = st.slider("Enter number of components:", min_value=2, max_value=10, value=2)
                                model = TSNE(n_components=n_components)

                            transformed_data = model.fit_transform(X)
                            st.success(f"{dr_model} applied successfully!")
                            st.write(f"Transformed Data Shape: {transformed_data.shape}")
                        
                        # Save the model
                        model_filename = "unsupervised_model.pkl"
                        joblib.dump(model, model_filename)
                        with open(model_filename, "rb") as file:
                            st.download_button("Download Pretrained Model", file, file_name=model_filename)

                    elif model_type == "Semi-Supervised":
                        task_type = st.radio("Choose a Semi-Supervised Task:", ["Label Propagation", "Label Spreading"], key="ml_semi_task")
                        target = st.selectbox("Select the target variable:", options=["None"] + list(data.columns), key="ml_semi_target")

                        X = data.select_dtypes(include=[np.number])  # Select numeric columns only
                        # Assuming you have partially labeled data in `data`, where some target values are missing (NaN)
                        y = data[target].copy() 
                        y[::2] = np.nan  # Simulating partial labeling for demonstration

                        if task_type == "Label Propagation":
                            model = LabelPropagation()
                        elif task_type == "Label Spreading":
                            model = LabelSpreading()

                        model.fit(X, y)
                        st.success(f"{task_type} model trained successfully!")
                        st.write(f"Predictions: {model.transduction_}")

                        # Save the model
                        model_filename = "semi_supervised_model.pkl"
                        joblib.dump(model, model_filename)
                        with open(model_filename, "rb") as file:
                            st.download_button("Download Pretrained Model", file, file_name=model_filename)

                except Exception as e:
                    st.warning(f"An error occurred: {e}")

            # Tab 6: DL Modeling
            

            # Deep Learning section
            with tabs[5]:
                st.subheader(":rainbow[Deep Learning Modeling]")
                
                # Dataset is already loaded from previous steps
                if 'data' in locals() and not data.empty:
                    try:
                        target = st.selectbox("Select the target variable:", options=["None"] + list(data.columns), key="dl_target")
                        features = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target], key="dl_features")

                        if features and target != "None":
                            # Prepare data
                            X = data[features]
                            y = data[target]

                            # Ensure y is not empty and avoid ambiguous Series comparisons
                            if y.empty:
                                raise ValueError("The target variable is empty. Please check the dataset.")

                            # Split the data into training and test sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            st.write("### Configure Neural Network")
                            
                            # Hyperparameters
                            hidden_units = st.slider("Number of Hidden Units:", 1, 256, 64)
                            epochs = st.slider("Number of Epochs:", 1, 100, 10)
                            batch_size = st.slider("Batch Size:", 1, 64, 32)
                            
                            # Select the model type
                            model_type = st.selectbox("Select Model Type:", 
                                                    ["MLP", "CNN", "RNN", "LSTM", "GRU", "Transformer", "GAN", "Autoencoder"])

                            # Build the selected model
                            if model_type == "MLP":
                                model_dl = Sequential([
                                    Dense(hidden_units, activation="relu", input_dim=X_train.shape[1]),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "CNN":
                                model_dl = Sequential([
                                    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[1], 1)),
                                    Flatten(),
                                    Dense(hidden_units, activation="relu"),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "RNN":
                                model_dl = Sequential([
                                    LSTM(hidden_units, input_shape=(X_train.shape[1], 1), return_sequences=True),
                                    Dropout(0.2),
                                    LSTM(hidden_units),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "LSTM":
                                model_dl = Sequential([
                                    LSTM(hidden_units, input_shape=(X_train.shape[1], 1)),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "GRU":
                                model_dl = Sequential([
                                    GRU(hidden_units, input_shape=(X_train.shape[1], 1)),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "Transformer":
                                model_dl = Sequential([
                                    Attention(),
                                    Dense(hidden_units, activation="relu"),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "GAN":
                                model_dl = Sequential([
                                    Dense(hidden_units, activation="relu", input_dim=X_train.shape[1]),
                                    Dropout(0.2),
                                    Dense(1, activation="sigmoid" if len(y.unique()) == 2 else "linear")
                                ])

                            elif model_type == "Autoencoder":
                                model_dl = Sequential([
                                    Dense(hidden_units, activation="relu", input_dim=X_train.shape[1]),
                                    Dense(X_train.shape[1], activation="sigmoid"),
                                    Dropout(0.2),
                                    Dense(hidden_units, activation="relu")
                                ])

                            # Compile the model
                            model_dl.compile(optimizer=Adam(), 
                                            loss="binary_crossentropy" if len(y.unique()) == 2 else "mse", 
                                            metrics=["accuracy"])

                            # Train the model
                            st.write("### Training the Model")
                            history = model_dl.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

                            st.success(f"{model_type} model trained successfully!")

                            # Training Progress - Accuracy Plot
                            st.write("### Training Progress")
                            st.line_chart(history.history["accuracy"])

                            st.write("### Download Trained Model")
                            model_file = joblib.dump(model_dl, 'dl_trained_model.pkl')
                            st.download_button("Download Trained DL Model", model_file[0], file_name="dl_trained_model.pkl")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        


    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")


st.markdown("""
---
## 📄 About Me  
Hi! I'm **Muhammad Dawood**, a data scientist passionate about Machine Learning, NLP, and building intelligent solutions.  
## Gmail: 
 muhammaddawoodmoria@gmail.com 
## whatsapp: 
 +92-370-915-2202


### 🌐 Connect with Me:  
- [GitHub](https://github.com/muhammadmoria)  
- [LinkedIn](https://www.linkedin.com/in/muhammadmoria/)  
- [Portfolio](https://muhammadmoria.github.io/portfolio-new/)  
""")
