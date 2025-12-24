# Interactive EDA Dashboards

Static plots are great for analysis, but stakeholders prefer interactive dashboards. This guide covers building web-based EDA tools.

---

## When to Use This
- Sharing EDA with non-technical teams
- Executive dashboards
- Internal data exploration tools
- Customer-facing analytics
- Real-time monitoring

---

## 1. Streamlit (Fastest to Build)

**Best for:** Quickly turning Python scripts into web apps

### Basic App Structure
```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("📊 Customer Churn Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    contract_type = st.sidebar.multiselect(
        "Contract Type",
        options=df['Contract'].unique(),
        default=df['Contract'].unique()
    )
    
    # Filter data
    df_filtered = df[df['Contract'].isin(contract_type)]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df_filtered))
    col2.metric("Churn Rate", f"{df_filtered['Churn'].mean():.1%}")
    col3.metric("Avg Monthly Charges", f"${df_filtered['MonthlyCharges'].mean():.2f}")
    
    # Interactive plot
    fig = px.histogram(df_filtered, x='tenure', color='Churn', 
                      title='Tenure Distribution by Churn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Raw Data")
    st.dataframe(df_filtered.head(100))
```

**Run:** `streamlit run app.py`

### Advanced Features

**1. Caching (Speed up repeated operations)**
```python
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

df = load_data('data.csv')  # Only loads once
```

**2. Session State (Maintain state across interactions)**
```python
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button('Increment'):
    st.session_state.counter += 1

st.write(f"Count: {st.session_state.counter}")
```

**3. Tabs**
```python
tab1, tab2, tab3 = st.tabs(["Overview", "Deep Dive", "Export"])

with tab1:
    st.write("Summary statistics")
    
with tab2:
    st.write("Detailed analysis")
```

---

## 2. Plotly Dash (More Control)

**Best for:** Production-grade dashboards with complex callbacks

### Basic Dashboard
```python
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.read_csv('data.csv')

app.layout = html.Div([
    html.H1("Customer Churn Dashboard"),
    
    # Dropdown
    dcc.Dropdown(
        id='contract-dropdown',
        options=[{'label': i, 'value': i} for i in df['Contract'].unique()],
        value='Month-to-month',
        multi=False
    ),
    
    # Graph
    dcc.Graph(id='churn-graph')
])

# Callback (update graph when dropdown changes)
@app.callback(
    Output('churn-graph', 'figure'),
    Input('contract-dropdown', 'value')
)
def update_graph(selected_contract):
    filtered_df = df[df['Contract'] == selected_contract]
    fig = px.histogram(filtered_df, x='tenure', color='Churn')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

### Advanced Callbacks

**Chained Callbacks (Dependent dropdowns)**
```python
@app.callback(
    Output('city-dropdown', 'options'),
    Input('state-dropdown', 'value')
)
def update_city_dropdown(selected_state):
    filtered_cities = df[df['State'] == selected_state]['City'].unique()
    return [{'label': i, 'value': i} for i in filtered_cities]
```

---

## 3. Panel/HoloViz (Flexible)

**Best for:** Jupyter-based dashboards

```python
import panel as pn
import pandas as pd
import holoviews as hv
from holoviews import opts

pn.extension()

df = pd.read_csv('data.csv')

# Widgets
contract_selector = pn.widgets.Select(
    name='Contract Type',
    options=list(df['Contract'].unique())
)

# Function to create plot
@pn.depends(contract_selector.param.value)
def create_plot(contract):
    filtered = df[df['Contract'] == contract]
    return hv.Histogram(filtered['tenure'].values, label=contract)

# Dashboard
dashboard = pn.Column(
    "# Churn Analysis",
    contract_selector,
    create_plot
)

dashboard.show()
```

---

## 4. State Management in Streamlit

Standard Streamlit scripts run from top to bottom on every interaction. **Session State** lets you remember data.

```python
# Check if key exists
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Update state
if st.button("Log In"):
    st.session_state.logged_in = True
    st.rerun()

# Use state
if st.session_state.logged_in:
    st.write("Welcome to the secure dashboard!")
```

---

## 5. Custom Styling (CSS)

```python
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
```

---

## 6. Authentication & Security

1. **Streamlit Authenticator:** Simple YAML-based auth for Small teams.
2. **SSO/OAuth:** For Enterprise (requires Streamlit Cloud or custom deployment).

---

## 7. Best Practices for Production

- **Logging:** Use Python's `logging` to track errors in the dashboard.
- **Profiling:** Use `st.write(st.session_state)` during development to see current state.
- **Environment Variables:** Use `st.secrets` or `.env` files for DB passwords.

---

## 8. Dashboard vs BI Tool (Tableau/PowerBI)

| Feature | Python Dashboard (Streamlit/Dash) | BI Tool (Tableau/PowerBI) |
|---------|-----------------------|-----------------------|
| **Logic** | Full Python Power (ML, NLP, Scipy) | Restricted Formula Language |
| **Data Source** | Any (API, DB, Scraper) | Most standard DBs |
| **Version Control**| Git / Code-based | Proprietary XML / Binary |
| **Customization** | Unlimited (CSS/JS) | Restricted to UI options |
| **Cost** | Free / Compute cost | Per-user License fees |

**Conclusion:** Use **BI Tools** for standard reporting. Use **Python Dashboards** when you need custom ML logic or complex data transformations.

---

## Example: Complete Streamlit EDA App

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="EDA Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('telco_churn.csv')

df = load_data()

# Sidebar
st.sidebar.title("⚙️ Controls")
numeric_cols = df.select_dtypes(include='number').columns
selected_col = st.sidebar.selectbox("Select Feature", numeric_cols)

# Main content
st.title("📊 Automated EDA Dashboard")

# Row 1: Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Missing", df.isnull().sum().sum())
col4.metric("Duplicates", df.duplicated().sum())

# Row 2: Distribution plot
st.subheader(f"Distribution of {selected_col}")
fig = px.histogram(df, x=selected_col, marginal="box")
st.plotly_chart(fig, use_container_width=True)

# Row 3: Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    corr = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
```

---

## Checklist for Dashboard Development

- [ ] Define target audience (technical vs non-technical)
- [ ] Choose framework (Streamlit for speed, Dash for production)
- [ ] Cache expensive operations
- [ ] Add filters and interactivity
- [ ] Test on mobile (responsive design)
- [ ] Deploy to cloud (Streamlit Cloud, Heroku, or Docker)
- [ ] Secure secrets (environment variables)
- [ ] Add documentation/tooltips for users
