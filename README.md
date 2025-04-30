# Financial Stock Platform

## Overview

The Financial Stock Platform is a comprehensive web application for stock analysis, built with Streamlit and Python. It provides users with detailed financial analysis, news sentiment, market positioning, institutional activity tracking, and risk assessment for stocks.

## Architecture

The application uses a modular architecture with these main components:

```
financial-stock-platform/
├── streamlit_app.py             # Main Streamlit frontend application
├── backend/                     # Backend services and agents
│   ├── agent_system/            # Analysis agent modules
│   │   ├── agent_types.py       # Type definitions for agents
│   │   ├── event_bus.py         # Event handling system
│   │   └── agents/              # Individual analysis agents
│   │       ├── base_agent.py    # Base agent class
│   │       ├── institutional_activity_agent.py
│   │       ├── market_position_agent.py
│   │       ├── news_sentiment_agent.py
│   │       ├── risk_assessment_agent.py
│   │       └── wrapper_methods.py
│   └── data_services/           # Data retrieval services
│       ├── data_connector.py    # Main data access coordinator
│       ├── stock_server.py      # FastAPI server for stock data
│       ├── alpha_vantage_client.py  # Alpha Vantage API client
│       ├── sentiment_analyzer.py    # Text sentiment analysis
│       ├── stock_client.py      # Stock data client
│       ├── supabase_client.py   # Database client
│       └── tavily_scraper.py    # News scraping service
├── config/                      # Configuration
│   └── config.py                # Configuration loading
├── tests/                       # Test utilities
│   └── mock_data_provider.py    # Mock data for testing/dev
├── .env                         # Environment variables
└── requirements.txt             # Dependencies
```

## How It Works

### Data Flow

1. **User Interaction**: Users input stock symbols and select analysis types through the Streamlit UI.

2. **Request Processing**:
   - The Streamlit app (`streamlit_app.py`) handles user input and delegates to appropriate analysis functions.
   - Depending on the selection, specific analysis agents are invoked.

3. **Agent System**:
   - Each agent (`news_sentiment_agent.py`, `market_position_agent.py`, etc.) specializes in a specific type of analysis.
   - Agents process data and generate insights, visualizations, and recommendations.

4. **Data Services**:
   - The `data_connector.py` serves as a central coordinator for all data requests.
   - It determines whether to use real or mock data based on configuration.
   - For real data, it communicates with the stock server and other APIs.
   - For mock/testing, it uses the `mock_data_provider.py`.

5. **Stock Server**:
   - The `stock_server.py` is a FastAPI application that provides financial data.
   - It handles requests for historical prices, financial metrics, and company information.
   - Aggregates data from various sources, including Alpha Vantage or other providers.

6. **Results Display**:
   - Data and analysis results are returned to the Streamlit frontend.
   - The UI displays visualizations, metrics, and insights to the user.

### Mock Mode vs. Real Mode

The application can operate in two modes:

- **Real Mode**: Connects to actual financial APIs and retrieves live data
- **Mock Mode**: Uses the `mock_data_provider.py` to generate realistic but synthetic data, useful for development, testing, or when APIs are unavailable

### Key Components

#### Streamlit Frontend (`streamlit_app.py`)

- Manages UI/UX and user interaction
- Creates visualizations with Plotly and Streamlit components
- Handles navigation between different analysis views
- Provides fallback mechanisms when data isn't available

#### Agent System

- **News Sentiment Agent**: Analyzes news articles and sentiment
- **Market Position Agent**: Evaluates a stock's position relative to peers and the market
- **Institutional Activity Agent**: Tracks institutional ownership and movements
- **Risk Assessment Agent**: Calculates risk metrics and generates risk profiles

#### Data Services

- **Data Connector**: Central hub for all data requests
- **Stock Server**: FastAPI server providing stock data
- **API Clients**: Interfaces with external services (Alpha Vantage, Tavily, etc.)
- **Supabase Client**: Handles database operations

## Environment Variables

The application uses environment variables for configuration, stored in the `.env` file:

```
# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key      # For stock price history, financials, and technical indicators
FINANCIAL_API_KEY=your_financialdatasets_api_key      # For additional financial data (financialdatasets.ai)
TAVILY_API_KEY=your_tavily_api_key                    # For news and web research
OPENAI_API_KEY=your_openai_api_key                    # For NLP analysis and insights
SEC_API_KEY=your_sec_api_key                          # For SEC filings and institutional data

# Server Configuration  
STOCK_SERVER_HOST=localhost
STOCK_SERVER_PORT=8000

# Supabase Configuration (for database storage)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Application Mode
MOCK_MODE=True  # Set to False to use real APIs
```

To use real APIs, you need to:
1. Set `MOCK_MODE=False`
2. Provide valid API keys for the services you want to use
3. Ensure the stock server is running (started automatically by the app)

## Stock Server Communication

When the application needs financial data, it follows this process:

1. The Streamlit app or an agent makes a data request through `data_connector.py`
2. The data connector checks if MOCK_MODE is enabled:
   - If TRUE: It gets data from `mock_data_provider.py`
   - If FALSE: It routes the request to the appropriate real data source

3. For real data:
   - The data connector communicates with `stock_server.py` via HTTP
   - The stock server processes the request, calling external APIs if needed
   - Data is returned, formatted according to the application's standards
   - Results are cached when appropriate to reduce API calls

4. Data is then passed to the requesting component (agent or UI)

## Getting the Code

You can get the Financial Stock Platform codebase in one of these ways:

### Option 1: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/username/financial-stock-platform.git

# Or using SSH
git clone git@github.com:username/financial-stock-platform.git
```

### Option 2: Fork the Repository (Recommended for Contributors)

1. Navigate to the [Financial Stock Platform repository](https://github.com/username/financial-stock-platform) on GitHub
2. Click the "Fork" button in the top-right corner
3. Select your GitHub account as the destination for the fork
4. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/financial-stock-platform.git
   ```

### Option 3: Download as ZIP

1. Navigate to the [Financial Stock Platform repository](https://github.com/username/financial-stock-platform) on GitHub
2. Click the "Code" button and select "Download ZIP"
3. Extract the ZIP file to your desired location

## Setup and Running

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your `.env` file with appropriate API keys (copy from `.env.example`)

3. **Set up the Supabase database**:

   You can set up the Supabase database tables in two ways:
   
   **Option 1: Using the Python setup script (Recommended)**
   ```
   python setup_supabase.py
   ```
   This will automatically create all required tables in your Supabase project.
   
   **Option 2: Manually using the SQL Editor**
   - Copy the contents of `supabase_setup.sql`
   - Go to your Supabase project dashboard
   - Navigate to the SQL Editor
   - Paste the SQL script and run it

4. **Starting the Stock Server**:
   
   While the Streamlit app can auto-start the stock server, it's often better to run it separately especially for development:
   ```
   # In terminal #1
   python -m backend.data_services.stock_server

   The stock server will start on http://localhost:8000 by default. You can verify it's running by accessing http://localhost:8000/docs in your browser.

5. Run the Streamlit application:
   ```
   # In terminal #2
   streamlit run streamlit_app.py
   ```

5. The application will automatically:
   - Check for API keys and determine if MOCK_MODE should be used
   - Connect to the stock server if running in real mode (or start it if not already running)
   - Initialize all necessary components
   - Present the UI for analysis

## API Integration by Component

The platform uses several financial APIs for different purposes:

| Component | API | Usage |
|-----------|-----|-------|
| Stock Server | Alpha Vantage | Stock prices, technical indicators, financials, company overview |
| Stock Server | SEC API | SEC filings, institutional ownership data |
| Stock Server | Financial Datasets API | Extended financial datasets, alternate data sources |
| News Sentiment Agent | Tavily API | News articles, web research, sentiment data |
| News Sentiment Agent | OpenAI API | Natural language processing for sentiment analysis |
| Institutional Activity Agent | SEC API | Institutional holdings, ownership changes |
| Risk Assessment Agent | Alpha Vantage | Historical price data for volatility calculations |
| Risk Assessment Agent | SEC API | Risk-related filings and disclosures |

When an API is unavailable or rate-limited, the system will automatically fall back to the mock data provider.

## Development

For development:
- Use MOCK_MODE=True to avoid consuming API quotas
- The mock data provider generates realistic, consistent test data
- Edit agent logic in the respective agent files
- Modify UI elements in streamlit_app.py
- Run the stock server separately to more easily monitor API requests

## Troubleshooting

- If experiencing issues with real data, check API keys and quotas
- Verify the stock server is running when in real mode
- For missing data, the application will fall back to mock data or display appropriate messages

## Step-by-Step Process Flow

Here's a detailed walkthrough of how data flows through the system when a user enters a stock ticker:

### 1. User Input & Request Initialization

1. **User enters ticker** in the Streamlit interface (`streamlit_app.py`)
   - Input captured in the sidebar: `symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")`

2. **Streamlit app processes input**
   - Validates the ticker symbol format
   - Updates session state to track the current stock: `st.session_state.current_symbol = symbol`

3. **User selects analysis type**
   - For example, clicking on the "Risk Assessment" tab
   - Tab selection handled in `streamlit_app.py`: `agent_tabs = st.tabs(["News Sentiment", "Market Position", "Institutional Activity", "Risk Assessment"])`

### 2. Data Request Process

4. **Streamlit app initiates data request**
   - For risk assessment: `risk_data = risk_assessment_agent.assess_risk(symbol)`

5. **Agent receives request** (`backend/agent_system/agents/risk_assessment_agent.py`)
   - The `assess_risk()` method prepares to gather required data

6. **Data connector invoked** (`backend/data_services/data_connector.py`)
   - Agent checks MOCK_MODE flag to determine data source
   - If MOCK_MODE=False: Makes HTTP request to the stock server
   - If MOCK_MODE=True: Uses mock_data_provider for synthetic data

### 3. Real Data Retrieval (when MOCK_MODE=False)

7. **Stock server receives request** (`backend/data_services/stock_server.py`)
   - FastAPI endpoint handles the specific data request type
   - For risk data: `/api/v1/risk/{ticker}` endpoint is called

8. **Stock server fetches external data**
   - Makes API calls to services like Alpha Vantage for price history
   - Calls SEC API for institutional data
   - Retrieves any cached data from local storage to minimize API calls

9. **Data processing and calculations**
   - Stock server calculates metrics like volatility, drawdown, risk scores
   - Formats results into standardized response format

10. **Response returned to data connector**
    - JSON response with success/failure status and formatted data

### 4. Agent Processing

11. **Agent processes raw data**
    - Receives data from data connector
    - Performs additional analysis and calculations
    - For risk assessment: calculates overall risk score, categorizes risk levels

12. **Agent generates insights**
    - Creates written analysis of the data
    - For risk assessment: Generates volatility statements, risk insights, recommendations

13. **Agent prepares visualization data**
    - Formats data for charts, tables, and other UI components
    - Creates color-coding based on metrics (e.g., red for high risk)

### 5. UI Rendering

14. **Streamlit app receives processed data**
    - Agent returns completed analysis to the main app

15. **Display functions render the UI** (`streamlit_app.py`)
    - For risk assessment: `display_risk_assessment(risk_data)`
    - Creates visual components: risk gauge, metrics display, charts
    - Formats and displays textual analysis

16. **Interactive elements activated**
    - Expandable sections, tooltips, and interactive charts become available
    - User can interact with the displayed analysis

### 6. Caching and Optimization

17. **Results cached for performance**
    - Recent queries stored in memory to reduce redundant API calls
    - Session state preserves analysis between tab switches

18. **Background refresh (optional)**
    - If enabled, periodic refresh of volatile data (e.g., price, news)
    - Uses Streamlit's background jobs for non-blocking updates

This entire process typically completes in a few seconds when using cached or mock data, or 5-10 seconds when fetching fresh data from external APIs.
