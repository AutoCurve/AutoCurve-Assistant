# AutoCurve Assistant

AutoCurve Assistant is a Streamlit-based vehicle valuation tool that combines historical market data, statistical price estimation, image-based condition analysis, and online discussion lookup to estimate the value range of a used vehicle.

The project is designed to help users get a more informed used-car estimate by considering both structured vehicle details, such as make, model, year, odometer, fuel type, transmission, and drivetrain, and unstructured vehicle images that are analyzed by an AI vision model.

## Overview

Used vehicle pricing is often inconsistent because two vehicles with the same year and model can have very different values depending on mileage, condition, trim-related attributes, and visible wear. AutoCurve Assistant attempts to solve this by combining multiple signals into one valuation workflow.

The application allows a user to:

1. Select a vehicle manufacturer and model from the dataset.
2. Enter the vehicle year and odometer reading.
3. Choose optional filters such as fuel type, transmission, and drive type.
4. Upload vehicle images for AI-based condition analysis.
5. Generate an estimated vehicle value.
6. View similar vehicles from the dataset.
7. See a market price graph.
8. Review online discussion snippets related to the selected vehicle.

## Features

- Interactive Streamlit web interface
- Vehicle selection by manufacturer and model
- User input for year, odometer, fuel type, transmission type, and drivetrain
- Image upload support for vehicle condition analysis
- AI-based condition classification using OpenRouter-compatible chat completion models
- Condition score adjustment based on visible vehicle defects
- Statistical valuation using historical vehicle data
- Price trend visualization with Plotly
- Comparable vehicle table
- Reddit-based social proof lookup using DuckDuckGo Search
- Simple in-process rate limiter for OpenRouter API calls

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- Plotly
- Pillow
- python-dotenv
- OpenAI Python client with OpenRouter base URL
- DuckDuckGo Search

## Project Structure

```text
AutoCurve-Assistant/
│
├── AutoCurve/
│   ├── backend.py
│   ├── frontend.py
│   ├── rate_limiter.py
│   ├── database.xlsx
│   └── .gitignore
│
├── README.md
├── .gitattributes
└── .DS_Store
```

## How It Works

AutoCurve Assistant has two main parts:

```text
Frontend: Streamlit user interface
Backend: valuation logic, image analysis, data loading, and search utilities
```

### 1. Data Loading

The backend loads the vehicle dataset from `database.xlsx` using pandas. During loading, key text columns are normalized by stripping whitespace and converting values to lowercase. This helps avoid mismatches when filtering by manufacturer, model, condition, fuel type, transmission, drive type, title status, and state.

### 2. User Input

The frontend displays input controls for the user to provide vehicle details:

- Manufacturer
- Model
- Year
- Odometer reading
- Fuel type
- Transmission type
- Drive type
- Vehicle images

The manufacturer and model options are generated from the dataset, which prevents users from selecting vehicles that are not available in the stored data.

### 3. Image-Based Condition Analysis

The user uploads one or more vehicle images. The app sends up to four images to an OpenRouter-compatible vision model through the OpenAI Python client.

The model is prompted to classify the vehicle condition as one of:

```text
new
like new
excellent
good
fair
salvage
```

The model also returns:

- reasoning
- visible defects
- condition score

The condition score is used as a multiplier in the final valuation.

### 4. Vehicle Valuation

The backend estimates vehicle value using a weighted combination of several pricing signals.

The valuation logic considers:

- average price for the same make and model
- price trend based on year
- price trend based on odometer reading
- similar vehicles with matching fuel, transmission, and drive type
- similar vehicles with the same condition
- AI-generated condition score from uploaded images

The final price is calculated by combining category-based price, condition-based price, year-based regression price, odometer-based regression price, and the condition multiplier.

### 5. Market Visualization

The app generates a Plotly scatter plot showing price versus odometer for similar vehicles. A trendline is included to help users understand how mileage relates to market price for the selected make and model.

### 6. Comparable Listings

The app displays a small table of similar vehicles from the dataset, including:

- year
- model
- odometer
- price
- condition
- fuel type
- transmission
- drive type

This helps users see the examples behind the estimate instead of only seeing a single number.

### 7. Online Discussion Lookup

The app uses DuckDuckGo Search to look for Reddit discussions related to the selected vehicle. The query focuses on common problems and reliability for the chosen year, manufacturer, and model.

This gives the user extra context beyond the dataset, such as common owner complaints or reliability discussions.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AutoCurve/AutoCurve-Assistant.git
cd AutoCurve-Assistant/AutoCurve
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file has not been added yet, install the main dependencies manually:

```bash
pip install streamlit pandas scikit-learn plotly pillow python-dotenv openai duckduckgo-search openpyxl
```

### 4. Add Environment Variables

Create a `.env` file inside the `AutoCurve/` folder:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The app uses OpenRouter through the OpenAI Python client, so the API key is required for image-based condition analysis.

### 5. Run the App

```bash
streamlit run frontend.py
```

## Required Dataset

The app expects a file named:

```text
database.xlsx
```

inside the `AutoCurve/` directory.

The dataset should include at least the following columns:

```text
manufacturer
model
year
price
odometer
```

Additional supported columns include:

```text
condition
fuel
transmission
drive
title_status
state
```

These columns improve filtering and valuation quality.

## Environment Variables

| Variable | Description |
| --- | --- |
| `OPENROUTER_API_KEY` | API key used to call the OpenRouter vision model for vehicle image analysis |

## Current Limitations

This project is a valuation assistant, not a guaranteed appraisal tool. The estimate depends heavily on the quality, size, and coverage of the dataset.

Current limitations include:

- The valuation model uses relatively simple regression and weighted pricing logic.
- The estimate may be weak when the dataset has very few examples for a selected make and model.
- The image analysis depends on the quality and angle of uploaded photos.
- The AI vision model may miss damage that is not visible in the images.
- The current system does not verify VIN, trim, accident history, location-based pricing, or live marketplace listings.
- The current rate limiter is in-process, so it is best suited for single-instance deployments.

## Future Improvements

Potential improvements include:

- Add a proper saved model pipeline using joblib or pickle.
- Add stronger model evaluation metrics.
- Compare multiple models such as Random Forest, Gradient Boosting, and XGBoost.
- Add live marketplace scraping or API integration.
- Add VIN decoding.
- Include trim-level vehicle matching.
- Add location-based price adjustments.
- Add confidence intervals or estimated price ranges.
- Improve error handling and logging.
- Add unit tests for backend valuation functions.
- Remove development-only debug captions from the frontend.
- Deploy the app on Streamlit Community Cloud or another hosting platform.
- Add screenshots and a live demo link to the README.

## Suggested Requirements File

```text
streamlit
pandas
scikit-learn
plotly
pillow
python-dotenv
openai
duckduckgo-search
openpyxl
```

## Disclaimer

AutoCurve Assistant provides an estimated used-vehicle value based on available data, statistical trends, and AI-assisted image analysis. It should be used as an informational tool only. Final vehicle value may vary based on market conditions, vehicle history, trim, location, mechanical condition, accident records, and professional inspection results.
