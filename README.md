## Abstract

Road traffic collisions, though infrequent, are among the leading causes of global mortality. This study aims to forecast future traffic incidents in Wales involving a single vehicle. Data shows that single-vehicle road collisions in Wales constitute 90% of all pedestrian casualties from road incidents. Single-vehicle collisions accounted for more than 58% of pedestrian fatalities in Wales between 2015 and 2022. This highlights the importance of comprehending single-vehicle road crashes. By employing different forecasting techniques, results indicate that the FB Prophet and the XGBoost models produced the best performances with low RMSE and MAPE values. The LSTM and GRU recurrent neural network models didn't work as well as they could have on this dataset. This suggests that they need to be studied more, or that data transformation or normalisation is required before they can be used in forecasting. The performance of the statistical model, SARIMA, was moderate and did not exceed that of the machine learning models.

## 1	INTRODUCTION

<p> Road collisions are among the top ten causes of death worldwide despite them being rare events. They result in human, economic, and environmental costs. The Department for Transport (DfT) has estimated that a single road collision costs the government £105,000 using 2022 prices. The World Health Organization (WHO) estimated that road collisions' economic burden in the UK is around 0.05% of the gross domestic product (GDP).  In 2020, the United Nations General Assembly declared the Decade of Action for Road Safety 2021–2030 with the goal of reducing fatal road casualties by over 50% by 2030.  According to the [Welsh Government Road Collisions dashboard] (https://www.gov.wales/police-recorded-road-collisions-interactive-dashboard), road collisions in Wales resulted in 96 deaths and 921 serious injuries in 2022, demonstrating the severity of these incidents. </p> 
<p> Single-vehicle collisions in Wales are responsible for 90% of all pedestrian casualties from road collisions. Between 2015 and 2022, nearly three in five (58%) of pedestrian fatalities in Wales were a result of single-vehicle collisions. This highlights the significant risk pedestrians face from single-vehicle collisions. The Welsh Government in 2023 implemented a [20mph default speed limit] (https://www.gov.wales/introducing-default-20mph-speed-limits) to combat casualties people sustain as a result of road collisions in built up areas. </p> 
<p> As road casualties are a direct consequence of road collisions, the paper aims to forecast road collisions in Wales that will have a direct and positive impact on road casualty analysis by providing vital quantifiable numeric information that can be compared against the set targets. </p> 
<p> As used in this project, a single vehicle road collision occurs when a vehicle (motorised/non-motorised) collides with pedestrian, animal, geographical features, or architectural barriers, on public roads leading to injuries. </p> 


## 2	METHODS

### 2.1	BACKGROUND

<p>Time series data is a type of data that has a time component in it and is collected in a chronological order as it was observed. To be used for forecasting with statistical approaches, time series data must be stationary in levels, i.e., its mean, variance, and covariance remain constant over time. However, this requirement is not important for machine learning models or deep learning models that are able to handle non-stationarity problems in the data. </p>

### 2.2	ANALYSIS AND DESIGN 
<p> Forecasting is a form of prediction. In time series analysis, we frequently aim to forecast future events based on past observations. The importance of forecasting was demonstrated in year 2020 during Covid-19 pandemic where authorities had used the technique to impose travel restrictions as one way of containing the spread of the pandemic. The NHS uses forecasting to model winter pressures by projecting the expected number of hospital admissions and bed capacity, and by identifying the primary health care that needs more support, which has been successfully utilised in resource management, according to Public Health Wales (2019). </p>
<p>The approach used in this project involves the conversion of raw data into time series, visualising the series, and comparing different model performances in forecasting the future trajectory of road collisions. </p>

### 2.3	LEGAL, SOCIAL, ETHICAL AND PROFESSIONAL ISSUES
<p> Road collision data at the collection point contains information about people involved in road collisions. As such, this data is susceptible, as it contains personally identifiable information (PII), such as home postcodes for both drivers and casualties, vehicle registration numbers, driver's age, gender, breath analyzer tests, and results of drivers involved in a drink-and-drive collision.  To avoid causing harm and violating the UK GDPR requirements for handling sensitive data, the Department for Transport removed all PII. The only identifiable information relates to the collision location where the crash occurred. </p>
<p>An audit was conducted to get insight into the data collection, storage, and administration processes, as well as how they adhere to the principles outlined in the UK-GDPR regulations for the processing of personal information. The Department of Transport's data collection and cleaning processes, as well as the metadata associated with them, were examined. Furthermore, the data were reviewed for bias and found to be representative of the Welsh population without any protected characteristics, demographic parameters, or ethnic or sexual orientation information, postcodes, or vehicle registration numbers that the DfT removes before making the data public. Because of this, we can train the model with objective data and get objective outputs. </p>

### 2.4	DATA SOURCES

<p> Data used in this project was downloaded from [the DfT Road Safety Data website] (https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) and is publicly available under the [Open Government License v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) without any cost elements. Each row in the dataset represents single police recorded road collision. The following data files were downloaded: 

<ul>
  <li>[Road Safety Data - Vehicles 1979 - Latest Published Year](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv) (with twenty-eight features) </li>
  <li>[Road Safety Data - Collisions 1979 - Latest Published Year](https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv) (with thirty-six features) </li>
</ul>

Only data from 2010 to 2022 will be used in the project where Python programming language was utilised for all data pre-processing and exploratory data analysis (EDA). We detected outliers in our dataset for years between 2020 and 2021 because of the COVID-19 pandemic. However, these will not be discarded as they reflect an event that impacted the series, not a suspected data collection error.

Figure 1 displays the spatial distribution of all single vehicle collisions (SVC) in Wales by severity over the study period. South Wales and North Wales report the majority of SVC compared to Mid-Wales. This aids in identifying the areas where single-vehicle road collisions occur frequently.


<img src="imgs\Collision Location.png" alt="Description of Image" />
