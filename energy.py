import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


#dframe = pd.read_csv("eco2mix-regional-cons-def.csv", sep=";")

st.sidebar.title("Renewable Energy in France")
st.sidebar.write("## Menu")

pages = ["Project Context","Datasets","Analysis", "ML Methodology", "Predictions", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

st.sidebar.image("gline.png")

st.sidebar.title("Team Members")
st.sidebar.markdown("**Nataly Ortiz Botero, [LinkedIn](https://www.linkedin.com/in/natalyortizbotero/)**")
st.sidebar.markdown("**Piotr Psiuk, [LinkedIn](https://www.linkedin.com/in/piotr-psiuk-17629911a/)**")
st.sidebar.markdown("**Max Barantandikiye, [LinkedIn](https://www.linkedin.com/in/barantandikiye-max-bryan/)**")

st.sidebar.write("**:green[Data Analyst Bootcamp - February 2023 Promotion]**")

if page==pages[0]:
    st.title("Renewable Energy in France")
    st.image("renewable_img.png")

    st.write("## Objective")

    st.markdown("The purpose of this project is to analyse the energy consumption and production in France, "
             "as well as the prediction of renewable energy production according to the weather conditions.")

    st.write("## Context")
    st.markdown("With this report we want to provide a comprehensive overview of the energy project in Python "
             "that analyzes the energy consumption and production data in France during the last 10 years.")
    st.markdown("The project is designed to study the different sources of energy production, consumption, and "
             "distribution in different regions of France.")
    st.markdown(" After an in-depth analysis of the data, in the second part of the project for the Machine Learning "
            "analysis, a forecast of the production of renewable energy (Solar and Wind) was carried out based on weather data.")

elif page==pages[1]:
    st.write("## Datasets")

    tab1, tab2 = st.tabs(["Principal", "Secondary"])

    with tab1:
        st.header("Principal Dataset")

        st.subheader("1. Data Source")
        st.markdown("The data source is that of the ODRE [Open Data Energy Networks](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature&sort=-date_heure)")

        st.subheader("2. Time Span")
        st.markdown("We have access to all consumption and production information by sector, day by day "
                    "(every 1/2 hour) since January 2013 until May 2022.")

        st.subheader("3. Data Analysis")
        st.markdown("The data used in the project was collected from the French electricity grid RTE's Open Data platform. "
                    "The dataset **eco2mix-regional-cons-def** contains information about energy consumption and production in "
                    "different regions of France. It includes details on the type of energy being consumed (e.g. coal, natural "
                    "gas, nuclear, etc.), the region where the energy is being consumed, and the date and time of the consumption. "
                    "The data is sorted by region, date and time in descending order. The data consists of the energy consumption "
                    "and production of different regions of France from 2013 to 2022, which includes the date and hour, the INSEE "
                    "region code, the region, the source of energy production, and the energy consumption in megawatts (MW). "
                    "The data also includes usage of energy source (TCO%) and percentage of total energy produced from each "
                    "energy source (TCH%). The data is sorted by region, date and time in descending order.")
        st.subheader("4. Data Import and Preparation")
        st.markdown("The data was loaded into the project using the **pandas** library in Python. "
                    "In order to increase readability and make it easier to work with, the columns were renamed from"
                    " French to English. Additionally, it was discovered that the first observation for every region had "
                    "missing information about energy production and consumption. To address this issue, the missing values "
                    "were replaced with the values from the next time period (30 minutes later).")
        st.subheader("5. Data Exploration")
        st.markdown("The data was explored to get an idea of its structure and nature. The shape of the data was found "
                    "to be (1.980.288, 32) and the first 20 records were displayed to get a preview of the data. "
                    "The data was then explored to see if there were any missing values and to get an idea of "
                    "the modality of the columns.")
        st.write("DataFrame Extract:")

        st.image("df.png")

        st.subheader("6. Data Processing")
        st.markdown("**Missing values and columns dropped**")
        st.markdown("The data was found to have several missing values, and the columns *Column30* "
                    "and the *TCH* and *TCO* columns were found to be redundant and were dropped as they may distort the prediction "
                    "results. The columns of *storage*, *destocking*, and *physical exchange* of energy were also dropped as they"
                    " are not in the scope of the project. 17 columns were dropped.")
        st.markdown(" Most of the missing values disappeared as they belonged to columns "
                    "that were dropped. The rest of the missing values were replaced by *zeros* as we are implying that there is no production.")

        if st.checkbox("Show missing values"):
            st.image("mv.png")

        st.markdown("**Adding New Variables**")
        st.markdown(
            """
            We created 8 variables for better analysis, so we had 20 variables in total:
            - **Wind_MW**: As the wind data is collected for offshore and onshore only since the beginning of 2021, we considered in our further analysis only the whole wind energy production. We concatenated columns *Wind_MW, Offshore_wind and Onshore_wind*.
            - **Production**: We concatenated the production variables *Thermal_MW, Nuclear_MW, Wind_MW, Solar_MW, Hydraulic_MW, Pumping_MW, Bioenergies_MW*.
            - **Renewable_MW**: We concatenated the Renewable energy variables *Bioenergies_MW, Wind_MW, Solar_MW, Hydraulic_MW*.
            - **Time Variables**: *Year, Quarter, Month, Day, DoW*.
            """)

        st.write("**Final Dataset**")
        st.write(pd.DataFrame({
            'Column Name': ['INSEE_Region_code', 'Region', 'Date', 'Hour', 'Date_hour', 'Consumption_MW', 'Thermal_MW',
                            'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Hydraulic_MW', 'Pumping_MW', 'Bioenergies_MW', 'Production',
                            'Renewable_MW', 'Year', 'Month', 'Quarter', 'DoW', 'Day'],
            'Type': ["int64","object", "datetime64[ns]", "object", "object","float64", "float64", "float64", "float64","float64",
                     "float64", "float64", "float64", "float64", "float64", "int64", "int64", "int64", "int64", "int64" ],
        }))

    with tab2:
        st.header("Secondary Dataset")
        st.markdown("We used this secondary dataset for the predictions (second) part of our project.")
        st.subheader("1. Data Source")
        st.markdown(
            "We used an additional data source for weather characteristics. It comes from [OpenDataSoft](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/?sort=date) - "
            "a cloud-based data publishing and sharing platform.")

        st.subheader("2. Time Span")
        st.markdown("We have *historical weather observation from France (SYNOP)* from 2009 until 2023.")

        st.subheader("3. Data Analysis")
        st.markdown("The dataset **donnees-synop-essentielles-omm** is a dataset that contains essential synoptic data"
                    " from the World Meteorological Organization (WMO). The data includes information on temperature, "
                    "pressure, humidity, wind speed, and other meteorological variables, collected at various weather "
                    "stations. The data is organized by date and the records are sorted by date in ascending order.")
        st.subheader("4. Data Import and Preparation")
        st.markdown("The data may be useful for research and analysis related to weather patterns and climate changes. "
                    "We load only the following columns for purposes of machine learning: *date, region, region code, wind speed, "
                    "wind direction and temperature*.")
        st.subheader("5. Data Exploration")
        st.markdown("This data set has 2.211.569 records but not all of them were used. This data set was merged with the principal "
                    "dataset on the date, region, and region code.")

        df1 = pd.read_csv("meteo.csv", sep=",")
        st.write("DataFrame Extract:")

        st.dataframe(df1.head(10))

        st.subheader("6. Data Processing")
        st.markdown("**Missing values**")
        st.markdown("The data from the variables that were used, had insignificant percentage of missing values (0.04%). "
                    "They will be replaced by the mean.")
        st.markdown("**Adding New Variables**")
        st.markdown(
            """
            There will be 3 new variables used: 
            - Wind_dir
            - Wind_speed
            - Temperature
            """)


elif page==pages[2]:
    st.write("## Analysis")

    st.subheader("Energy Consumption and Production")
    st.markdown("The data from the principal source was used to study the energy consumption and production in France as "
                "a whole and in the different regions of France. We created some visuals to analyse the seasonality of energy "
                "consumption, the consumption vs. production by Region, and lastly we made a focus in Renewable Energies.")
    st.markdown("The data was also used to study the correlation between different sources of energy production and consumption. "
                "The data showed that there was a positive correlation between wind energy production and solar energy production, "
                "and a negative correlation between wind energy production and nuclear energy production.")

    tab1, tab2, tab3 = st.tabs(["Seasonality", "Regions", "Renewables"])

    with tab1:
        st.subheader("Seasonality of Energy Consumption")
        options = ['Year', 'Month', 'Day', 'Hour']

        selected_option = st.radio("Select a time unit:", options)

        if selected_option == 'Year':
            st.image("energyyear.png")
            st.markdown(
                "Energy consumption has a big decrease in 2020 most probably because of the covid lockdowns "
                "and companies being closed. It starts going up again in 2021 and 2022. We should still point out that for 2022"
                " we only have data until 31st of May.")
        elif selected_option == 'Month':
            st.image("energymonth.png")
            st.markdown(
                "Energy consumption is higher in colder months (Winter), and its highest are January "
                "and February as they are usually the coldest months of the year.")
        elif selected_option == 'Day':
            st.image("energyday.png")
            st.markdown(
                "The energy consumption decreases on the weekend. We can potentially say that this "
                "trend is because offices, company warehouses, and such are closed.")
        elif selected_option == 'Hour':
            st.image("energyhour.png")
            st.markdown("Energy consumption decreases during the night, regularly the time we are sleeping.")

    with tab2:
        st.subheader("Regional Analysis")
        st.markdown("**Consumption by Region**")
        st.image("regions.png")
        st.markdown("The two regions that have the biggest consumption of energy are *Île-de-France* and "
                    "*Auvergne-Rhône-Alpes* as in this two regions there are the two major cities "
                    "of agglomeration by population, with Paris (10,858,874 inhabitants) and Lyon (1,693,159 inhabitants).")

        st.markdown("**Consumption vs. Production by Region**")
        st.image("avregions1.png")
        st.image("avregions2.png")
        st.image("avregions3.png")
        st.markdown("There are 4 regions that have a high production of energy. Theee regions "
                    "are: *Auvergne-Rhône-Alpes*, *Centre-Val de loire*, *Grand Est* and *Normandie* and indeed these regions"
                    "produce 70% of the total energy in France.")

    with tab3:
        st.subheader("Renewable Energy Analysis")
        st.markdown("For this project our main focus was to understand how France is producing renewable energies, in which"
                    "regions is the highest production and how this is changing over the years.")
        st.markdown("**Energy production by type**")
        st.image("shares.png")
        st.markdown("The primary energy production in France is *Nuclear* but we can observe a slow increase "
                    "in the share of renewable energies and a decrease in the share of nuclear "
                    "energy through the years.")
        st.markdown("**Nuclear and Renewable Energies by Region**")
        st.write(":blue[**Nuclear Energy**]")
        st.image("nuclearprod.png")
        st.markdown("As the primary energy production in France is *Nuclear* we analysed its production by Region with the "
                    "2021 (the most recent full data in our dataset) and the regions that produced the most Nuclear Energy "
                    "are: *Auvergne-Rhone-Alpes*, *Centre-Val de Loire*, *Normandie* and *Grand Est* with a total of "
                    "77% of the country production.")
        st.write(":green[**Renewable Energies**]")
        st.image("renewableprod.png")
        st.markdown( "The regions that produced the most Renewable Energy are *Auvergne-Rhone-Alpes*, *Grand Est*"
                     " and *Occitanie* with an aggregate of 55% of the country production.")
        st.markdown("**Renewable energies production vs. consumption**")
        st.image("quaterly.png")
        st.markdown("The gap between production of green energy and total consumption is very high. Nonetheless the "
                    "energy production from renewable sources is slowly growing through the years.")



elif page == pages[3]:
    st.write("## Machine Learning Methodology")
    st.markdown("Initially our idea was to predict renewable energy (as the target variable) but when trying to predict with "
                "all the different models, we were having as a result, a perfect score of 1.0. This was simply because it was the sum of other"
                " key variables in the dataset. This is why we decided to add additional data to our dataset, so we can "
                "continue with the second part of our project thanks to Machine Learning methods.")
    st.markdown("We decided to add weather data for the same period of time, to be able to predict and have as target "
                "variables: *Wind production* and *Solar production*. "
                "To have accurate prediction results, it is very helpful to use weather data as it is assumed that the "
                "wind and the sun have the greatest impact on the production of energy from these sources. We decided to "
                "add more columns that describe the wind and sun parameters in a given region at a given time.")
    st.markdown("**Chosen Variables and Dataset Merge**")
    st.markdown("""
    - ***Dropped variables***: From the principal dataset of 20 variables, we dropped 2 variables that were created by us : *Production* and *Renewable_MW* as they will no longer be needed for this ML part.
    - ***Adding New Variables***: From the additional data set we selected the following variables: *date, region, region code, wind speed, wind direction and temperature*.
    - ***Datasets Merge***: The weather dataset was merged with the principal dataset on the date, region, and region code using an inner join method. 3 new variables were created: *Wind_dir, Wind_speed* and *Temperature*.
    - ***Dropped variables after the merge***: *Region, Date* and *Date_Hour* columns were dropped as we could use the Region Code(ID) instead, and for the Date columns we used the seasonal variables (Year, Month, Day, Hour) instead for better performance. We will analyse *18 variables*.
     """)
    st.markdown("**Final Dataframe for ML Analysis:**")
    df = pd.read_csv("new_clean.csv", index_col=0)
    st.dataframe(df.head())

    options = ["Show Summary", 'Show Dimensions']
    selected_option = st.radio("Select:", options)

    if selected_option == 'Show Summary':
        st.write("Summary:", df.describe())

    elif selected_option == 'Show Dimensions':
        st.write("Dimensions:", df.shape)

elif page == pages[4]:
    st.write("## Predictions")

    st.markdown("With our data we want to predict Wind energy production and Solar energy production, depending on weather conditions. "
                "This is a typical problem of regression, as all the features and targets are quantitative. We will compare the "
                "prediction results separately for two targets - Wind and Solar energy producion, in 4 models: *Linear Regression, "
                "Decision Tree, Lasso* and *Random Forest*.")

    df = pd.read_csv("new_clean.csv", index_col=0)
    data1 = df.drop("Wind_MW", axis=1)
    target1 = df["Wind_MW"]
    data2 = df.drop("Solar_MW", axis=1)
    target2 = df["Solar_MW"]

    target_choice = st.selectbox(label = "Target Selection", options = ["Wind_MW", "Solar_MW"])
    model_choice = st.selectbox(label="Model Selection",
                                options=["Linear Regression", "Decision Tree", "Lasso", "Random Forest"])

    def train_model(model_choice, target_choice, size=0.8):
        if target_choice == "Wind_MW":
            data = data1
            target = target1
        elif target_choice == "Solar_MW":
            data = data2
            target = target2
        X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=False, train_size=size)
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor()
        elif model_choice == 'Lasso':
            model = Lasso()
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score1 = model.score(X_train, y_train)
        score2 = model.score(X_test, y_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return score1, score2, r2, mae, mse

    st.write('Score on train set:', round(train_model(model_choice, target_choice)[0],3), "/", "Score on test set:", round(train_model(model_choice, target_choice)[1],3))
    st.write('**R2** on test set:', round(train_model(model_choice, target_choice)[2],3))
    st.write('**MAE** on test set:', round(train_model(model_choice, target_choice)[3],3))
    st.write('**RMSE** on test set:', round(np.sqrt(train_model(model_choice, target_choice)[4]),3))

elif page == pages[5]:
    st.write("## Conclusion")
    st.markdown("In conclusion, we can observe that France is making a transition to Renewable Energies throught the last years and will "
                "continue to grow during the coming years. One of the reason is also because France is one of the first european countries "
                "imposing laws to fight climate change and reduce carbon emmissions thanks to the use of sustainable energies.")
    st.markdown("Regarding the results of our Machine Learning model predictions, we can demonstrate the effectiveness of "
                "Decision tree and Random forest classification in the prediction of Wind and Solar energy production. With "
                "high R2 scores, low MAE, RMSE, and very good scores on the test data sets.")
    st.markdown("Overall, this study provides evidence that Decision tree and Random Forest classification are a viable "
                "solution for predicting Solar and Wind energy production based on weather data, and can be a useful tool "
                "in making task-related decisions or actions.")
