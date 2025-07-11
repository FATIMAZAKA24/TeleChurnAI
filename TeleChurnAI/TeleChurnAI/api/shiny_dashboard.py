############################
from pathlib import Path
import pandas as pd
#app_dir = Path(__file__).parent
testingset = pd.read_csv(r"C:\Users\hppco\Desktop\FYP py\testingset.csv")
############################
import faicons as fa
import plotly.express as px
import numpy as np
# Load data and compute static values
#from shared1 import app_dir, testingset
from shiny import reactive, render
from shiny.express import input, ui
from shinywidgets import render_plotly


# Convert all column names to lowercase
testingset.columns = testingset.columns.str.lower()

tenure_rng = (min(testingset.tenure), max(testingset.tenure))
monthlycharges_rng = (min(testingset.monthlycharges), max(testingset.monthlycharges))   
totalcharnges_rng = (min(testingset.totalcharges), max(testingset.totalcharges))  

# Add page title and sidebar
#ui.page_opts(title="Customer Churn Analysis", fillable=True)
# Add a large, centered title
ui.markdown(
    "<h1 style='text-align: center; font-size: 36px; font-weight: bold;'>Customer Churn Analysis</h1>"
)
ui.markdown(
    "<style>body {background: linear-gradient(135deg, #2c003e, #4b0082, #8a2be2); background-image: url('background.png'); background-attachment: fixed; }</style>",
)
def dynamic_filters():
    filters = []
    # Handle Missing Values
    for col in testingset.columns:
        if testingset[col].isnull().sum() > 0 or (testingset[col].dtype == 'object' and testingset[col].str.strip().eq('').any()):
            if testingset[col].dtype == 'object':
                testingset[col] = testingset[col].replace(r'^\s*$', pd.NA, regex=True)
            if pd.api.types.is_numeric_dtype(testingset[col]):
                testingset[col] = testingset[col].fillna(testingset[col].mean())  # ✅ Fixed
            else:
                testingset[col] = testingset[col].fillna(testingset[col].mode()[0])  # ✅ Fixed

    # Convert Data Types
    for col in testingset.columns:
        if testingset[col].dtype == 'object':
            # Check if all values in the column are numeric (after handling possible dot-separated numbers)
            if testingset[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
                testingset[col] = pd.to_numeric(testingset[col], errors='coerce')
            elif testingset[col].nunique() > 2:  # If it's a multi-category column, convert it to categorical
                testingset[col] = testingset[col].astype('category')
            elif testingset[col].nunique() == 2:  # If it's a column with exactly 2 unique values
                testingset[col] = testingset[col].astype('category')  # Treat as categorical but don't convert to 0/1

    # Generate Filters
    for column in testingset.columns:
        if pd.api.types.is_numeric_dtype(testingset[column]):  # Numeric columns
            # Check if the column has only 0 and 1
            if set(testingset[column].unique()) == {0, 1}:  
                filters.append(
                    ui.input_checkbox_group(
                        column,
                        f"{column}",
                        [0, 1],
                        selected=list(set(testingset[column].unique())),
                        inline=True
                    )
                )
            else:
                min_val, max_val = testingset[column].min(), testingset[column].max()
                step_size = (max_val - min_val) / 100 if min_val != max_val else 1

                filters.append(
                    ui.input_slider(
                        column,
                        f"{column}",
                        min=min_val,
                        max=max_val,
                        value=[min_val, max_val],
                        step=step_size
                    )
                )
        elif pd.api.types.is_categorical_dtype(testingset[column]):  # Categorical columns
            unique_values = testingset[column].unique().tolist()

            filters.append(
                ui.input_checkbox_group(
                    column,
                    f"{column} ",
                    unique_values,
                    selected=unique_values,
                    inline=True
                )
            )
    return filters



###########################################################################
with ui.sidebar(open="desktop", style="background-color: #F5F5F5;"):
    ui.h3("Filters", style="text-align: center; margin-bottom: 10px;")  # Heading for filters

    dynamic_filters()  # Add filters here
    ui.input_action_button("reset", "Reset filter")

    # Spacer to push the upload button to the bottom
    ui.markdown("<div style='flex-grow:1;'></div>")

    # Upload button at the bottom
    ui.input_file("upload_data", "Upload CSV File", accept=[".csv"])





############################################################################
from shiny import reactive

testingset_reactive = reactive.value(testingset)  # Store as reactive value

@reactive.effect
def handle_uploaded_data():
    file_info = input.upload_data()
    if file_info:
        file_path = file_info[0]["datapath"]
        try:
            new_data = pd.read_csv(file_path)
            new_data.columns = new_data.columns.str.lower()
            testingset_reactive.set(new_data)  # Update reactive value
        except Exception as e:
            print(f"Error reading file: {e}")


# Add main content
ICONS = {
    #"total_customers": fa.icon_svg("user", "regular"),
    #"churning_customers": fa.icon_svg("user", "regular"),
    #"non-churning_customers": fa.icon_svg("user", "regular"),
    "total_customers": fa.icon_svg("users", "solid"),  # Represents all customers
    "churning_customers": fa.icon_svg("user-slash", "solid"),  # Represents lost customers (Churn)
    "non-churning_customers": fa.icon_svg("user-check", "solid"),  # Represents retained customers
    "churn_rate": fa.icon_svg("arrow-trend-down", "solid"),  # Churn rate (customers leaving)
    "retention_rate": fa.icon_svg("arrow-trend-up", "solid"),  # Retention rate (customers staying)
    "ellipsis": fa.icon_svg("ellipsis"),
    "wallet": fa.icon_svg("wallet")
}

 

## Dynamic Testing for the Dashboard ##

with ui.layout_columns(fill=True):

    with ui.value_box(showcase=ICONS["total_customers"]):

        "Total Customers"
        @render.express

        def total_customers():

            ui.markdown(str(churn_data().shape[0]))  # Dynamically render the total customer count

 

    with ui.value_box(showcase=ICONS["churning_customers"]):

        "Total Churning Customers"
        @render.express

        def churn_rate():

            d = churn_data()  # Get the filtered dataset

            ui.markdown(str(d[d["churn"] == "Yes"].shape[0]))  # Render the number of churning customers

 

    with ui.value_box(showcase=ICONS["non-churning_customers"]):

        "Loyal Customers"
        @render.express

        def loyalty_rate():

            d = churn_data()  # Get the filtered dataset

            ui.markdown(str(d[d["churn"] == "No"].shape[0]))  # Render the number of loyal customers

    

    with ui.value_box(showcase=ICONS["churn_rate"]):
        "Churn Rate"
        @render.express

        def churningcustomers_rate():
            d = churn_data()  # Get the filtered dataset
            churned_customers = d[d["churn"] == "Yes"].shape[0]
            total_customers = d.shape[0]
            # Calculate churn rate percentage
            churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
            ui.markdown(f"{churn_rate:.2f}%")  # Render churn rate percentage

    with ui.value_box(showcase=ICONS["retention_rate"]):
        "Retention Rate"
        @render.express

        def Retention_Rate():
            d = churn_data()  # Get the filtered dataset
            # Count the number of customers who have not churned
            retained_customers = d[d["churn"] == "No"].shape[0]
            total_customers = d.shape[0]
            # Calculate retention rate percentage
            retention_rate = (retained_customers / total_customers) * 100 if total_customers > 0 else 0
            ui.markdown(f"{retention_rate:.2f}%")  # Render retention rate percentage

    with ui.value_box(showcase=ICONS["wallet"]):
        "Customer Lifetime Value (CLV)"
        @render.express
        def CLV():
            d = churn_data()  # Get the filtered dataset  
            # Example calculation of CLV
            avg_monthly_revenue = d['monthlycharges'].mean()  # Replace with actual revenue column
            avg_lifetime_months = d['tenure'].mean()  # Assuming 'tenure' gives customer lifetime in months
            # Calculate CLV
            clv = avg_monthly_revenue * avg_lifetime_months
            ui.markdown(f"{clv:,.2f}")  # Render CLV value in dollar format

    


################VISUALIZATIONS####################

# Layout for Cards and Scatterplots
#############################

with ui.layout_columns(col_widths=[6,6,12]):
    with ui.card(style="width: 100%;",fill=True):

        ui.card_header("Customer Data")
        @render.data_frame
        ##table
        def table():
            # Display the filtered churn data in a table
            return render.DataGrid(churn_data())
        
        ####tenure vs churn
    with ui.card(fill=True,style="width: 100%;"):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Churn Probabilities"
            with ui.popover(title="Add a color variable"):
                ICONS["ellipsis"]
                ui.input_radio_buttons(
                    "churn_perc_y",
                    "Split by:",
                    [ "paperlessbilling", "onlinesecurity", "onlinebackup", "deviceprotection", "streamingtv", "partner", "techsupport", "streamingmovies", "seniorcitizen","dependents","internetservice", "paymentmethod", "contract"],
                    selected="paymentmethod",
                    inline=True,
            )

        @render_plotly
        def tip_perc():
            from ridgeplot import ridgeplot
            data = churn_data()
            # Ensure column exists and has no null values
            yvar = input.churn_perc_y()
            if yvar not in data.columns or data[yvar].isnull().all():
                return None  # Exit if yvar is invalid or entirely null
            # Map churn values to 0/1 for percentage calculation
            data["percent"] = data["churn"].map({"Yes": 1, "No": 0})

            # Get unique values for the chosen categorical column
            uvals = data[yvar].dropna().unique()

            # Prepare samples for ridgeplot
            samples = [data["percent"][data[yvar] == val].tolist() for val in uvals]

            # Check if samples contain empty lists (which would cause the error)
            if any(len(sample) == 0 for sample in samples):
                print("Warning: Some categories in yvar have no data.")
                return None

            # Ensure correct formatting of input
            import numpy as np
            samples = [np.array(sample) for sample in samples]

            # Create ridge plot
            plt = ridgeplot(
                samples=samples,
                labels=list(map(str, uvals)),  # Convert labels to strings
                bandwidth=0.1,
                colorscale="viridis",
                colormode="row-index",
            )

            plt.update_layout(
                title=f"Churn Probability by {yvar}",
                font=dict(size=14),  # Reduce font size
                legend=dict(
                    orientation="h", yanchor="top", y=0.98, xanchor="center", x=0.5
                ),
            )
            return plt

        
##################################################################################################################################################3
  # New Pie Chart Card for Churn Distribution
with ui.layout_columns(col_widths=[6,6,12]):  # Equal width for Pie and Line chart
    with ui.card(full_screen=True):

        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Churn Distribution"
            with ui.popover(title="Display Churn Distribution"):
                ICONS["ellipsis"]
                ui.input_radio_buttons(
                    "churn_status_split",
                    "Split by:",
                    ["paperlessbilling", "paymentmethod", "techsupport", "streamingmovies"],
                    selected="paymentmethod",
                    inline=True,
                )

        @render_plotly
        def pie_chart():
            data = churn_data()
            # Map churn values to numeric
            churn_counts = data["churn"].value_counts()

            # Create pie chart
            fig = px.pie(
                names=churn_counts.index,
                values=churn_counts.values,
                title="Overall Churn Distribution",
                labels={"Churn": "Churn Status"},
            )
            return fig
        
 
########################################################################33
# New Line Chart Card for Churn by Time Period (Tenure)
    with ui.card(full_screen=True):
        with ui.card_header(class_="d-flex justify-content-between align-items-center"):
            "Churn Over Time (Tenure)"
            with ui.popover(title="Display Churn by Time Period"):
                ICONS["ellipsis"]
                ui.input_radio_buttons(
                    "time_period_split",
                    "Split by:",
                   ["paperlessbilling", "paymentmethod", "techsupport", "streamingmovies"],
                    selected="paymentmethod",
                   inline=True,
              )

        @render_plotly
        def churn_time_series():

            data = churn_data()

            # Ensure 'tenure' is treated as continuous variable and calculate churn rate over tenure
            data["Churn_numeric"] = data["churn"].map({"Yes": 1, "No": 0}).astype(float)

            # Grouping by 'tenure' to calculate the churn rate over time
            churn_rate = data.groupby("tenure",observed=False)["Churn_numeric"].mean().reset_index()

            # Create line chart for churn over time (tenure)
            fig = px.line(
                churn_rate,
                x="tenure",
                y="Churn_numeric",
                title="Churn Rate Over Time (Tenure)",
                labels={"Churn_numeric": "Churn Rate", "tenure": "Tenure (Months)"},
            )
            return fig
###############################################################################################################################
#####################
# New Bar Chart Card with Dropdown Filters for Feature Selection and Churn Comparison
with ui.card(full_screen=True):

    with ui.card_header(class_="d-flex justify-content-between align-items-center"):
        "Feature vs Churn Distribution"
        with ui.popover(title="Select Feature to Visualize"):
            ICONS["ellipsis"]
            ui.input_select(
                "feature_selection",
                "Choose a feature:",
                choices=[  # Corrected from 'options' to 'choices'
                        "paperlessbilling",
                        "onlinesecurity",
                        "onlinebackup",
                        "deviceprotection",
                        "streamingTtv",
                        "partner",
                        "techsupport",
                        "streamingmovies",
                        "seniorcitizen",
                        "dependents",
                        "internetservice",
                        "contract",
                        "paymentmethod"
                ],
                selected="paperlessbilling",  # Default selected feature
            )

    @render_plotly
    def bar_chart():
        data = churn_data()
        selected_feature = input.feature_selection()  # Get the selected feature
        # Group by selected feature and churn, then reset the index
        feature_churn_counts = data.groupby([selected_feature, "churn"], observed=False).size().reset_index(name="Count")

        # Create bar chart
        fig = px.bar(
            feature_churn_counts,
            x=selected_feature,
            y="Count",
            color="churn",  # Color bars by Churn status
            title=f"Churn Distribution by {selected_feature}",
            labels={selected_feature: f"{selected_feature} Value", "Count": "Customer Count"},
            barmode="group",  # Group the bars by Churn status
        )

        return fig
######################################################################################################33
###paymnent method and monthly charges and total charges.

#ui.include_css(app_dir / "styles.css")
# Reactive calculations and effects
##############################################
##testing##
##############################################


@reactive.calc
def churn_data():
        # Get input from the tenure slider (input.tenure() returns a tuple, which defines the tenure range)
        tenure_range = input.tenure()  # Assumed to be a tuple (min_tenure, max_tenure)
        monthlycharges_range = input.monthlycharges()
        totalcharges_range = input.totalcharges()
        idx1 = testingset.tenure.between(tenure_range[0], tenure_range[1])  # Filter by tenure range
        idx2 = testingset.contract.isin(input.contract())  # Filter by selected contract types
        idx3 = testingset.paymentmethod.isin(input.paymentmethod())  # Filter by selected contract types
        idx4 = testingset.partner.isin(input.partner())  # Filter by partner
        idx5 = testingset.dependents.isin(input.dependents())  # Filter by dependents
        idx6 = testingset.paperlessbilling.isin(input.paperlessbilling())  # Filter by paperlessbilling
        idx7 = testingset.streamingmovies.isin(input.streamingmovies())  # Filter by streamingmovies
        #issue with seniorcitizen 
        #idx8 = testingset.seniorcitizen.isin(input.seniorcitizen())  # Filter by seniorcitizen
        idx8 = testingset.seniorcitizen.isin(map(int, input.seniorcitizen()))  # Convert input to integer before filtering

        idx9 = testingset.internetservice.isin(input.internetservice())  # Filter by internetservice
        idx10 = testingset.onlinesecurity.isin(input.onlinesecurity())  # Filter by onlinesecurity
        idx11 = testingset.onlinebackup.isin(input.onlinebackup())  # Filter by onlinebackup
        idx12 = testingset.deviceprotection.isin(input.deviceprotection())  # Filter by deviceprotection
        idx13 = testingset.techsupport.isin(input.techsupport())  # Filter by techsupport
        idx14 = testingset.streamingtv.isin(input.streamingtv())  # Filter by streamingtv
        idx15 = testingset.churn.isin(input.churn())  # Filter by churn
        idx16 = testingset.monthlycharges.between(monthlycharges_range[0], monthlycharges_range[1])  # Filter by tenure range
        idx17 = testingset.totalcharges.between(totalcharges_range[0], totalcharges_range[1])  # Filter by tenure range


        combined_idx = idx1 & idx2 & idx3 & idx4 & idx5 & idx6 & idx7 & idx8 & idx9 & idx10 & idx11 & idx12 & idx13 & idx14 & idx15 & idx16 & idx17
        # Return the filtered DataFrame
        return testingset_reactive()[combined_idx]



#############################################
###################FILTER RESET###############
@reactive.effect
###################FILTER RESET
@reactive.event(input.reset)
def _():
    # Reset filters when the reset button is pressed
    ui.update_slider("tenure", value=tenure_rng)
    ui.update_slider("monthlycharges", value=monthlycharges_rng)
    ui.update_slider("totalcharges", value=totalcharnges_rng)

    # Dynamically reset checkbox groups using unique values from dataset
    categorical_columns = [
        "partner", "dependents", "paperlessbilling", "streamingmovies", "seniorcitizen",
        "internetservice", "onlinesecurity", "onlinebackup", "deviceprotection",
        "techsupport", "streamingtv", "contract", "paymentmethod", "churn"
    ]

    for col in categorical_columns:
        unique_values = testingset[col].unique().tolist()  # Extract unique values
        ui.update_checkbox_group(col, selected=unique_values)  # Update dynamically