import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

from utils.data import read_data
from utils.data import *
from utils.data import predict_model

#powerbi_path = 'Tax default prediction.pbix'
#model_path     = 'taxdefault_model.pkl'

import streamlit.components.v1 as components


# Set Page config

st.set_page_config(
    page_title="Tax Default Prediction Tool", 
    page_icon="💶", 
    layout="wide"
)

# menu bar
with st.sidebar:
    selected = option_menu(None, ["Home", "EDA", "Interactive Dashboard", "Prediction", "Interpretation"],
    icons     =['house', 'cloud-upload', "list-task", 'gear'],
    menu_icon ="cast", 
    default_index=0, 
    orientation="vertical",

    styles={
    "icon"              : {"color": "orange", "font-size": "16px"},
    "nav-link"          : {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected" : {"background-color": "blue"},
    })









############################## Home Page  #################################################    
###########################################################################################
def main():
    if selected == "Home":

        st.title("Tax Default Prediction Tool")

        st.markdown("### Background")

        with st.container():
        # Logo section
         st.image('Tax to GDP ratio.png')  

        

        # Information
        tax_to_gdp_avg = 15.6
        kenya_tax_to_gdp = 15.2
        difference = 0.4

        # Display information using Markdown
        st.markdown("In numerous developing nations, the percentage of tax collection in relation to their Gross Domestic Product (GDP) sits between 15-20% (UNDP, 2022).")
        st.markdown("Kenya’s tax-to-GDP ratio in 2021 stood at **{}%**, which was marginally below the average of **{}%** for the 33 African countries, with a difference of **{}** percentage points.".format(kenya_tax_to_gdp, tax_to_gdp_avg, difference))


    
        st.markdown("### Problem Statement")

        st.markdown("""
        In today's dynamic tax landscape, ensuring tax compliance is crucial for sustaining government 
        revenue collection. However, a persistent challenge lies in the inability to predict and prevent 
        instances of late or non-payment of taxes, leading to an annual loss of Ksh. 20 billion in Kenya. 
        The current manual monitoring system lacks the capacity for early intervention, relying on 
        reactive measures after taxpayers default. With a staff member on average monitoring 1,000 
        taxpayers and limited resources for proactive engagement, the need for a predictive model to 
        identify defaulter taxpayers in real-time is evident. Implementing machine learning (ML) 
        algorithms to anticipate non-compliance behaviors and facilitate targeted interventions such as 
        enforcement actions, tax audits, and taxpayer education is imperative to minimize revenue loss 
        and optimize resource allocation for effective tax administration.            
        """)

        st.markdown("### Objectives")

        st.markdown("""
        The primary goal of this research was to predict risky taxpayers in real-time using a ML model
        thereby enhancing the audit case selection process. Specifically, the following objectives are 
        explored;

        1.	To identify features that are important for predicting tax evasion.
        
        2.	To apply, test and evaluate a ML model for case selection in Kenya.
        
        3.	To deploy the ML model using a web application to predict taxpayer risks for case selection.
        """)            

  

if __name__ == "__main__":
    main()

   

############################## EDA Page  #################################################    
###########################################################################################
if selected == 'EDA':
    st.header(":orange[Tax Default Prediction Tool Exploratory Analysis]",divider=True)

    
    with st.container():
        # Logo section
         st.image('Payment default by region.png')

    with st.container():
        # Logo section
         st.image('Payment default by Station.png')

    with st.container():
        # Logo section
         st.image('Payment default by Sector.png')

    with st.container():
        # Logo section
         st.image('Distribution of payers.png')


############################## Data Analysis  #################################################    
###########################################################################################
if selected == 'Interactive Dashboard':
    st.header(":orange[Tax Default Prediction Tool Exploratory Analysis]",divider=True)
    

    renderer = get_pyg_renderer()
    renderer.render_explore()





############################## Prediction Page  #################################################    
#################################################################################################
if selected == "Prediction":

    st.header(":orange[Prediction Page]",divider=True)
    st.markdown(" ")

    # Declare sector options
    sector_options = ('AGRICULTURE, FORESTRY AND FISHING', 'CONSTRUCTION', 'DIGITAL ECONOMY', 'EDUCATION', 'ENERGY',
                      'FINANCIAL AND INSURANCE ACTIVITIES', 'INFORMATION AND COMMUNICATION', 'MANUFACTURING',
                      'MINING AND QUARRYING', 'Other', 'REAL ESTATE ACTIVITIES', 'SERVICE ACTIVITIES',
                      'TRANSPORTATION AND STORAGE', 'WHOLESALE AND RETAIL TRADE')

    st.text("Please fill out the form below to receive a prediction of the taxpayer's default status.")
    with st.form('my_form',border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gross_turnover = st.number_input("Gross Turnover", value=0)
        with col2:
            total_expenses        = st.number_input("Total Expenses",value=30000)
        with col3:
            net_profit           = st.number_input("Net Profit", value=-40000)

        col4, col5,col6 = st.columns(3)
        with col4:
            capital_allowance = st.number_input("Capital Allowance", value=0)
        with col5:
            installment_tax_paid = st.number_input("Installment Tax Paid", value=0)  
        with col6:
            total_sales = st.number_input("Total Sales", value=3017.24)

        col7, col8,col9 = st.columns(3)
        with col7:
            total_purchases          = st.number_input("Total Purchases", value=0)
        with col8:
            less_crdt_bal_prv_mnth = st.number_input("Previous Month Credit Balance", value=7877323.808)
        with col9:
            WVAT_credit = st.number_input("WVAT_credit", value=0)
            
        col10, col11,col12 = st.columns(3)
        with col10:
            VAT_advance_paid = st.number_input("VAT Advance Paid", value=0)
        with col11:
            no_of_employees = st.number_input("Number of Employees", value=0)    
        with col12:
            adm_exp_tot_expenses = st.number_input("Administrative Expenses to Total Expenses", value=0)

        col13, col14,col15 = st.columns(3)
        with col13:
            current_ratio = st.number_input("Current Ratio", value=0.554972804)
        with col14:
            quick_ratio = st.number_input(label="Quick Ratio", value=0.554972804)
        with col15:
            perc_zero_exempt_totalsales          = st.number_input("Exempt and Zero Sales to Total Sales", value=0)

        col16, col17,col18 = st.columns(3)
        with col16:
            profit_margin = st.number_input("Profit Margin", value=0)
        with col17:
            Gearing_ratio = st.number_input("Gearing Ratio", value=58.82527188)
        with col18:
            VAT_input_output = st.number_input("VAT Input Output", value=0)

        col19, col20, col21 = st.columns(3)
        with col19:
            Effective_tax_rate = st.number_input("Effective Tax Rate", value=-2110.429011)
        with col20:
            debt_to_sales_ratio = st.number_input("Debt to Sales Ratio", value=6238.843115)
        with col21:
            total_liabilities = st.number_input("Total Liabilities", value=25152116)

        col22, col23 = st.columns(2)
        with col22:
            Age = st.number_input("Age", value=6) 
        with col23:
            sector = st.selectbox('Select Sector', sector_options)

        submitted = st.form_submit_button(label='GET TAX DEFAULT PREDICTION')

    if submitted:
        # check if all paramaters have been checked
        list_of_params  = [gross_turnover, total_expenses, net_profit, capital_allowance, installment_tax_paid,
                           total_sales, total_purchases, installment_tax_paid, less_crdt_bal_prv_mnth,
                           WVAT_credit, VAT_advance_paid, no_of_employees, adm_exp_tot_expenses, 
                           current_ratio, quick_ratio, perc_zero_exempt_totalsales, profit_margin, Gearing_ratio, 
                           VAT_advance_paid,VAT_input_output, Effective_tax_rate, debt_to_sales_ratio, total_liabilities,
                           Age, sector]

    


            # create a dataframe
        user_data = {
        'gross_turnover': gross_turnover, 'total_expenses': total_expenses, 'net_profit': net_profit,
        'capital_allowance': capital_allowance, 'installment_tax_paid': installment_tax_paid, 'total_sales': total_sales,
        'total_purchases': total_purchases, 'less_crdt_bal_prv_mnth': less_crdt_bal_prv_mnth,
        'WVAT - credit': WVAT_credit, 'VAT_advance_paid': VAT_advance_paid,
        'no_of_employees': no_of_employees, 'adm_exp_tot_expenses': adm_exp_tot_expenses, 'current_ratio': current_ratio,
        'quick_ratio': quick_ratio, '%_zero_exempt_totalsales': perc_zero_exempt_totalsales, 'profit_margin': profit_margin,
        'Gearing_ratio': Gearing_ratio, 'VAT_input_output': VAT_input_output, 'Effective_tax_rate': Effective_tax_rate,
        'debt_to_sales_ratio': debt_to_sales_ratio, 'total_liabilities': total_liabilities, 'Age': Age,
                }
        user_sector = sector

        # Add sector to user data
        for sector in ['AGRICULTURE, FORESTRY AND FISHING', 'CONSTRUCTION', 'DIGITAL ECONOMY', 'EDUCATION', 'ENERGY',
                   'FINANCIAL AND INSURANCE ACTIVITIES', 'INFORMATION AND COMMUNICATION', 'MANUFACTURING',
                   'MINING AND QUARRYING', 'Other', 'REAL ESTATE ACTIVITIES', 'SERVICE ACTIVITIES',
                   'TRANSPORTATION AND STORAGE', 'WHOLESALE AND RETAIL TRADE']:
            user_data[f'sector_{sector}'] = 1 if sector == user_sector else 0

                # Initialize empty data frame
        columns = ['gross_turnover', 'total_expenses', 'net_profit', 'capital_allowance', 'installment_tax_paid', 'total_sales',
                      'total_purchases', 'less_crdt_bal_prv_mnth', 'WVAT - credit', 'VAT_advance_paid',
                      'no_of_employees', 'adm_exp_tot_expenses', 'current_ratio', 'quick_ratio', '%_zero_exempt_totalsales',
                      'profit_margin', 'Gearing_ratio', 'VAT_input_output', 'Effective_tax_rate', 'debt_to_sales_ratio',
                      'total_liabilities', 'Age', 'sector_AGRICULTURE, FORESTRY AND FISHING',
                      'sector_CONSTRUCTION', 'sector_DIGITAL ECONOMY', 'sector_EDUCATION', 'sector_ENERGY',
                      'sector_FINANCIAL AND INSURANCE ACTIVITIES', 'sector_INFORMATION AND COMMUNICATION', 'sector_MANUFACTURING',
                      'sector_MINING AND QUARRYING', 'sector_Other', 'sector_REAL ESTATE ACTIVITIES', 'sector_SERVICE ACTIVITIES',
                      'sector_TRANSPORTATION AND STORAGE', 'sector_WHOLESALE AND RETAIL TRADE']

        df = pd.DataFrame(columns=columns)

            # Append the user inputs to the DataFrame
        df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
            # st.dataframe(data)

        # Call the RF model to make predictions on the user input

        #if user_data:
        #    st.success(predict_model(df))

            # st.write(predict_model(data))
        st.markdown("#### Predictions Results")
       
        predict_results = predict_model(df)

        st.warning("Model Prediction:")
        st.success(model_category_using_y_preds(predict_results['prediction'][0]))
        st.markdown(':grey[Probability] ')
        st.info(round((predict_results['probability'])* 100,2))


############################## Interpretation Page  #################################################    
###########################################################################################
if selected == 'Interpretation':
    data   = read_data()
    st.markdown("### :orange[Explainable AI]")

    data_instance = st.sidebar.selectbox("Select a Data Instance",options=data.index.to_list())
    st.data_editor(data,use_container_width=-True,height=250)
    st.markdown('👈Please select Data Instance')

    if data_instance:
        data_picked=data.loc[[data_instance]]
        st.write('Data Instance Selected')
        st.data_editor(data_picked, use_container_width=True)

        on = st.toggle("Show Interpretability")
        if on:
            with st.container(border=True):
                components.html(lime_explainer(read_data(),12), height=800, width=900, scrolling=True)

    
# Assuming you have an image file named "image.jpg" in the same directory as your script
image = "tax.webp"

# Display the image in the sidebar
st.sidebar.image(image, caption='Your Image Caption', use_column_width=True)





