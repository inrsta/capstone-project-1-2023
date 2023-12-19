import streamlit as st
import pandas as pd
import pickle


# Load the model

with open(r"best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open(r"dict_vectorizer.pkl", "rb") as file:
    dv = pickle.load(file)

# Define the app


def main():
    
    # Add a tab name
    st.set_page_config(page_title="House Price Prediction App", layout="wide")
    st.title("House Price Prediction App")

    # Create inputs for the user to fill in the details

    size_in_feet = st.number_input("Size in feet", min_value=0.0, value=1000.0)
    no_of_bedrooms = st.number_input("No of bedrooms", min_value=0, value=1)
    no_of_bathrooms = st.number_input("No of bathrooms", min_value=0, value=1)
    year_built = st.number_input("Year built", min_value=0, value=2000)
    neighborhood = st.selectbox("Neighborhood", ["Rural", "Suburb", "Urban"])
    # Button to predict price
    if st.button("Predict Price"):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame(
            {
                "SquareFeet": [size_in_feet],
                "Bedrooms": [no_of_bedrooms],
                "Bathrooms": [no_of_bathrooms],
                "Neighborhood": [neighborhood],
                "YearBuilt": [year_built],
            }
        )

        input_dict = input_data.to_dict(orient="records")
        
        # Vectorize the input
        input_vectorized = dv.transform(input_dict)
        
        # Make the prediction
        prediction = model.predict(input_vectorized)[0]
        
        # Display the prediction
        prediction = round(prediction, 0)
        st.success(f"The predicted price is {prediction} $")

if __name__ == "__main__":
    main()
