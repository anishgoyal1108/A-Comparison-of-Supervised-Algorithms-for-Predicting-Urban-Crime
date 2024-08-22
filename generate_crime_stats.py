import pandas as pd
from typing import Tuple


# Write a combined DataFrame to CSV
def write_combined_csv(filename: str, combined_data: pd.DataFrame) -> None:
    combined_data.to_csv(filename, index=False)


# Count occurrences by new crime categories
def categorize_counts(date: str, data: pd.DataFrame) -> Tuple[int, int, int, int]:
    category1 = len(
        data[
            (data["Date"] == date)
            & (
                data["Crime Type"].isin([
                    "LARCENY-FROM VEHICLE",
                    "LARCENY-NON VEHICLE",
                ])
            )
        ]
    )

    category2 = len(
        data[
            (data["Date"] == date)
            & (
                data["Crime Type"].isin([
                    "BURGLARY",
                    "AUTO THEFT",
                ])
            )
        ]
    )

    category3 = len(
        data[
            (data["Date"] == date)
            & (
                data["Crime Type"].isin([
                    "AGG ASSAULT",
                    "ROBBERY",
                ])
            )
        ]
    )

    category4 = len(
        data[(data["Date"] == date) & (data["Crime Type"].isin(["HOMICIDE"]))]
    )

    return category1, category2, category3, category4


def filter_year(year: str) -> pd.DataFrame:
    # Load dataset and select relevant columns
    data = pd.read_csv("crime_2010_2020.csv", low_memory=False)
    selected_columns = [
        "Occur Date",
        "Crime Type",
        "Neighborhood",
        "Longitude",
        "Latitude",
    ]
    filtered_data = data[selected_columns].copy()
    filtered_data.rename(columns={"Occur Date": "Date"}, inplace=True)

    # Convert 'Date' column to datetime format
    filtered_data["Date"] = pd.to_datetime(filtered_data["Date"], format="%m/%d/%Y")

    # Create a new 'Year' column
    filtered_data["Year"] = filtered_data["Date"].dt.year

    # Apply geographic filters for longitude and latitude
    filtered_data = filtered_data[
        (filtered_data["Longitude"] >= -84.5)
        & (filtered_data["Longitude"] <= -84.2)
        & (filtered_data["Latitude"] >= 33.61)
        & (filtered_data["Latitude"] <= 33.92)
    ]

    # Filter rows based on the user-inputted year
    filtered_data = filtered_data[filtered_data["Year"] == int(year)]

    # Drop rows with missing 'Neighborhood' values
    filtered_data.dropna(subset=["Neighborhood"], inplace=True)

    # Format the 'Date' column to only show the date part (no time)
    filtered_data["Date"] = filtered_data["Date"].dt.strftime("%Y-%m-%d")

    # Sort the filtered data by 'Date'
    filtered_data.sort_values(by="Date", inplace=True)

    return filtered_data


def get_date_counts(
    filtered_data: pd.DataFrame,
) -> pd.DataFrame:
    # Get the unique dates for which occurrences will be counted
    unique_dates = filtered_data["Date"].unique()

    # List to store the counts for each date
    records = []

    for date in unique_dates:
        counts = categorize_counts(date, filtered_data)
        records.append([date] + list(counts))

    # Create a DataFrame from the records
    date_counts_df = pd.DataFrame(
        records,
        columns=["Date", "Category 1", "Category 2", "Category 3", "Category 4"],
    )

    return date_counts_df


if __name__ == "__main__":
    combined_data = pd.DataFrame(
        columns=["Date", "Category 1", "Category 2", "Category 3", "Category 4"]
    )

    for year in range(2010, 2021):
        year = str(year)
        filtered_data = filter_year(year)
        date_counts_df = get_date_counts(filtered_data)

        # Append the data for this year to the combined data
        combined_data = pd.concat([combined_data, date_counts_df], ignore_index=True)

    # Write the combined data to a CSV file
    write_combined_csv("combined_crime_cats.csv", combined_data)
