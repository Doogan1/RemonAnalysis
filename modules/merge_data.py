import pandas as pd

def merge_county_data(
    state_survey_path, population_path, road_path, wetlands_path
):
    """
    Merges data from four CSV files on the county name column.

    Parameters:
        state_survey_path (str): Path to the 2023 biennial state survey data.
        population_path (str): Path to the population density data.
        road_path (str): Path to the road density data.
        wetlands_path (str): Path to the wetlands data.

    Returns:
        pd.DataFrame: A merged DataFrame containing data from all four sources.
    """
    # Load the CSV files
    state_survey = pd.read_csv(state_survey_path)
    population = pd.read_csv(population_path)
    road = pd.read_csv(road_path)
    wetlands = pd.read_csv(wetlands_path)

    # Ensure unique column names to avoid conflicts during merging
    state_survey = state_survey.rename(columns={"County Without Asterisks and Trimmed": "County"})
    population = population.rename(columns={"NAME *": "County", "pop_d": "pop_d"})
    road = road.rename(columns={"NAME": "County", "roadoverarea": "roadoverarea"})
    wetlands = wetlands.rename(columns={"NAME": "County", "wetlandd": "wetlandd"})

    # Log initial column information
    print("Columns in state_survey:", state_survey.columns.tolist())
    print("Columns in population:", population.columns.tolist())
    print("Columns in road:", road.columns.tolist())
    print("Columns in wetlands:", wetlands.columns.tolist())

    # Check for duplicates in the County column
    for df, name in zip([state_survey, population, road, wetlands], ["state_survey", "population", "road", "wetlands"]):
        if df["County"].duplicated().any():
            raise ValueError(f"Duplicate entries found in 'County' column of {name}.")

    # Log the number of rows in each dataset
    print(f"Number of rows in state_survey: {len(state_survey)}")
    print(f"Number of rows in population: {len(population)}")
    print(f"Number of rows in road: {len(road)}")
    print(f"Number of rows in wetlands: {len(wetlands)}")

    # Perform incremental merges
    merged_data = state_survey
    print("Starting merge with population...")
    merged_data = pd.merge(merged_data, population[["County", "pop_d"]], on="County", how="left", validate="one_to_one")
    print("Merge with population complete. Columns:", merged_data.columns.tolist())

    print("Starting merge with road...")
    merged_data = pd.merge(merged_data, road[["County", "roadoverarea"]], on="County", how="left", validate="one_to_one")
    print("Merge with road complete. Columns:", merged_data.columns.tolist())

    print("Starting merge with wetlands...")
    merged_data = pd.merge(merged_data, wetlands[["County", "wetlandd"]], on="County", how="left", validate="one_to_one")
    print("Merge with wetlands complete. Columns:", merged_data.columns.tolist())

    return merged_data