import geopandas as gpd
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting script to calculate wetland area by county.")

# Load the wetlands shapefile
logging.info("Loading wetlands shapefile.")
wetlands = gpd.read_file('National_Wetland_Inventory_(NWI)_2005.shp')
logging.info("Wetlands shapefile loaded successfully.")

# Load the county boundaries shapefile
logging.info("Loading county boundaries shapefile.")
counties = gpd.read_file('County.shp')
logging.info("County boundaries shapefile loaded successfully.")

# Ensure both GeoDataFrames use the same coordinate reference system (CRS)
logging.info("Ensuring coordinate reference systems match.")
wetlands = wetlands.to_crs(counties.crs)
logging.info("CRS matching completed.")

# Perform a spatial join to associate wetlands with their respective counties
logging.info("Performing spatial join to associate wetlands with counties.")
wetlands_in_counties = gpd.sjoin(wetlands, counties, how='inner', predicate='intersects')
logging.info("Spatial join completed.")

# Calculate the area of wetlands within each county
logging.info("Calculating area of wetlands within each county.")
wetlands_in_counties['wetland_area'] = wetlands_in_counties['geometry'].area
logging.info("Wetland areas calculated.")

# Group by county and sum the wetland areas
logging.info("Grouping by county and summing wetland areas.")
wetland_area_by_county = wetlands_in_counties.groupby('Name')['wetland_area'].sum().reset_index()
logging.info("Grouping and summation completed.")

# Merge the wetland area data with the county data
logging.info("Merging wetland area data with county data.")
result = wetland_area_by_county.merge(counties[['Name']], on='Name')
logging.info("Merge completed.")

# Load existing CSV data
logging.info("Loading existing Data.csv file.")
csv_data = pd.read_csv('Data.csv')
logging.info("Data.csv loaded successfully.")

# Drop the last column of the CSV data
logging.info("Dropping the last column of Data.csv.")
csv_data = csv_data.iloc[:, :-1]
logging.info("Last column dropped.")

# Merge the wetland area data with the CSV data
logging.info("Merging wetland area data with Data.csv.")
csv_data = csv_data.merge(result[['Name', 'wetland_area']], 
                          how='left', left_on='StateProgressReport2021_County', right_on='Name')
logging.info("Merge with Data.csv completed.")

# Drop the duplicate county name column
logging.info("Dropping duplicate county name column.")
csv_data.drop(columns=['Name'], inplace=True)

# Save the updated DataFrame back to CSV
logging.info("Saving updated Data.csv.")
csv_data.to_csv('Data.csv', index=False)
logging.info("Data.csv updated successfully with wetland areas.")

print("Updated Data.csv with wetland areas.")
