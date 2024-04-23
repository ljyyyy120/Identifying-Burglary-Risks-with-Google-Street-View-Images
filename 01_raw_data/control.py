
import random
import pandas as pd


def generate_nyc_coordinates(num_points, existing_positions):
    coordinates = []
    unique_positions = set(existing_positions)  # Convert to set for O(1) average time complexity for checks
    attempts = 0
    max_attempts = num_points * 10  # Arbitrary limit to prevent infinite loops
    
    while len(coordinates) < num_points and attempts < max_attempts:
        latitude = random.uniform(40.477399, 40.917577)
        longitude = random.uniform(-74.25909, -73.700363)
        location = f"{latitude}, {longitude}"
        if location not in unique_positions:
            coordinates.append(location)
            unique_positions.add(location)  # Add new location to the set to ensure uniqueness
        attempts += 1
    
    if attempts == max_attempts:
        print("Reached maximum attempts to find unique locations.")
    
    return coordinates

if __name__ == "__main__":

    burglary = pd.read_csv("NYPD_Complaint_Data_Historic_BURGLARY.csv")
    burglary['position'] = burglary['Lat_Lon'].astype(str).str.replace(r"[() \[\]]", "", regex=True)

    existing_positions = list(burglary['position'].unique())
    new_coordinates = generate_nyc_coordinates(60000, existing_positions)

    # Convert list to DataFrame
    coordinates_df = pd.DataFrame(new_coordinates, columns=['Location'])

    # Save DataFrame to CSV file
    csv_file_path = 'control_coordinates.csv'
    coordinates_df.to_csv(csv_file_path, index=False)