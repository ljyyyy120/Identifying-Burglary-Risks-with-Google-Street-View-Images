import aiohttp
import asyncio
import os
import pandas as pd
import random

async def fetch_street_view(session, position, key, save_path):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': '600x300',
        'location': position,
        'fov': '90',
        'heading': '0.0',
        'pitch':'0.0',
        'key': key
    }
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, f"{position.replace(',', '_').replace(' ', '')}.jpg")
    
    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Check for HTTP errors.
            with open(file_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"File saved successfully at {file_path}.")
    except aiohttp.ClientError as e:
        print(f"An error occurred: {e}")

async def main(position_set, key, save_path):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_street_view(session, position, key, save_path) for position in position_set]
        await asyncio.gather(*tasks)




# Running the async main function
if __name__ == "__main__":

    #burglary = pd.read_csv("NYPD_Complaint_Data_Historic_BURGLARY.csv")
    #burglary['position'] = burglary['Lat_Lon'].astype(str).str.replace(r"[() \[\]]", "", regex=True)
    
    # filter the burglary data
    #sample = burglary.loc[(burglary['BORO_NM'] == 'STATEN ISLAND')]

    location = pd.read_csv('control_coordinates.csv')
    position_set = list(location['Location'].unique())

    # position_set = list(sample['position'].unique())
    key = ""
    save_path = "06_no_burglary"
    asyncio.run(main(position_set, key, save_path))