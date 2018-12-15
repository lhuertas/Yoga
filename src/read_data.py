
import pandas as pd

# Create URL to Excel file (alternatively this can be a filepath)
url = 'https://github.com/lhuertas/Yoga/blob/master/Data/test.xlsx'

# Load the first sheet of the Excel file into a data frame
df = pd.read_excel(url, sheet_name=0, header=1)

# View the first ten rows
df.head(10)