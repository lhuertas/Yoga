import boto3
import json
import pandas as pd
creds_df = pd.read_csv("C:/workspace/varios/review.csv", sep=',')
ACCESS_ID = creds_df.iloc[1,1]
ACCESS_KEY = creds_df.iloc[2,1]

client = boto3.client(
    service_name='comprehend',
    aws_access_key_id=ACCESS_ID,
    aws_secret_access_key=ACCESS_KEY,
    region_name='eu-west-1')

text = "Te has portado muy mal hoy"

print('Calling DetectSentiment')
print(json.dumps(client.detect_sentiment(Text=text, LanguageCode='es'), sort_keys=True, indent=4))
print('End of DetectSentiment\n')

