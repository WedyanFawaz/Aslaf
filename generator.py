from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from config import (IBM_API_KEY, IBM_PROJECT_ID)
from prompt_template import  explain_tempalte, recs_tempalte



class Generator:
    def __init__(self):
       self.model = self.connect_to_model()

    def connect_to_model(self) -> Model:
        credentials = Credentials(
             url = "https://eu-de.ml.cloud.ibm.com",
             api_key = IBM_API_KEY
                  )

        parameters = {
		    "decoding_method": "greedy",
		    "max_new_tokens": 300,
		    "repetition_penalty": 1,
                	}

        model = Model(
            model_id = "sdaia/allam-1-13b-instruct",
            params = parameters,
            credentials = credentials,
            project_id = IBM_PROJECT_ID
                     )
        return model


    def get_response(self, prompt:str) -> str: 
        return self.model.generate(prompt)['results'][0]['generated_text']
    
    def get_explanation(self, response:str) -> str:
        explaination =  explain_tempalte.format(response = response)
        print(explaination)
        return self.model.generate(explaination)['results'][0]['generated_text']
    
    
    def get_recs(self, track:str) -> str:
        recs = recs_tempalte(track = track)
        return self.model.generate(recs)['results'][0]['generated_text']
