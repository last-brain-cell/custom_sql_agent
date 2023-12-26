from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine, MetaData, Table
from nltk.corpus import stopwords
from datetime import date
import pandas as pd
import pickle
import nltk
import json
import time


class CustomAgent:
    def __init__(self, database_url: str, llm, verbose: bool = True):
        self.vectorizer = TfidfVectorizer()
        with open('kneighbors_classifier.pkl', 'rb') as model_file:
            self.loaded_classifier = pickle.load(model_file)
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.metadata = MetaData()
        self.llm = llm
        self.table = None
        self.prompt = None
        self.schema = None
        self.instructions = {
            'traffic_metrics': {
                'date_of_metrics': f"(in ISO format, for any time period, mention a range of dates in ISO format if "
                                   f"date is mentioned otherwise Null. If you need to know the current date in order "
                                   f"to determine the range of dates, current date is {date.today()}),",
                'partner_region': "(Regions like EMEA, APAC, etc.),",
                'partner_country': "(in locale code format only, example - US, JP, AU, UK),",
                'partner': "(example - AT&T, KDDI, Online Docomo, Amazon, Flipkart, Saturn DE. These are vendors. Set "
                           "it to Null when no partner is mentioned or detected partner is Apple, Google, Samsung, "
                           "Xiaomi, etc since THESE ARE BRANDS AND NOT PARTNERS),",
                'Brand': "(Example - Apple, Google, Samsung, Oppo, Xiaomi, Android etc.), ",
                'visits': ",",
                'model': "(actual full official model names, example - 'Google Pixel 7 Pro', 'Apple iPhone 14', "
                         "'Samsung S21 Ultra'. If multiple Models exist, add a list a models),"
            },
            "sales_data": {
                'Week': "(in integers(example 41),",
                'Year': "(only Year in YYYY form)",
                'Partner': "(example - AT&T, KDDI, Online Docomo. These are vendors. Partners cannot be Apple or "
                           "Google.),",
                'Region Type': ",",
                'Region Breakdown': ",",
                'Product Model': "(actual full official model names, example - 'Google Pixel 7 Pro', 'Apple iPhone "
                                 "14', 'Samsung S21 Ultra'),",
                'Current Week Sales': ",",
                'Quarterly Target': ",",
                'QTD Sales': ",",
                'Current RR': ",",
                'Amended RR': ",",
                'Quarterly %': ","
            }
        }
        self.data = [
            ("What is the total Google/Apple/Samsung traffic for X partner for A B Week/Month?", 'traffic_metrics'),
            ("Which country has the highest Google weekly traffic?", 'traffic_metrics'),
            ("What is the Google Share of Traffic (SoT) for US Partners?", 'traffic_metrics'),
            ("What is the Pixel 8 traffic for Verizon for the last quarter?", 'traffic_metrics'),
            ("Which model had the highest Week-over-Week (WoW) increase in traffic for Bestbuy?", 'traffic_metrics'),
            ("What is the total Google weekly traffic for each region?", 'traffic_metrics'),
            ("Which region has the highest share of Google Traffic?", 'traffic_metrics'),
            ("Can you identify regions with a significant increase/decrease in Google Traffic compared to the "
             "previous week?", 'traffic_metrics'),
            ("Which countries contribute the most to overall online traffic?", 'traffic_metrics'),
            ("What is the total Google weekly traffic for each country?", 'traffic_metrics'),
            ("Which country has the highest Google weekly traffic across countries?", 'traffic_metrics'),
            ("What is the Share of Traffic (SoT) of Google of traffic in each country?", 'traffic_metrics'),
            ("Which country has the highest/lowest Google SoT?", 'traffic_metrics'),
            ("Are there countries where Google Traffic has decreased/increased recently?", 'traffic_metrics'),
            ("What is the total Partner traffic?", 'traffic_metrics'),
            ("What is the total Google/Apple/Samsung traffic for X partner for AB Week/Month?", 'traffic_metrics'),
            ("Which partner has the highest/lowest traffic?", 'traffic_metrics'),
            ("Which partner has the highest/lowest Google Product Detail Page (PDP) traffic?", 'traffic_metrics'),
            ("What is the total Google traffic?", 'traffic_metrics'),
            ("Which partner has the highest/lowest Google SoT?", 'traffic_metrics'),
            ("What is Google SoT for X partner?", 'traffic_metrics'),
            ("Which partner has increased/dipped in SoT/visits for Google?", 'traffic_metrics'),
            ("Comparison of absolute traffic between 2/3 brands?", 'traffic_metrics'),
            ("Comparison of SOT traffic between 2/3 brands?", 'traffic_metrics'),
            ("Which Brand has the highest absolute traffic/SoT between X Countries/Y Partners?", 'traffic_metrics'),
            ("What is the total Model Traffic for Y Region/Z Country/A Retailer?", 'traffic_metrics'),
            ("Which model has the highest traffic for Y Region/Z Country/A Retailer?", 'traffic_metrics'),
            ('What is the total Metrics for Retailer?', 'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product?', 'sales_data'),
            ('What is the total Metrics Pxl Product?', 'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product for X LocationHolder?', 'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product for X SubLocationHolder?', 'sales_data'),
            ('Which retailer had higher sales for Pixel 8 Pro?', 'sales_data'),
            ('Which n stores have the highest/lowest sales? (by default, n resorts to 10)', 'sales_data'),
            ('Which stores have zero sales?', 'sales_data'),
            ('Which store has achieved the target?', 'sales_data'),
            ('Which stores have not achieved the target?', 'sales_data'),
            (' Which stores have not achieved their target in the past 4 weeks?', 'sales_data'),
            ('What is the metric for the XYZ store?', 'sales_data'),
            ('What are the total covered stores for AT&T?', 'sales_data'),
            ('How many TSMs/Markets/ do we have for AT&T?', 'sales_data'),
            ('What was the Total RETAILER Account Level Sales?', 'sales_data'),
            ('What was the total Target for RETAILER (Online/Offline)?', 'sales_data'),
            ('What was the total Achievement % for PxlProduct for RETAILER?', 'sales_data'),
            ('What is the Run Rate (Current/Required) for Retailer?', 'sales_data'),
            ('What is the Run Rate Per Door (Current/Required) for Retailer?', 'sales_data'),
            ('What was the Total PxlProduct Sales for Timeline?', 'sales_data'),
            ('What was the Total RETAILER Account Level Sales?', 'sales_data'),
            ('What were the Sales for specific Pxl Product x ?', 'sales_data'),
            ('What was the Total Attainment % for PxlProduct for RETAILER?', 'sales_data'),
            ('What was the Delta between Forecasted and Actuals % for All Pixel Products?', 'sales_data'),
            ('What was the Delta between Forecasted and Actuals % for 1 specific Pixel Product?', 'sales_data'),
            ('What is the total Google/Apple/Samsung visits for X partner for A B Week/Month?', 'traffic_metrics'),
            ('Which country has the highest Google weekly visits?', 'traffic_metrics'),
            ('What is the Google Share of Visits (SoV) for US Partners?', 'traffic_metrics'),
            ('What is the Pixel 8 visits for Verizon for the last quarter?', 'traffic_metric'),
            ('Which model had the highest Week-over-Week (WoW) increase in visits for Bestbuy?', 'traffic_metrics'),
            ('What is the total Google weekly visits for each region?', 'traffic_metrics'),
            ('Which region has the highest share of Google Visits?', 'traffic_metrics'),
            (
                'Can you identify regions with a significant increase/decrease in Google Visits compared to the previous week?',
                'traffic_metrics'),
            ('Which countries contribute the most to overall online visits?', 'traffic_metrics'),
            ('What is the total Google weekly visits for each country?', 'traffic_metrics'),
            ('Which country has the highest Google weekly visits across countries?', 'traffic_metrics'),
            ('What is the Share of Visits (SoV) of Google of visits in each country?', 'traffic_metrics'),
            ('Which country has the highest/lowest Google SoV?', 'traffic_metrics'),
            ('Are there countries where Google Visits has decreased/increased recently?', 'traffic_metrics'),
            ('What is the total Partner visits?', 'traffic_metrics'),
            ('What is the total Google/Apple/Samsung visits for X partner for AB Week/Month?', 'traffic_metrics'),
            ('Which partner has the highest/lowest visits?', 'traffic_metrics'),
            ('Which partner has the highest/lowest Google Product Detail Page (PDP) visits?', 'traffic_metrics'),
            ('What is the total Google visits?', 'traffic_metrics'),
            ('Which partner has the highest/lowest Google SoV?', 'traffic_metrics'),
            ('What is Google SoV for X partner?', 'traffic_metrics'),
            ('Which partner has increased/dipped in SoV/visits for Google?', 'traffic_metrics'),
            ('Comparison of absolute visits between 2/3 brands?', 'traffic_metrics'),
            ('Comparison of SOV visits between 2/3 brands?', 'traffic_metrics'),
            ('Which Brand has the highest absolute visits/SoV between X Countries/Y Partners?', 'traffic_metrics'),
            ('What is the total Model Visits for Y Region/Z Country/A Retailer?', 'traffic_metrics'),
            ('Which model has the highest visits for Y Region/Z Country/A Retailer?', 'traffic_metrics'),
            (
                'What is the average Google/Apple/Samsung engagement for X partner for A B Week/Month?',
                'traffic_metrics'),
            ('Which country has the highest average Google weekly engagement?', 'traffic_metrics'),
            ('What is the Google Share of Engagement (SoE) for US Partners?', 'traffic_metrics'),
            ('What is the Pixel 8 average engagement for Verizon for the last quarter?', 'traffic_metrics'),
            ('Which model had the highest average Week-over-Week (WoW) increase in engagement for Bestbuy?',
             'traffic_metrics'),
            ('What is the average Google weekly engagement for each region?', 'traffic_metrics'),
            ('Which region has the highest average share of Google Engagement?', 'traffic_metrics'),
            (
                'Can you identify regions with a significant increase/decrease in Google Engagement compared to the previous week?',
                'traffic_metrics'),
            ('Which countries contribute the most to overall average online engagement?', 'traffic_metrics'),
            ('What is the average total Google weekly engagement for each country?', 'traffic_metrics'),
            ('Which country has the highest average Google weekly engagement across countries?', 'traffic_metrics'),
            ('What is the average Share of Engagement (SoE) of Google engagement in each country?', 'traffic_metrics'),
            ('Which country has the highest/lowest average Google SoE?', 'traffic_metrics'),
            (
                'Are there countries where average Google Engagement has decreased/increased recently?',
                'traffic_metrics'),
            ('What is the average total Partner engagement?', 'traffic_metrics'),
            ('What is the average Google/Apple/Samsung engagement for X partner for AB Week/Month?', 'traffic_metrics'),
            ('Which partner has the highest/lowest average engagement?', 'traffic_metrics'),
            ('Which partner has the highest/lowest average Google Product Detail Page (PDP) engagement?',
             'traffic_metrics'),
            ('What is the average total Google engagement?', 'traffic_metrics'),
            ('Which partner has the highest/lowest average Google SoE?', 'traffic_metrics'),
            ('What is Google SoE for X partner?', 'traffic_metrics'),
            ('Which partner has increased/dipped in SoE/engagement for Google?', 'traffic_metrics'),
            ('Comparison of average engagement between 2/3 brands?', 'traffic_metrics'),
            ('Determine the delta in visits for Pixel 8 in the United States between November and October.',
             'traffic_metrics'),
            ('Determine the delta in visits for Samsung Galaxy S21 in the UK between September and October.',
             'traffic_metrics'),
            ('Determine the delta in visits for Oppo Reno 8 in the United States between November and October.',
             'traffic_metrics'),
            ('Comparison of SOE engagement between 2/3 brands?', 'traffic_metrics'),
            ('Which Brand has the highest average engagement/SoE between X Countries/Y Partners?', 'traffic_metrics'),
            ('What is the average total Model Engagement for Y Region/Z Country/A Retailer?', 'traffic_metrics'),
            ('Which model has the highest average engagement for Y Region/Z Country/A Retailer?', 'traffic_metrics'),
            ('What is the total Metrics for Retailer for Engagement?', 'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product for Engagement?', 'sales_data'),
            ('What is the total Metrics Pxl Product for Engagement?', 'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product for X LocationHolder for Engagement?',
             'sales_data'),
            ('What is the total Metrics for Retailer for Pxl Product for X SubLocationHolder for Engagement?',
             'sales_data'),
            ('Which retailer had higher sales for Pixel 8 Pro for Engagement?', 'sales_data'),
            (
                'Which n stores have the highest/lowest sales for Engagement? (by default, n resorts to 10)',
                'sales_data'),
            ('Which stores have zero sales for Engagement?', 'sales_data'),
            ('Which store has achieved the target for Engagement?', 'sales_data'),
            ('Which stores have not achieved the target for Engagement?', 'sales_data'),
            ('Which stores have not achieved their target in the past 4 weeks for Engagement?', 'sales_data'),
            ('What is the metric for the XYZ store for Engagement?', 'sales_data'),
            ('What are the total covered stores for AT&T for Engagement?', 'sales_data'),
            ('How many TSMs/Markets/ do we have for AT&T for Engagement?', 'sales_data'),
            ('What was the Total RETAILER Account Level Sales for Engagement?', 'sales_data'),
            ('What was the total Target for RETAILER (Online/Offline) for Engagement?', 'sales_data'),
            ('What was the total Achievement % for PxlProduct for RETAILER for Engagement?', 'sales_data'),
            ('What is the Run Rate (Current/Required) for Retailer for Engagement?', 'sales_data'),
            ('What is the Run Rate Per Door (Current/Required) for Retailer for Engagement?', 'sales_data'),
            ('What was the Total PxlProduct Sales for Timeline for Engagement?', 'sales_data'),
            ('What was the Total RETAILER Account Level Sales for Engagement?', 'sales_data'),
            ('What were the Sales for specific Pxl Product x for Engagement?', 'sales_data'),
            ('What was the Total Attainment % for PxlProduct for RETAILER for Engagement?', 'sales_data'),
            (
                'What was the Delta between Forecasted and Actuals % for All Pixel Products for Engagement?',
                'sales_data'),
            ('What was the Delta between Forecasted and Actuals % for 1 specific Pixel Product for Engagement?',
             'sales_data'),
            ('What was the total Target for BestBuy?', 'sales_data'),
        ]
        self.tokens = 0
        self.cost = 0
        self.verbose = verbose
        self.stop_words = set(stopwords.words('english'))
        self.contents = None

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_table_schema_mysql(self):
        table = Table(self.table, self.metadata, autoload_with=self.engine)
        if table.columns:
            cols = [column.name for column in table.columns]
            return cols
        else:
            print(f"Table '{self.table}' not found.")
            return None

    def get_all_contents(self):
        prompt = f"Extract {' '.join([str(header + self.instructions[self.table][header]) for header in self.schema])} from the following prompt: {self.prompt}. Also extract the of the user 'intent' behind the prompt(example - Total, Average, Highest sales', Lowest, Minimum , Maximum, SoT, WoW, delta(difference) etc. DESCRIBE THE INTENT VERBOSELY and provide context). Also find the TYPE OF OUTPUT expected (Example - Single Value or List of values or a DataFrame). Return the output in JSON format and if any data is not mentioned, leave the parameter empty."

        response = self.llm.invoke(prompt)
        tokens, cost = self.calculate_and_update_cost(prompt, response)
        self.tokens += tokens
        self.cost += cost

        return response

    def generate_sql_query(self):
        schema = self.get_table_schema_mysql()
        self.schema = schema
        self.contents = self.get_all_contents()
        if schema and self.contents:
            prompt = f"write a MySQL query for the following prompt: {self.prompt}.Here are the values of parameters " \
                     f"for the extracted contents of the prompts - {str(self.contents)} for table - {self.table} for " \
                     f"column names - {schema}. The date is present in the database in an ISO format as a string and " \
                     f"not a datetime object. Give me the query in a string"
            response = self.llm.invoke(prompt)

            (tokens, cost) = self.calculate_and_update_cost(prompt, response)
            self.tokens += tokens
            self.cost += cost

            return str(response).strip('```')[7:].strip("\n")
        return None

    def run(self, user_input_prompt):
        start = time.perf_counter()

        self.set_prompt(user_input_prompt)
        self.table = self.classify_intent(user_input_prompt)
        sql_query = self.generate_sql_query()

        if sql_query:
            result_df = pd.read_sql_query(sql_query, self.engine)
            result_json = result_df.to_json(orient='records')
        else:
            result_json = "Error with executing MySQL query"

        end = time.perf_counter()
        return {
            "detected_table": self.table,
            "table_headers": self.schema,
            "detected_content": self.contents,
            "generated_query": sql_query,
            "obtained_result": result_json,
            "execution_time": f"{round(end - start):.2f}s",
            "session_usage": self.get_session_usage()
        }

    def get_session_usage(self):
        return {"tokens": self.tokens, "cost": self.cost}

    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in self.stop_words]
        return ' '.join(tokens)

    def classify_intent(self, text):
        self.vectorizer.fit([self.preprocess_text(text) for text, _ in self.data])
        text = self.preprocess_text(text)
        text_vectorized = self.vectorizer.transform([text])
        intent = self.loaded_classifier.predict(text_vectorized)[0]
        return intent

    @staticmethod
    def calculate_and_update_cost(input_prompt: str, output_response: str):
        cost_per_1000_characters = 0.0001
        total_char_usage = len(input_prompt) + len(output_response)
        total_cost = total_char_usage * cost_per_1000_characters / 1000

        try:
            with open("token_usage.json", 'r') as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

        data['total_char_usage'] = data.get('total_char_usage', 0) + total_char_usage
        data['total_cost'] = data.get('total_cost', 0) + total_cost

        with open("token_usage.json", 'w') as file:
            json.dump(data, file, indent=2)

        return total_char_usage, total_cost

#     agent.run("give me a dataframe of highest visits of Google, apple and samsung for October 2023")  # 55680.032379"
