import os
import sys
import nest_asyncio
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply() # Allows running asyncio in environments with an existing event loop in the notebook


client = OpenAI(api_key=os.getenv("OpenAI_API_Key"))

prompt_generation = """
Generate a table with the 5 most popular pharmaceutical companies and their foundation years.
The response should include only the table, with no additional text.
Use the following example format:
---
Company | Foundation Year
Microsoft | 1975
Apple | 1976
Google | 1998
Amazon | 1994
Meta | 2004
---"""

response = client.responses.create(
    model = "gpt-5",
    instructions = "You are a helpful assistant.",
    input = prompt_generation,
    reasoning = {
        'effort': 'minimal'
    }
)

generated_response = response.output_text
print("Generated Response:\n", generated_response)

badly_formatted_table = """
Company | Foundation Year

Microsoft | 1975

Apple | 1976
Google | 1998
Amazon | 1994

Meta | 2004
"""


prompt_check_table_new = """
Your task is to verify if a given table matches the exact format and structure of a provided example table.

Here's an example of the format that the table should have:
---
Company | Foundation Year
Microsoft | 1975
Apple | 1976
Google | 1998
Amazon | 1994
Meta | 2004
---

Table to Check:
{table_to_check}

Instructions:
- The table to check should match the structure, headers, and format of the Example Table exactly.
- The column names must be "Company" and "Foundation Year".
- The values in each row should have the company names and their corresponding foundation years.
- If the given table matches the example table in all these aspects, write "Yes".
- Write "No" if there are any differences in structure, headers, or if any company/year is missing or incorrect.

Only respond with "Yes" or "No".
"""

prompt_check_company_type = """
Your task is to verify if a given table contains only companies from the pharmaceutical industry.

Here's an example of the format that the table should have:
---
Company | Foundation Year
Microsoft | 1975
Apple | 1976
Google | 1998
Amazon | 1994
Meta | 2004
---

Here's the table to check:
{table_to_check}

Instructions:
- The table to check should include only companies from the pharmaceutical industry.
- The column names will be "Company" and "Foundation Year".
- Evaluate whether each company in the table is part of the pharmaceutical sector.
- If all companies in the table are from the pharmaceutical industry, write "Yes".
- Write "No" if any company in the table is not from the pharmaceutical industry.

Only respond with "Yes" or "No".

"""

formatted_input = prompt_check_company_type.format(table_to_check=generated_response)
print(f"Formatted Text: {formatted_input}")


### LLM as Judge

judge_response = client.responses.create(
    model="gpt-5",
    instructions="You are a strict judge. Evaluate inputs based on the given criteria and provide only the required response",
    input=formatted_input,
    reasoning={
        "effort": "minimal"
    }
)
print(f"LLM Judge Response: {judge_response.output_text}")