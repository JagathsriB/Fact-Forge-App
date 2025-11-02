import google.generativeai as genai
genai.configure(api_key="AIzaSyCyiaZgeZy6zm6ZcH8qWHljDMHqsTTspJw")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
