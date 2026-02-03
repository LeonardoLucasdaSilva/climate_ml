from dotenv import load_dotenv
import os

load_dotenv()

print("Project name: ", os.getenv("PROJECT_NAME"))
print("DATA OWNER: ", os.getenv("DATA_OWNER"))