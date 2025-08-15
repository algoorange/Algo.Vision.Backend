from pymongo import MongoClient
import os
import json

client = MongoClient(os.getenv("MONGO_URI"))
db = client["algo_compliance_db_2"]
video_details_segment = db["video_details_segment"]

import re
import json

class GenericResponceToolService:
    def __init__(self, request):
        self.request = request
        self.db = request.app.db if request and hasattr(request.app, 'db') and hasattr(request.app, 'db') else db
