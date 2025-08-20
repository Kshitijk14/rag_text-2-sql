import io 
import os
import time
import re
import requests
import zipfile
import shutil
import gc
import traceback
import json
import json as pyjson

from pathlib import Path

from typing import List, Dict
from pydantic import BaseModel, Field

import pandas as pd

# setup Arize Phoenix for logging/observability
import phoenix as px

import chromadb