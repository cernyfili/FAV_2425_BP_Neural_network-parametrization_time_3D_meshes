#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: test_script.py
Author: Filip Cerny
Created: 12.04.2025
Version: 1.0
Description: 
"""
import json

from scripts.evaluate_nn_model import TRAIN_CONFIG
from utils.constants import DEFAULT_TRAIN_CONFIG, TrainConfig, FilePathConfig

#data = DEFAULT_TRAIN_CONFIG.to_dict()

filepath = "config.json"
#
# with open(filepath, "w") as f:
#     json.dump(data, f, indent=4)
#

# validate json with json schema

schema = TrainConfig.json_schema()


with open(filepath, "r") as f:
    data = json.load(f)

from jsonschema import validate, ValidationError

try:
    validate(instance=data, schema=schema)
    print("JSON is valid")
except ValidationError as e:
    print("JSON is invalid")
    print(e.message)
    #exit
    exit(1)

# save json to object
fileconfig : FilePathConfig = FilePathConfig.create_test_mode(data_foldername="test", processed_session_folderpath="test",raw_folderpath="test", processed_folderpath="test")
train_config = TrainConfig.from_dict(data, fileconfig)

print(train_config)


