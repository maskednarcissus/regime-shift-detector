# Databricks notebook source
import os
import sys

root = os.getcwd()
src = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from regime_detection.stages import run_normal_condition_model

run_normal_condition_model()

