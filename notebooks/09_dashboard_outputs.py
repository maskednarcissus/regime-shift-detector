# Databricks notebook source
import os
import sys

root = os.getcwd()
src = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from regime_detection.stages import build_dashboard_outputs

build_dashboard_outputs()

