# Databricks notebook source
import os
import sys

root = os.getcwd()
src = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from regime_detection.stages import emit_operational_alerts_stage

emit_operational_alerts_stage()
