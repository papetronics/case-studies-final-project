#!/bin/bash
aws ecr get-login-password --region us-east-2 \
| docker login --username AWS --password-stdin 367942341054.dkr.ecr.us-east-2.amazonaws.com