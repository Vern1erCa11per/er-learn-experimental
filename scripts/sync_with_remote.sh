#!/usr/bin/env bash
rsync -arv --exclude="data/pretrained/*" --exclude="test/performance/*" --exclude=".git"  ../neural-linkage/ ec2-user@aws_ml:~/neural-linkage/neural-linkage/
