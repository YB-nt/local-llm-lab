#!/usr/bin/env bash
# 개인정보 없는 머신 라벨 생성: apple-m4-16gb 형태
chip=$(sysctl -n machdep.cpu.brand_string | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
ram=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "${chip}-${ram}gb"
