#!/bin/bash

# 查找所有 NVIDIA GPU 设备的设备ID及其类型
device_info=$(lspci | grep -i nvidia | egrep "VGA compatible controller|3D controller" | awk '{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20}')

# 通过换行符分隔设备信息并遍历
echo "$device_info" | while IFS= read -r line; do
    id=$(echo $line | awk '{print $1}')
    type=$(echo $line | cut -d ' ' -f 2-)
    # 使用 lspci -n 获取特定设备的硬件信息
    device_info=$(lspci -n | grep -i $id | awk '{print $3}')

    echo "设备编号: $id"
    echo "设备类型: $type"
    echo "设备信息: $device_info"
    # 提取并打印 LnkCap 和 LnkSta 信息
    echo "带宽信息:"
    sudo lspci -n -d $device_info -vvv | grep -i width
    echo
done